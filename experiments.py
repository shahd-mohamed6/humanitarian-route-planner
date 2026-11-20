"""Repeatable experiments for the humanitarian route planner.

Run from the project root: python experiments.py
"""
from pathlib import Path
from time import perf_counter
import math

import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_config, load_dataset
from src.synthetic import generate_network
from src.network import Network
from src.costs import CostModel, OBJECTIVES
from src.scenarios import scenario_list, apply_scenario
from src.heuristics import make_heuristic
from src.search import astar, two_tier_astar, lastmile_table
from src.planner import ALGORITHMS, plan, result_row

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
REPEATS = 30


def experiment_scaling():
    """Compare all five algorithms on 30 seeded networks at each size."""
    config = load_config()
    rows = []
    for size in [50, 100, 200, 500, 1000, 2000]:
        print("Testing network size: {}".format(size))
        for i in range(REPEATS):
            seed = config["random_seed"] + i
            net, start, goal = generate_network(
                n_outer=size, n_inner=15, n_gateways=3, seed=seed
            )
            cost_model = CostModel.for_network(net, "time")
            results = plan(net, start, goal, cost_model,
                           min_capacity=0.0, w=config["weighted_astar_w"])
            baseline = results["Dijkstra"]
            for name, result in results.items():
                gap = None
                if result.found and baseline.found and baseline.cost > 0:
                    gap = 100 * (result.cost - baseline.cost) / baseline.cost
                rows.append({
                    "network_size": size + 15,
                    "n_outer": size,
                    "seed": seed,
                    "algorithm": name,
                    "nodes_expanded": result.nodes_expanded,
                    "nodes_generated": result.nodes_generated,
                    "runtime_ms": result.runtime_ms,
                    "cost": result.cost if result.found else None,
                    "optimality_gap_pct": gap,
                    "found": result.found
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "scaling_raw.csv", index=False)
    summary = df.groupby(
        ["network_size", "n_outer", "algorithm"], as_index=False
    ).agg(
        expanded_mean=("nodes_expanded", "mean"),
        expanded_std=("nodes_expanded", "std"),
        generated_mean=("nodes_generated", "mean"),
        generated_std=("nodes_generated", "std"),
        runtime_mean=("runtime_ms", "mean"),
        runtime_std=("runtime_ms", "std"),
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std"),
        gap_mean=("optimality_gap_pct", "mean"),
        gap_std=("optimality_gap_pct", "std")
    )
    summary.to_csv(RESULTS_DIR / "scaling_summary.csv", index=False)
    print("Scaling experiment completed! Saved {} raw rows.".format(len(df)))


def plot_scaling_charts():
    """Draw nodes-expanded, runtime and optimality-gap plots."""
    df = pd.read_csv(RESULTS_DIR / "scaling_summary.csv")
    for value_col, std_col, title, ylabel, filename in [
        ("expanded_mean", "expanded_std", "Nodes Expanded vs Network Size",
         "Mean Nodes Expanded", "nodes_expanded.png"),
        ("runtime_mean", "runtime_std", "Algorithm Runtime vs Network Size",
         "Mean Runtime (ms)", "runtime_scaling.png")
    ]:
        plt.figure(figsize=(10, 6))
        for algorithm in ALGORITHMS:
            data = df[df["algorithm"] == algorithm].sort_values("network_size")
            plt.errorbar(data["network_size"], data[value_col],
                         yerr=data[std_col], marker="o", capsize=3,
                         label=algorithm)
        plt.title(title)
        plt.xlabel("Network Size (Total Nodes)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / filename, dpi=200)
        plt.close()
        print("Saved {}".format(filename))

    sizes = sorted(df["network_size"].unique())
    x = list(range(len(sizes)))
    width = 0.35
    plt.figure(figsize=(10, 6))
    for i, algorithm in enumerate(["Weighted A*", "Greedy best-first"]):
        data = df[df["algorithm"] == algorithm].sort_values("network_size")
        positions = [v + (i - 0.5) * width for v in x]
        plt.bar(positions, data["gap_mean"], width=width, label=algorithm)
    plt.xticks(x, sizes)
    plt.title("Optimality Gap Compared with Dijkstra")
    plt.xlabel("Network Size (Total Nodes)")
    plt.ylabel("Mean Optimality Gap (%)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "optimality_gap.png", dpi=200)
    plt.close()
    print("Saved optimality_gap.png")


def experiment_real_network():
    """Run each objective/scenario 30 times and summarize runtime variation."""
    config = load_config()
    data = load_dataset()
    base = Network.from_frames(data["locations"], data["legs"])
    rows = []
    for objective in OBJECTIVES:
        # Base-network references preserve balanced-score comparability.
        cost_model = CostModel.for_network(base, objective,
                                           config["balanced_weights"])
        for scenario_id, scenario_name in scenario_list(data["scenarios"]):
            net, _ = apply_scenario(base, data["scenarios"], scenario_id)
            timings = {name: [] for name in ALGORITHMS}
            first_results = None
            print("Testing {} / {}".format(objective, scenario_id))
            for i in range(REPEATS):
                results = plan(net, config["origin"], config["destination"],
                               cost_model,
                               min_capacity=config["shipment_tonnes"],
                               w=config["weighted_astar_w"])
                if first_results is None:
                    first_results = results
                for name, result in results.items():
                    timings[name].append(result.runtime_ms)
            for name in ALGORITHMS:
                row = result_row(first_results[name], net)
                row.update({
                    "algorithm": name,
                    "objective": objective,
                    "scenario": scenario_id,
                    "scenario_name": scenario_name,
                    "repeats": REPEATS,
                    "runtime_mean_ms": pd.Series(timings[name]).mean(),
                    "runtime_std_ms": pd.Series(timings[name]).std()
                })
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "real_network_results.csv", index=False)
    print("Experiment 2 completed! Saved {} rows.".format(len(df)))


def experiment_replanning():
    """Compare fresh A* with hierarchical re-planning and cached last-mile paths.

    The reported hierarchical total includes preprocessing if a scenario
    changes Gaza legs; unchanged last-mile tables are reused without charge.
    """
    config = load_config()
    data = load_dataset()
    base = Network.from_frames(data["locations"], data["legs"])
    inner = base.gaza_inner_nodes()
    gateways = base.gateways()
    goal = config["destination"]
    start = config["origin"]
    capacity = config["shipment_tonnes"]
    cost_model = CostModel.for_network(base, "time", config["balanced_weights"])

    baseline_t0 = perf_counter()
    baseline_table = lastmile_table(base, goal, cost_model, inner,
                                    min_capacity=capacity)
    baseline_precompute_ms = (perf_counter() - baseline_t0) * 1000
    reuse_scenarios = {"S3", "S4", "S7", "S9", "S10"}
    rows = []

    for scenario_id, scenario_name in scenario_list(data["scenarios"]):
        if scenario_id == "S0":
            continue
        net, changes = apply_scenario(base, data["scenarios"], scenario_id)
        h, _ = make_heuristic(net, cost_model, goal)
        reuse = scenario_id in reuse_scenarios
        astar_times = []
        tier_times = []
        table_times = []
        total_tier_times = []
        a_first = None
        two_first = None
        table_expansions = 0
        print("Re-planning {} / {}".format(scenario_id,
                                            "reuse" if reuse else "recompute"))

        for i in range(REPEATS):
            a_result = astar(net, start, goal, cost_model, h,
                             min_capacity=capacity)
            if reuse:
                table = baseline_table
                table_ms = 0.0
                prep_expansions = 0
            else:
                t0 = perf_counter()
                table = lastmile_table(net, goal, cost_model, inner,
                                       min_capacity=capacity)
                table_ms = (perf_counter() - t0) * 1000
                prep_expansions = table[2]

            two_result = two_tier_astar(
                net, start, goal, cost_model, h,
                min_capacity=capacity, inner=inner, gateways=gateways,
                table=table
            )
            if a_result.found != two_result.found:
                raise AssertionError("Route found mismatch in {}".format(scenario_id))
            if a_result.found and not math.isclose(
                a_result.cost, two_result.cost, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise AssertionError("Route cost mismatch in {}: {} vs {}".format(
                    scenario_id, a_result.cost, two_result.cost
                ))
            if a_first is None:
                a_first = a_result
                two_first = two_result
                table_expansions = prep_expansions
            astar_times.append(a_result.runtime_ms)
            tier_times.append(two_result.runtime_ms)
            table_times.append(table_ms)
            total_tier_times.append(two_result.runtime_ms + table_ms)

        rows.append({
            "scenario": scenario_id,
            "scenario_name": scenario_name,
            "table_action": "reused" if reuse else "recomputed",
            "repeats": REPEATS,
            "a_star_found": a_first.found,
            "two_tier_found": two_first.found,
            "a_star_cost": a_first.cost if a_first.found else None,
            "two_tier_cost": two_first.cost if two_first.found else None,
            "cost_match": True,
            "a_star_route": " > ".join(a_first.path),
            "two_tier_route": " > ".join(two_first.path),
            "a_star_nodes_expanded": a_first.nodes_expanded,
            "two_tier_search_nodes_expanded": two_first.nodes_expanded,
            "table_nodes_expanded": table_expansions,
            "two_tier_total_nodes_expanded": (two_first.nodes_expanded
                                              + table_expansions),
            "a_star_runtime_mean_ms": pd.Series(astar_times).mean(),
            "a_star_runtime_std_ms": pd.Series(astar_times).std(),
            "two_tier_search_runtime_mean_ms": pd.Series(tier_times).mean(),
            "table_runtime_mean_ms": pd.Series(table_times).mean(),
            "two_tier_total_runtime_mean_ms": pd.Series(total_tier_times).mean(),
            "two_tier_total_runtime_std_ms": pd.Series(total_tier_times).std(),
            "baseline_table_precompute_ms": baseline_precompute_ms
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "replanning_results.csv", index=False)
    print("Experiment 3 completed! Saved {} rows.".format(len(df)))
    print(df[["scenario", "table_action", "a_star_nodes_expanded",
              "two_tier_total_nodes_expanded", "cost_match"]].to_string(index=False))


def plot_delivery_time():
    """Plot Dijkstra's fastest delivery-time baseline for S0 to S10."""
    df = pd.read_csv(RESULTS_DIR / "real_network_results.csv")
    rows = df[(df["objective"] == "time") &
              (df["algorithm"] == "Dijkstra")].copy()
    rows["scenario_order"] = rows["scenario"].str[1:].astype(int)
    rows = rows.sort_values("scenario_order")
    plt.figure(figsize=(11, 6))
    plt.bar(rows["scenario"], rows["time_days"])
    plt.title("Fastest Delivery Time by Disruption Scenario")
    plt.xlabel("Disruption Scenario")
    plt.ylabel("Delivery Time (days)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "delivery_time_scenarios.png", dpi=200)
    plt.close()
    print("Saved delivery_time_scenarios.png")


if __name__ == "__main__":
    if not (RESULTS_DIR / "scaling_summary.csv").exists():
        experiment_scaling()
    plot_scaling_charts()
    if not (RESULTS_DIR / "real_network_results.csv").exists():
        experiment_real_network()
    experiment_replanning()
    plot_delivery_time()
    print("\nAll experiments and four charts completed!")
