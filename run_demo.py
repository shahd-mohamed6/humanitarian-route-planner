"""Command-line demo: plan Dubai -> Gaza City under every disruption scenario.

Usage:
    python run_demo.py                 # time objective, all scenarios
    python run_demo.py --objective cost
    python run_demo.py --objective balanced --tonnes 50

Saves results/scenario_results.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from src.costs import CostModel, OBJECTIVES
from src.data_loader import load_config, load_dataset
from src.network import Network
from src.planner import plan, result_row
from src.scenarios import apply_scenario, scenario_list


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", choices=OBJECTIVES, default=cfg["default_objective"])
    ap.add_argument("--tonnes", type=float, default=cfg["shipment_tonnes"])
    ap.add_argument("--all-objectives", action="store_true",
                    help="run every objective (used for the report table)")
    args = ap.parse_args()

    data = load_dataset()
    base = Network.from_frames(data["locations"], data["legs"])
    objectives = OBJECTIVES if args.all_objectives else [args.objective]

    rows = []
    for obj in objectives:
        cm = CostModel.for_network(base, obj, cfg["balanced_weights"])
        print(f"\n=== Objective: {obj} | shipment {args.tonnes} t ===")
        for sid, name in scenario_list(data["scenarios"]):
            net, _ = apply_scenario(base, data["scenarios"], sid)
            results = plan(net, cfg["origin"], cfg["destination"], cm,
                           min_capacity=args.tonnes, w=cfg["weighted_astar_w"])
            best = results["Dijkstra"]
            print(f"{sid:>4} {name:<32} ", end="")
            if best.found:
                r = result_row(best, net)
                print(f"{r['time_days']:>5} days  ${r['cost_usd_t']:>6.0f}/t  "
                      f"P(success)={r['success_prob']:.2f}  {r['route']}")
            else:
                print("NO FEASIBLE ROUTE")
            for res in results.values():
                rows.append({"objective": obj, "scenario": sid,
                             "scenario_name": name, **result_row(res, net)})

    out = Path("results") / "scenario_results.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
