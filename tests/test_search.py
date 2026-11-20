import pytest

from src.data_loader import load_dataset
from src.costs import CostModel
from src.network import Network
from src.planner import plan
from src.costs import OBJECTIVES
from src.data_loader import load_config
from src.scenarios import apply_scenario
import networkx as nx
from src.heuristics import make_heuristic
from src.search import dijkstra
from src.scenarios import apply_scenario, close_nodes
from src.synthetic import generate_network

@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_dataset_counts(dataset):
    """Check that all 28 locations and 45 transport legs are loaded."""
    assert len(dataset["locations"]) == 28
    assert len(dataset["legs"]) == 45

def test_leg_endpoints_exist(dataset):
    """Check that every leg starts and ends at a known location."""
    location_ids = set(dataset["locations"]["node_id"])

    for leg in dataset["legs"].itertuples():
        assert leg.from_node in location_ids, f"Unknown start on {leg.leg_id}"
        assert leg.to_node in location_ids, f"Unknown end on {leg.leg_id}"

@pytest.fixture(scope="module")
def network(dataset):
    return Network.from_frames(dataset["locations"], dataset["legs"])


def test_fastest_route_baseline(network):
    """Check that all three algorithms find the same fastest travel time."""
    time_model = CostModel.for_network(network, "time")
    algorithms = ("Dijkstra", "A*", "Two-tier A*")
    results = plan(network, "ORG", "GZC", time_model,
                   algorithms=algorithms, min_capacity=20)

    for name in algorithms:
        assert results[name].found, f"{name} could not find a route"

    fastest_time = results["Dijkstra"].cost
    assert results["A*"].cost == pytest.approx(fastest_time)
    assert results["Two-tier A*"].cost == pytest.approx(fastest_time)
@pytest.mark.parametrize("scenario_id", [f"S{i}" for i in range(11)])
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_optimal_cost_across_scenarios(network, dataset, scenario_id, objective):
    """Check that all three algorithms agree on the best cost for each case."""
    scenario_network, _ = apply_scenario(network, dataset["scenarios"], scenario_id)
    weights = load_config()["balanced_weights"]
    cost_model = CostModel.for_network(network, objective, weights)
    algorithms = ("Dijkstra", "A*", "Two-tier A*")

    results = plan(scenario_network, "ORG", "GZC", cost_model,
                   algorithms=algorithms, min_capacity=20)

    for name in algorithms:
        assert results[name].found, f"{name} found no route"

    best_cost = results["Dijkstra"].cost
    assert results["A*"].cost == pytest.approx(best_cost)
    assert results["Two-tier A*"].cost == pytest.approx(best_cost)

def test_dijkstra_matches_networkx(network):
    """Check the fastest baseline travel time against NetworkX."""
    graph = nx.DiGraph()
    graph.add_nodes_from(network.nodes)

    for leg in network.legs.values():
        if leg.capacity_t >= 20:
            graph.add_edge(leg.u, leg.v, hours=leg.time_h)

    expected_hours = nx.dijkstra_path_length(graph, "ORG", "GZC", weight="hours")

    time_model = CostModel.for_network(network, "time")
    results = plan(network, "ORG", "GZC", time_model,
                   algorithms=("Dijkstra",), min_capacity=20)

    assert results["Dijkstra"].found
    assert results["Dijkstra"].cost == pytest.approx(expected_hours)

def test_time_heuristic_is_admissible(network):
    """Check that A* never overestimates the remaining travel time."""
    goal = "GZC"
    time_model = CostModel.for_network(network, "time")
    estimate, _ = make_heuristic(network, time_model, goal)

    for start in network.nodes:
        best = dijkstra(network, start, goal, time_model, min_capacity=20)
        if best.found:
            assert estimate(start) <= best.cost + 1e-9, f"Estimate too high at {start}"

def test_time_heuristic_is_consistent(network):
    """Check that the time estimate is consistent across every directed leg."""
    time_model = CostModel.for_network(network, "time")
    estimate, _ = make_heuristic(network, time_model, "GZC")

    for leg in network.legs.values():
        upper_limit = time_model(leg) + estimate(leg.v)
        assert estimate(leg.u) <= upper_limit + 1e-9, f"Inconsistent at {leg.id}"

def test_rafah_closure(network, dataset):
    """Check that every algorithm finds a route without using a closed leg."""
    closed_network, _ = apply_scenario(network, dataset["scenarios"], "S1")
    time_model = CostModel.for_network(network, "time")
    results = plan(closed_network, "ORG", "GZC", time_model, min_capacity=20)

    closed_legs = {"E32", "E37"}
    for name, route in results.items():
        assert route.found, f"{name} could not find an alternative route"
        for leg in route.legs:
            assert leg.id not in closed_legs, f"{name} used closed leg {leg.id}"

def test_route_respects_capacity(network):
    """Check that every leg on each route can carry a 100-tonne shipment."""
    shipment = 100
    time_model = CostModel.for_network(network, "time")
    results = plan(network, "ORG", "GZC", time_model, min_capacity=shipment)

    for name, route in results.items():
        assert route.found, f"{name} could not find a route for {shipment} tonnes"
        for leg in route.legs:
            assert leg.capacity_t >= shipment, f"{name} used undersized leg {leg.id}"

def test_start_equals_goal(network):
    """Check that starting at the destination requires no travel."""
    time_model = CostModel.for_network(network, "time")
    results = plan(network, "GZC", "GZC", time_model, min_capacity=20)

    for name, route in results.items():
        assert route.found, f"{name} failed to recognize the destination"
        assert route.path == ["GZC"], f"{name} returned an unexpected path"
        assert route.cost == 0, f"{name} added travel time"
def test_destination_unreachable(network):
    """Check that every algorithm reports failure when all entry points close."""
    gateways = ["RAF", "KSH", "ZKM", "DZN", "PIER"]
    closed_network = close_nodes(network, gateways)
    time_model = CostModel.for_network(network, "time")
    results = plan(closed_network, "ORG", "GZC", time_model, min_capacity=20)

    for name, route in results.items():
        assert not route.found, f"{name} reported a route with all gateways closed"

def test_weighted_and_greedy_route_costs(network):
    """Compare weighted A* and greedy travel times with the optimal time."""
    time_model = CostModel.for_network(network, "time")
    algorithms = ("Dijkstra", "Weighted A*", "Greedy best-first")
    results = plan(network, "ORG", "GZC", time_model,
                   algorithms=algorithms, min_capacity=20, w=1.5)

    for name, route in results.items():
        assert route.found, f"{name} could not find a route"

    optimal = results["Dijkstra"].cost
    weighted = results["Weighted A*"].cost
    greedy = results["Greedy best-first"].cost
    tolerance = 1e-9

    assert optimal - tolerance <= weighted <= 1.5 * optimal + tolerance
    assert greedy >= optimal - tolerance

@pytest.mark.parametrize("n_outer", [50, 200])
@pytest.mark.parametrize("seed", range(1, 11))
def test_astar_on_synthetic_networks(n_outer, seed):
    """Check A* travel time and the number of nodes it expands against Dijkstra."""
    net, start, goal = generate_network(n_outer=n_outer, seed=seed)
    time_model = CostModel.for_network(net, "time")
    results = plan(net, start, goal, time_model, algorithms=("Dijkstra", "A*"))

    baseline = results["Dijkstra"]
    astar = results["A*"]

    assert baseline.found and astar.found
    assert astar.cost == pytest.approx(baseline.cost)
    assert astar.nodes_expanded <= baseline.nodes_expanded

def test_same_seed_reproduces_legs():
    """Check that the same seed generates identical transport legs."""
    first, _, _ = generate_network(n_outer=50, seed=7)
    second, _, _ = generate_network(n_outer=50, seed=7)

    assert first.legs == second.legs