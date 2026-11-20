"""High-level planner: one call that prepares cost model + heuristic and runs
the chosen algorithms. Used by the app, the demo script and the experiments.
"""
from .costs import CostModel, route_summary
from .heuristics import make_heuristic
from .search import (dijkstra, astar, weighted_astar, greedy_best_first,
                     two_tier_astar)

ALGORITHMS = ("Dijkstra", "A*", "Two-tier A*", "Weighted A*", "Greedy best-first")


def plan(net, start, goal, cost_model, algorithms=ALGORITHMS, min_capacity=0.0,
         w=1.5, inner=None, gateways=None):
    """Run each algorithm; return {name: SearchResult}."""
    h, _ = make_heuristic(net, cost_model, goal)
    out = {}
    for name in algorithms:
        if name == "Dijkstra":
            r = dijkstra(net, start, goal, cost_model, min_capacity)
        elif name == "A*":
            r = astar(net, start, goal, cost_model, h, min_capacity)
        elif name == "Weighted A*":
            r = weighted_astar(net, start, goal, cost_model, h, w, min_capacity)
        elif name == "Greedy best-first":
            r = greedy_best_first(net, start, goal, cost_model, h, min_capacity)
        elif name == "Two-tier A*":
            r = two_tier_astar(net, start, goal, cost_model, h, min_capacity,
                               inner, gateways)
        else:
            raise ValueError(f"Unknown algorithm {name}")
        out[name] = r
    return out


def result_row(result, net):
    """Flatten a SearchResult into a dict for tables / CSV."""
    row = {
        "algorithm": result.algorithm,
        "found": result.found,
        "objective_cost": round(result.cost, 3) if result.found else None,
        "nodes_expanded": result.nodes_expanded,
        "nodes_generated": result.nodes_generated,
        "runtime_ms": round(result.runtime_ms, 3),
        "route": " > ".join(result.path) if result.found else "NO ROUTE",
    }
    if result.found:
        row.update(route_summary(result.legs))
    return row


__all__ = ["plan", "result_row", "ALGORITHMS", "CostModel"]
