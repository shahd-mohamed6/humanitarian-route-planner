"""Admissible and consistent heuristics for A*.

Idea (works for every objective):
    k = min over all legs of  weight(leg) / straight_line_km(leg)
    h(n) = k * straight_line_km(n, goal)

Why it is CONSISTENT (and therefore admissible):
    For any leg u->v:  weight(u,v) >= k * d(u,v)          (definition of k)
    Triangle inequality: d(u,goal) <= d(u,v) + d(v,goal)
    =>  h(u) = k*d(u,goal) <= k*d(u,v) + k*d(v,goal) <= weight(u,v) + h(v)
    and h(goal) = 0. Consistent + h(goal)=0  =>  never overestimates.

For the 'time' objective, 1/k is simply the fastest effective speed (km/h)
anywhere in the network, so h(n) = "time if we flew straight to Gaza at the
best speed ever observed" - an optimistic guess, as A* requires.
"""
from .network import node_distance_km


def ratio_constant(net, weight):
    """k = min weight / straight-line distance over legs with distance > 0."""
    k = float("inf")
    for leg in net.legs.values():
        d = node_distance_km(net, leg.u, leg.v)
        if d > 1e-6:
            k = min(k, weight(leg) / d)
    return 0.0 if k == float("inf") else max(k, 0.0)


def make_heuristic(net, weight, goal):
    """Return (h, k) where h(node_id) is an admissible estimate to `goal`."""
    k = ratio_constant(net, weight)
    cache = {}

    def h(node_id):
        if node_id not in cache:
            cache[node_id] = k * node_distance_km(net, node_id, goal)
        return cache[node_id]

    return h, k


def zero_heuristic(_node_id):
    """h = 0 turns A* into Dijkstra / uniform-cost search."""
    return 0.0
