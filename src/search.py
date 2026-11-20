"""Search algorithms, written from scratch.

All algorithms share one best-first search loop; they differ only in how the
priority f(n) of a node is computed:

    Dijkstra / UCS     f(n) = g(n)                 optimal, explores widely
    A*                 f(n) = g(n) + h(n)          optimal if h is consistent
    Weighted A*        f(n) = g(n) + w*h(n), w>1   faster, cost <= w * optimal
    Greedy best-first  f(n) = h(n)                 fast, NOT optimal

g(n) = cost of the best path found so far from start to n
h(n) = heuristic estimate of the remaining cost from n to the goal

Plus a two-tier (hierarchical) A*: international tier first, then the Gaza
last-mile tier, joined at the entry gateways (see two_tier_astar).
"""
from dataclasses import dataclass, field
import heapq
import itertools
import time

from .heuristics import zero_heuristic


@dataclass
class SearchResult:
    algorithm: str
    found: bool
    path: list = field(default_factory=list)       # node ids
    legs: list = field(default_factory=list)       # Leg objects
    cost: float = float("inf")                     # objective value
    nodes_expanded: int = 0                        # nodes popped & expanded
    nodes_generated: int = 0                       # nodes pushed on the frontier
    runtime_ms: float = 0.0
    expanded_order: list = field(default_factory=list)  # for visualisation


def _best_first(net, start, goal, weight, priority, min_capacity, name):
    """Generic best-first graph search.

    priority(g, node) -> f value used to order the frontier (min-heap).
    Uses lazy deletion: a node can be pushed more than once; stale entries
    are skipped when popped.
    """
    t0 = time.perf_counter()
    tie = itertools.count()                        # FIFO tie-breaker
    g = {start: 0.0}
    parent = {start: (None, None)}                 # node -> (prev node, leg)
    frontier = [(priority(0.0, start), next(tie), start)]
    closed = set()
    expanded_order = []
    generated = 1

    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node in closed:
            continue                               # stale entry
        if node == goal:
            path, legs = _reconstruct(parent, goal)
            return SearchResult(name, True, path, legs, g[goal], len(closed),
                                generated, (time.perf_counter() - t0) * 1000,
                                expanded_order)
        closed.add(node)
        expanded_order.append(node)
        for leg in net.neighbours(node, min_capacity):
            new_g = g[node] + weight(leg)
            if leg.v not in closed and new_g < g.get(leg.v, float("inf")):
                g[leg.v] = new_g
                parent[leg.v] = (node, leg)
                heapq.heappush(frontier, (priority(new_g, leg.v), next(tie), leg.v))
                generated += 1

    return SearchResult(name, False, nodes_expanded=len(closed),
                        nodes_generated=generated,
                        runtime_ms=(time.perf_counter() - t0) * 1000,
                        expanded_order=expanded_order)


def _reconstruct(parent, goal):
    path, legs, node = [], [], goal
    while node is not None:
        path.append(node)
        prev, leg = parent[node]
        if leg is not None:
            legs.append(leg)
        node = prev
    return path[::-1], legs[::-1]


# ------------------------------------------------------------------ public API
def dijkstra(net, start, goal, weight, min_capacity=0.0):
    return _best_first(net, start, goal, weight,
                       lambda g, n: g, min_capacity, "Dijkstra")


def astar(net, start, goal, weight, h, min_capacity=0.0):
    return _best_first(net, start, goal, weight,
                       lambda g, n: g + h(n), min_capacity, "A*")


def weighted_astar(net, start, goal, weight, h, w=1.5, min_capacity=0.0):
    return _best_first(net, start, goal, weight,
                       lambda g, n: g + w * h(n), min_capacity, f"Weighted A* (w={w})")


def greedy_best_first(net, start, goal, weight, h, min_capacity=0.0):
    return _best_first(net, start, goal, weight,
                       lambda g, n: h(n), min_capacity, "Greedy best-first")


# ------------------------------------------------------------ two-tier search
def lastmile_table(net, goal, weight, inner, min_capacity=0.0):
    """Tier 2: exact cost from every inner (Gaza) node to the goal.

    Runs Dijkstra BACKWARDS from the goal over legs whose both ends are in
    `inner`. Returns (dist, next_leg, expansions). The table only depends on
    the last-mile legs, so it can be cached and reused when a disruption only
    affects the international tier.
    """
    rev = {n: [] for n in inner}
    for leg in net.legs.values():
        if leg.u in inner and leg.v in inner and leg.capacity_t >= min_capacity:
            rev[leg.v].append(leg)
    dist, nxt = {goal: 0.0}, {goal: None}
    heap, tie, closed = [(0.0, 0, goal)], itertools.count(1), set()
    while heap:
        d, _, node = heapq.heappop(heap)
        if node in closed:
            continue
        closed.add(node)
        for leg in rev.get(node, []):
            nd = d + weight(leg)
            if nd < dist.get(leg.u, float("inf")):
                dist[leg.u], nxt[leg.u] = nd, leg
                heapq.heappush(heap, (nd, next(tie), leg.u))
    return dist, nxt, len(closed)


def two_tier_astar(net, start, goal, weight, h, min_capacity=0.0,
                   inner=None, gateways=None, table=None):
    """Hierarchical A*.

    Tier 1: A* over the international network (outside Gaza). It is not
            allowed to wander inside Gaza; instead, reaching an entry gateway
            g creates a virtual edge g -> GOAL whose cost is the exact
            last-mile cost from the tier-2 table.
    Tier 2: the last-mile table (backward Dijkstra inside Gaza).

    Because the virtual edges carry EXACT last-mile costs and h stays
    consistent, the combined route is optimal (same cost as Dijkstra).
    """
    if start == goal:
        return SearchResult("Two-tier A*", True, [start], [], 0.0, 0, 1, 0.0, [])
    t0 = time.perf_counter()
    inner = inner if inner is not None else net.gaza_inner_nodes()
    if start in inner or goal not in inner:
        # Hierarchy does not apply (e.g. starting inside Gaza): plain A*.
        r = astar(net, start, goal, weight, h, min_capacity)
        r.algorithm = "Two-tier A*"
        return r
    gateways = gateways if gateways is not None else net.gateways()
    tier2_exp = 0
    if table is None:
        table = lastmile_table(net, goal, weight, inner, min_capacity)
        tier2_exp = table[2]
    dist, nxt, _ = table

    VGOAL = "__GOAL__"
    tie = itertools.count()
    g = {start: 0.0}
    parent = {start: (None, None)}
    frontier = [(h(start), next(tie), start)]
    closed, expanded_order, generated = set(), [], 1

    while frontier:
        _, _, node = heapq.heappop(frontier)
        if node in closed:
            continue
        if node == VGOAL:
            gw = parent[VGOAL][0]
            path, legs = _reconstruct(parent, gw)
            cur = gw
            while nxt.get(cur) is not None:          # append last-mile legs
                leg = nxt[cur]
                legs.append(leg)
                path.append(leg.v)
                cur = leg.v
            return SearchResult("Two-tier A*", True, path, legs, g[VGOAL],
                                len(closed) + tier2_exp, generated,
                                (time.perf_counter() - t0) * 1000, expanded_order)
        closed.add(node)
        expanded_order.append(node)

        if node in gateways and node in dist:        # hand over to tier 2
            new_g = g[node] + dist[node]
            if new_g < g.get(VGOAL, float("inf")):
                g[VGOAL] = new_g
                parent[VGOAL] = (node, None)
                heapq.heappush(frontier, (new_g, next(tie), VGOAL))
                generated += 1
            continue                                  # don't explore inside Gaza

        for leg in net.neighbours(node, min_capacity):
            if leg.v in inner and leg.v not in gateways:
                continue                              # tier-1 stays outside Gaza
            new_g = g[node] + weight(leg)
            if leg.v not in closed and new_g < g.get(leg.v, float("inf")):
                g[leg.v] = new_g
                parent[leg.v] = (node, leg)
                heapq.heappush(frontier, (new_g + h(leg.v), next(tie), leg.v))
                generated += 1

    return SearchResult("Two-tier A*", False, nodes_expanded=len(closed) + tier2_exp,
                        nodes_generated=generated,
                        runtime_ms=(time.perf_counter() - t0) * 1000,
                        expanded_order=expanded_order)
