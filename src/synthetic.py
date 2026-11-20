"""Generate larger, realistic-looking synthetic networks for scaling experiments.

The real dataset has only 28 nodes, which is too small to show how the
algorithms scale. This generator builds networks with the SAME structure:

    - an origin in the east (Dubai-like) and a destination in the west (Gaza-like)
    - many tier-1 "international" nodes spread over the region
    - a small tier-2 "last-mile" cluster around the destination
    - a few entry gateways joining the two tiers
    - legs with mode-dependent speed, cost and reliability

Everything is driven by a random seed, so experiments are reproducible.
"""
import math
import random

from .network import Network, Node, Leg, haversine_km

MODES = {
    #            speed km/h, handling h, USD per t per km, reliability range
    "Road":      (55, 3.0, 0.16, (0.80, 0.97)),
    "Air":       (650, 8.0, 1.00, (0.88, 0.98)),
    "Sea":       (28, 24.0, 0.06, (0.65, 0.95)),
}


def _make_leg(lid, a, b, mode, rng):
    speed, handling, usd_km, (r_lo, r_hi) = MODES[mode]
    d = haversine_km(a.lat, a.lon, b.lat, b.lon) * rng.uniform(1.05, 1.35)  # detour
    t = d / speed + handling
    return Leg(id=lid, u=a.id, v=b.id, mode=mode, distance_km=round(d, 1),
               time_h=round(t, 2), time_p90_h=round(t * rng.uniform(1.1, 1.6), 2),
               cost_usd_t=round(d * usd_km + rng.uniform(10, 60), 1),
               capacity_t=rng.choice([60, 150, 300, 600, 1000]),
               risk_score=rng.randint(1, 4),
               reliability=round(rng.uniform(r_lo, r_hi), 3),
               corridor=f"SYN-{mode}")


def generate_network(n_outer=100, n_inner=10, n_gateways=3, k=4, seed=42):
    """Return (network, start_id, goal_id).

    n_outer    : number of tier-1 (international) nodes, incl. the origin
    n_inner    : number of tier-2 (last-mile) nodes, incl. gateways and goal
    n_gateways : how many inner nodes are entry gateways
    k          : each node is linked to its k nearest neighbours (both ways)
    """
    rng = random.Random(seed)
    nodes = [Node("S", "Synthetic origin", "Warehouse", "SYN", 25.0, 55.1, 1, "Origin", False)]
    for i in range(1, n_outer):
        nodes.append(Node(f"O{i}", f"Outer {i}", "Hub", "SYN",
                          rng.uniform(22.0, 36.0), rng.uniform(31.0, 56.0),
                          1, "Transfer point", False))
    goal = Node("G", "Synthetic destination", "Destination", "SYN", 31.50, 34.47,
                2, "Destination", False)
    inner = [goal]
    for i in range(1, n_inner):
        is_gw = i <= n_gateways
        inner.append(Node(f"I{i}", f"Inner {i}", "Border crossing" if is_gw else "Hub",
                          "SYN", 31.50 + rng.uniform(-0.35, 0.35),
                          34.47 + rng.uniform(-0.35, 0.35), 2,
                          "Entry crossing" if is_gw else "Alternative destination",
                          is_gw))
    gateways = [n for n in inner if n.is_gateway]

    legs, ids = [], iter(range(1, 10**9))

    def link(a, b, mode=None):
        if mode is None:
            d = haversine_km(a.lat, a.lon, b.lat, b.lon)
            mode = "Road" if d < 350 else rng.choice(["Air", "Air", "Sea", "Road"])
        legs.append(_make_leg(f"L{next(ids)}", a, b, mode, rng))
        legs.append(_make_leg(f"L{next(ids)}", b, a, mode, rng))

    def knn_links(group, kk):
        pairs = set()
        for a in group:
            near = sorted((n for n in group if n.id != a.id),
                          key=lambda n: haversine_km(a.lat, a.lon, n.lat, n.lon))[:kk]
            for b in near:
                pairs.add(tuple(sorted((a.id, b.id))))
        return pairs

    by_id = {n.id: n for n in nodes + inner}
    outer_pairs = knn_links(nodes, k)
    _connect_components(nodes, outer_pairs)
    for a, b in outer_pairs:
        link(by_id[a], by_id[b])

    inner_pairs = knn_links(inner, min(3, len(inner) - 1))
    _connect_components(inner, inner_pairs)
    for a, b in inner_pairs:
        link(by_id[a], by_id[b], "Road")

    for gw in gateways:                      # gateways reachable from outer tier
        near = sorted(nodes, key=lambda n: haversine_km(gw.lat, gw.lon, n.lat, n.lon))[:3]
        for o in near:
            legs.append(_make_leg(f"L{next(ids)}", o, gw, "Road", rng))

    return Network(nodes + inner, legs), "S", "G"


def _connect_components(group, pairs):
    """Add the shortest bridging pairs until the undirected graph is connected."""
    parent = {n.id: n.id for n in group}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    while len({find(n.id) for n in group}) > 1:
        best = None
        for a in group:
            for b in group:
                if find(a.id) != find(b.id):
                    d = haversine_km(a.lat, a.lon, b.lat, b.lon)
                    if best is None or d < best[0]:
                        best = (d, a.id, b.id)
        _, a, b = best
        pairs.add(tuple(sorted((a, b))))
        parent[find(a)] = find(b)
