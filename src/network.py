"""Graph representation of the multimodal humanitarian network.

Node = a location (warehouse, airport, port, border crossing, hub).
Leg  = a directed edge: one hop from one node to another by one transport mode.

We use our own light adjacency-list structure (instead of networkx) so that the
search algorithms in search.py are written from scratch and can count exactly
how many nodes they expand.
"""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    node_type: str
    country: str
    lat: float
    lon: float
    tier: int
    role: str
    is_gateway: bool


@dataclass(frozen=True)
class Leg:
    id: str
    u: str                 # from node id
    v: str                 # to node id
    mode: str              # Air, Sea, Road, Road (transfer), Airdrop
    distance_km: float
    time_h: float          # typical transit hours (incl. handling/inspection)
    time_p90_h: float      # 90th percentile hours (bad but plausible day)
    cost_usd_t: float      # USD per tonne
    capacity_t: float      # tonnes per week
    risk_score: int        # 1 (very low) .. 5 (very high)
    reliability: float     # 0..1 probability the movement completes on schedule
    corridor: str


class Network:
    """Directed multimodal network with an adjacency list."""

    def __init__(self, nodes, legs):
        self.nodes = {n.id: n for n in nodes}
        self.legs = {l.id: l for l in legs}
        self.adj = {nid: [] for nid in self.nodes}
        for leg in self.legs.values():
            self.adj[leg.u].append(leg)

    # ---------- construction ----------
    @classmethod
    def from_frames(cls, locations, legs):
        """Build a Network from the dataset DataFrames."""
        nodes = [
            Node(r.node_id, r.name, r.node_type, r.country, float(r.lat),
                 float(r.lon), int(r.tier), r.role, bool(r.is_entry_gateway))
            for r in locations.itertuples(index=False)
        ]
        leg_objs = [
            Leg(r.leg_id, r.from_node, r.to_node, r.mode, float(r.distance_km),
                float(r.transit_hours_typical), float(r.transit_hours_p90),
                float(r.cost_usd_per_tonne), float(r.capacity_tonnes_per_week),
                int(r.risk_score_1to5), float(r.reliability_pct) / 100.0,
                r.corridor_group)
            for r in legs.itertuples(index=False)
        ]
        return cls(nodes, leg_objs)

    # ---------- queries ----------
    def neighbours(self, node_id, min_capacity=0.0):
        """Outgoing legs from node_id that can carry at least min_capacity tonnes.

        This is where the capacity constraint is enforced during search.
        """
        return [l for l in self.adj.get(node_id, []) if l.capacity_t >= min_capacity]

    def gaza_inner_nodes(self):
        """Tier-2 last-mile nodes: entry gateways plus Gaza distribution points."""
        return {n.id for n in self.nodes.values()
                if n.is_gateway or n.role in ("Destination", "Alternative destination")}

    def gateways(self):
        return {n.id for n in self.nodes.values() if n.is_gateway}

    # ---------- modification (always returns a NEW network) ----------
    def without_legs(self, leg_ids):
        leg_ids = set(leg_ids)
        return Network(self.nodes.values(),
                       [l for l in self.legs.values() if l.id not in leg_ids])

    def with_multiplier(self, leg_ids, time_mult=1.0, cost_mult=1.0):
        leg_ids = set(leg_ids)
        new_legs = []
        for l in self.legs.values():
            if l.id in leg_ids:
                l = replace(l, time_h=l.time_h * time_mult,
                            time_p90_h=l.time_p90_h * time_mult,
                            cost_usd_t=l.cost_usd_t * cost_mult)
            new_legs.append(l)
        return Network(self.nodes.values(), new_legs)

    def __len__(self):
        return len(self.nodes)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle (straight-line on the globe) distance in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def node_distance_km(net, a, b):
    na, nb = net.nodes[a], net.nodes[b]
    return haversine_km(na.lat, na.lon, nb.lat, nb.lon)
