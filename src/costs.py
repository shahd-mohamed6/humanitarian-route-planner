"""Cost models: turn a leg into a single number the search can minimise.

Objectives
----------
time      : typical transit hours
robust    : 90th percentile hours (plan for a bad day)
cost      : USD per tonne
risk      : -ln(reliability). Adding these along a route is the same as
            multiplying reliabilities, so minimising the sum MAXIMISES the
            probability that the whole route completes on schedule.
balanced  : weighted sum of time, cost and risk, each divided by a reference
            value (median over the base network) so the units cancel out.
"""
import math
import statistics

OBJECTIVES = ("time", "robust", "cost", "risk", "balanced")


def leg_risk(leg):
    return -math.log(max(leg.reliability, 1e-9))


class CostModel:
    def __init__(self, objective="time", weights=None, refs=None):
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective '{objective}'. Use one of {OBJECTIVES}")
        self.objective = objective
        self.weights = weights or {"time": 0.4, "cost": 0.3, "risk": 0.3}
        self.refs = refs or {"time": 1.0, "cost": 1.0, "risk": 1.0}

    @classmethod
    def for_network(cls, net, objective="time", weights=None):
        """Create a model whose normalisation references come from `net`.

        Pass the BASE (undisrupted) network so that scores stay comparable
        across disruption scenarios.
        """
        legs = list(net.legs.values())
        refs = {
            "time": statistics.median(l.time_h for l in legs),
            "cost": statistics.median(l.cost_usd_t for l in legs),
            "risk": statistics.median(leg_risk(l) for l in legs),
        }
        return cls(objective, weights, refs)

    def __call__(self, leg):
        o = self.objective
        if o == "time":
            return leg.time_h
        if o == "robust":
            return leg.time_p90_h
        if o == "cost":
            return leg.cost_usd_t
        if o == "risk":
            return leg_risk(leg)
        w, r = self.weights, self.refs
        return (w["time"] * leg.time_h / r["time"]
                + w["cost"] * leg.cost_usd_t / r["cost"]
                + w["risk"] * leg_risk(leg) / r["risk"])


def route_summary(legs):
    """Real-world totals for a route (independent of the objective used)."""
    total_time = sum(l.time_h for l in legs)
    return {
        "time_h": round(total_time, 1),
        "time_days": round(total_time / 24, 2),
        "time_p90_h": round(sum(l.time_p90_h for l in legs), 1),
        "cost_usd_t": round(sum(l.cost_usd_t for l in legs), 0),
        "success_prob": round(math.prod(l.reliability for l in legs), 3) if legs else 0.0,
        "max_risk": max((l.risk_score for l in legs), default=0),
        "bottleneck_t": min((l.capacity_t for l in legs), default=0),
        "hops": len(legs),
        "modes": " > ".join(l.mode for l in legs),
    }
