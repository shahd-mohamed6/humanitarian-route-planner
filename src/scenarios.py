"""Apply disruption scenarios (closures and delays) to a network.

Scenario actions from the dataset:
    CLOSE      remove the leg
    TIME_MULT  multiply transit times by `value`
    COST_MULT  multiply cost by `value`
    COMBINE    apply each listed scenario, e.g. "S3;S4"

We also support ad-hoc disruptions for the interactive demo:
closing whole nodes (e.g. an airport), corridors or transport modes.
"""


def scenario_list(scenarios_df):
    """[(scenario_id, scenario_name), ...] in dataset order."""
    seen, out = set(), []
    for r in scenarios_df.itertuples(index=False):
        if r.scenario_id not in seen:
            seen.add(r.scenario_id)
            out.append((r.scenario_id, r.scenario_name))
    return out


def apply_scenario(net, scenarios_df, scenario_id):
    """Return (new_network, list_of_change_descriptions)."""
    rows = scenarios_df[scenarios_df["scenario_id"] == scenario_id]
    if rows.empty:
        raise ValueError(f"Unknown scenario {scenario_id}")
    changes = []
    for r in rows.itertuples(index=False):
        action = str(r.action) if r.action == r.action else ""   # NaN check
        if action == "CLOSE" and r.leg_id in net.legs:
            net = net.without_legs([r.leg_id])
            changes.append(f"closed {r.leg_id}")
        elif action == "TIME_MULT" and r.leg_id in net.legs:
            net = net.with_multiplier([r.leg_id], time_mult=float(r.value))
            changes.append(f"{r.leg_id} time x{float(r.value)}")
        elif action == "COST_MULT" and r.leg_id in net.legs:
            net = net.with_multiplier([r.leg_id], cost_mult=float(r.value))
            changes.append(f"{r.leg_id} cost x{float(r.value)}")
        elif action == "COMBINE":
            for sub in str(r.value).split(";"):
                net, sub_changes = apply_scenario(net, scenarios_df, sub.strip())
                changes += sub_changes
    return net, changes


def close_nodes(net, node_ids):
    """Close every leg entering or leaving the given nodes (e.g. airport shut)."""
    node_ids = set(node_ids)
    ids = [l.id for l in net.legs.values() if l.u in node_ids or l.v in node_ids]
    return net.without_legs(ids)


def close_corridors(net, corridors):
    corridors = set(corridors)
    return net.without_legs([l.id for l in net.legs.values() if l.corridor in corridors])


def close_modes(net, modes):
    modes = set(modes)
    return net.without_legs([l.id for l in net.legs.values() if l.mode in modes])
