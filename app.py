"""Interactive Humanitarian Route Planner (Streamlit + Folium).

Run:  streamlit run app.py
"""
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.costs import CostModel, OBJECTIVES
from src.data_loader import load_config, load_dataset
from src.network import Network
from src.planner import plan, result_row, ALGORITHMS
from src.scenarios import (apply_scenario, scenario_list, close_nodes,
                           close_corridors, close_modes)
from src.search import astar, dijkstra
from src.heuristics import make_heuristic

st.set_page_config(page_title="Gaza Humanitarian Route Planner", layout="wide")

ALGO_COLOURS = {"Dijkstra": "#1f77b4", "A*": "#2ca02c", "Two-tier A*": "#9467bd",
                "Weighted A*": "#ff7f0e", "Greedy best-first": "#d62728"}
MODE_DASH = {"Air": "8,6", "Airdrop": "2,6", "Sea": "14,6"}
OBJ_LABELS = {"time": "Fastest (typical hours)", "robust": "Fastest on a bad day (P90 hours)",
              "cost": "Cheapest (USD/tonne)", "risk": "Safest (max on-time probability)",
              "balanced": "Balanced (time + cost + risk)"}


@st.cache_data
def get_data():
    return load_dataset(), load_config()


data, cfg = get_data()
base = Network.from_frames(data["locations"], data["legs"])
scen = scenario_list(data["scenarios"])

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Planner settings")
objective = st.sidebar.selectbox("Optimise for", OBJECTIVES,
                                 format_func=lambda o: OBJ_LABELS[o])
weights = cfg["balanced_weights"]
if objective == "balanced":
    wt = st.sidebar.slider("Weight: time", 0.0, 1.0, weights["time"], 0.05)
    wc = st.sidebar.slider("Weight: cost", 0.0, 1.0, weights["cost"], 0.05)
    wr = st.sidebar.slider("Weight: risk", 0.0, 1.0, weights["risk"], 0.05)
    total = (wt + wc + wr) or 1.0
    weights = {"time": wt / total, "cost": wc / total, "risk": wr / total}
tonnes = st.sidebar.number_input("Shipment size (tonnes)", 1, 3000,
                                 int(cfg["shipment_tonnes"]), 5)
algos = st.sidebar.multiselect("Algorithms", ALGORITHMS,
                               default=["Dijkstra", "A*", "Two-tier A*"])

st.sidebar.markdown("---")
st.sidebar.subheader("Disruptions")
sid = st.sidebar.selectbox("Scenario", [s for s, _ in scen],
                           format_func=lambda s: f"{s}: {dict(scen)[s]}")
node_names = {n.id: f"{n.id} - {n.name}" for n in base.nodes.values()}
extra_nodes = st.sidebar.multiselect("Also close nodes (airports, ports, crossings)",
                                     [n for n in node_names if n not in ("ORG", "GZC")],
                                     format_func=node_names.get)
extra_corr = st.sidebar.multiselect("Also close corridors",
                                    sorted({l.corridor for l in base.legs.values()}))
extra_modes = st.sidebar.multiselect("Ban transport modes",
                                     sorted({l.mode for l in base.legs.values()}))
show_explored = st.sidebar.checkbox("Show nodes explored by Dijkstra vs A*", True)

# ------------------------------------------------------------- build network
net, changes = apply_scenario(base, data["scenarios"], sid)
if extra_nodes:
    net = close_nodes(net, extra_nodes)
if extra_corr:
    net = close_corridors(net, extra_corr)
if extra_modes:
    net = close_modes(net, extra_modes)
closed_ids = set(base.legs) - set(net.legs)

cm = CostModel.for_network(base, objective, weights)
origin, dest = cfg["origin"], cfg["destination"]
results = plan(net, origin, dest, cm, algos, min_capacity=tonnes,
               w=cfg["weighted_astar_w"]) if algos else {}

# ----------------------------------------------------------------- header
st.title("Gaza Humanitarian Air Bridge: Route Planner")
st.caption("Finds the best multimodal route from the Dubai humanitarian hub to "
           "Gaza City when airports, ports, crossings or roads are unavailable.")

tab_map, tab_scen, tab_replan = st.tabs(["Route map & comparison",
                                         "All scenarios", "Re-plan en route"])


def draw_map(net_view, results_view, closed, explored=None):
    m = folium.Map(location=[30.0, 42.0], zoom_start=5, tiles="OpenStreetMap")
    for leg in base.legs.values():                       # whole network, faint
        a, b = base.nodes[leg.u], base.nodes[leg.v]
        is_closed = leg.id in closed
        folium.PolyLine([(a.lat, a.lon), (b.lat, b.lon)],
                        color="red" if is_closed else "#bbbbbb",
                        weight=2 if is_closed else 1,
                        dash_array="4,4" if is_closed else MODE_DASH.get(leg.mode),
                        tooltip=f"{leg.id} {leg.u}->{leg.v} {leg.mode}"
                                + (" (CLOSED)" if is_closed else "")).add_to(m)
    offset = 0.0
    for name, res in results_view.items():               # chosen routes
        if not res.found:
            continue
        pts = [(base.nodes[n].lat + offset, base.nodes[n].lon + offset) for n in res.path]
        folium.PolyLine(pts, color=ALGO_COLOURS.get(name, "black"), weight=5,
                        opacity=0.8, tooltip=f"{name}: {' > '.join(res.path)}").add_to(m)
        offset += 0.04
    for n in base.nodes.values():
        colour = "black"
        if explored:
            in_d, in_a = n.id in explored[0], n.id in explored[1]
            colour = ("#2ca02c" if in_a and in_d else "#1f77b4" if in_d else
                      "#2ca02c" if in_a else "#999999")
        radius = 7 if n.id in (origin, dest) else 4
        folium.CircleMarker((n.lat, n.lon), radius=radius, color=colour, fill=True,
                            fill_opacity=0.9,
                            tooltip=f"{n.id}: {n.name} ({n.node_type})").add_to(m)
    return m


with tab_map:
    if changes or closed_ids:
        st.info(f"Disruption active: {len(closed_ids)} legs closed. "
                + ("; ".join(c for c in changes if 'x' in c) or ""))
    explored = None
    if show_explored:
        h, _ = make_heuristic(net, cm, dest)
        d_res = dijkstra(net, origin, dest, cm, tonnes)
        a_res = astar(net, origin, dest, cm, h, tonnes)
        explored = (set(d_res.expanded_order), set(a_res.expanded_order))
    col_map, col_tbl = st.columns([3, 2])
    with col_map:
        st_folium(draw_map(net, results, closed_ids, explored), height=560,
                  use_container_width=True, returned_objects=[])
        st.caption("Coloured lines = routes per algorithm. Red dashed = closed. "
                   + ("Node colour: green = expanded by A* (and Dijkstra), "
                      "blue = expanded only by Dijkstra, grey = never expanded."
                      if show_explored else ""))
    with col_tbl:
        if not results:
            st.warning("Select at least one algorithm.")
        else:
            rows = [result_row(r, net) for r in results.values()]
            df = pd.DataFrame(rows)
            if not df["found"].any():
                st.error("No feasible route under these disruptions and shipment size.")
            else:
                cols = ["algorithm", "objective_cost", "time_days", "cost_usd_t",
                        "success_prob", "nodes_expanded", "runtime_ms"]
                st.subheader("Algorithm comparison")
                st.dataframe(df[[c for c in cols if c in df]], hide_index=True)
                best = next(r for r in results.values() if r.found)
                st.subheader(f"Route ({best.algorithm})")
                st.dataframe(pd.DataFrame([{
                    "leg": l.id, "from": base.nodes[l.u].name, "to": base.nodes[l.v].name,
                    "mode": l.mode, "hours": l.time_h, "USD/t": l.cost_usd_t,
                    "reliability": l.reliability, "capacity t/wk": l.capacity_t}
                    for l in best.legs]), hide_index=True)

with tab_scen:
    st.subheader(f"Best route under every scenario ({OBJ_LABELS[objective]}, {tonnes} t)")
    rows = []
    for s, name in scen:
        n2, _ = apply_scenario(base, data["scenarios"], s)
        res = plan(n2, origin, dest, cm, ["Dijkstra", "A*"], min_capacity=tonnes)
        d, a = res["Dijkstra"], res["A*"]
        row = {"scenario": f"{s}: {name}"}
        if d.found:
            row.update(result_row(d, n2))
            row["A* expanded"] = a.nodes_expanded
            row["Dijkstra expanded"] = d.nodes_expanded
            row["same cost"] = abs(a.cost - d.cost) < 1e-9
        else:
            row["route"] = "NO FEASIBLE ROUTE"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows)[["scenario", "route", "time_days", "cost_usd_t",
                                     "success_prob", "Dijkstra expanded",
                                     "A* expanded", "same cost"]], hide_index=True)

with tab_replan:
    st.subheader("Dynamic re-planning: disruption happens while the cargo is moving")
    plan0 = plan(base, origin, dest, cm, ["A*"], min_capacity=tonnes)["A*"]
    if not plan0.found:
        st.error("No baseline route.")
    else:
        st.write("**Original plan (no disruption):** " + " > ".join(plan0.path))
        k = st.slider("Cargo has completed this many legs when the disruption hits",
                      0, len(plan0.legs) - 1, 1)
        current = plan0.path[k]
        st.write(f"Cargo is now at **{base.nodes[current].name} ({current})**. "
                 f"Disruption applied: **{sid}: {dict(scen)[sid]}** plus any extra closures.")
        new = plan(net, current, dest, cm, ["A*"], min_capacity=tonnes)["A*"]
        done = plan0.legs[:k]
        if new.found:
            full = done + new.legs
            changed = [l.id for l in full] != [l.id for l in plan0.legs]
            st.success(("Route CHANGED. " if changed else "Original route still valid. ")
                       + "New full route: " + " > ".join(plan0.path[:k] + new.path))
            c1, c2 = st.columns(2)
            c1.metric("Original total time (h)", round(sum(l.time_h for l in plan0.legs), 1))
            c2.metric("Re-planned total time (h)", round(sum(l.time_h for l in full), 1),
                      delta=round(sum(l.time_h for l in full)
                                  - sum(l.time_h for l in plan0.legs), 1),
                      delta_color="inverse")
            st.caption(f"Re-planning took {new.runtime_ms:.2f} ms and expanded "
                       f"{new.nodes_expanded} nodes.")
            st_folium(draw_map(net, {"Original": plan0, "A*": new}, closed_ids),
                      height=450, use_container_width=True, returned_objects=[],
                      key="replan_map")
        else:
            st.error(f"No feasible route from {current} under this disruption: the "
                     "cargo must wait or return to a hub.")
