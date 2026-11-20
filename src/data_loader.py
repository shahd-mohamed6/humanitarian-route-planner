"""Load and validate the Gaza Air Bridge dataset (Excel workbook).

The workbook has three sheets used by the planner:
    locations             -> graph nodes
    network_legs          -> directed graph edges ("legs" = one hop by one mode)
    disruption_scenarios  -> closures / delays applied before searching
"""
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "locations": ["node_id", "name", "node_type", "country", "lat", "lon",
                  "tier", "role", "is_entry_gateway"],
    "network_legs": ["leg_id", "from_node", "to_node", "mode", "distance_km",
                     "transit_hours_typical", "transit_hours_p90",
                     "cost_usd_per_tonne", "capacity_tonnes_per_week",
                     "risk_score_1to5", "reliability_pct", "corridor_group"],
    "disruption_scenarios": ["scenario_id", "scenario_name", "leg_id",
                             "action", "value"],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    """Read config.json (project root by default)."""
    path = Path(path) if path else PROJECT_ROOT / "config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_dataset(path=None):
    """Return a dict of validated DataFrames: locations, legs, scenarios.

    Raises ValueError if a sheet or column is missing, or if a leg refers to
    a node that does not exist.
    """
    if path is None:
        path = PROJECT_ROOT / load_config()["dataset_path"]
    sheets = pd.read_excel(path, sheet_name=None)

    for sheet, cols in REQUIRED_COLUMNS.items():
        if sheet not in sheets:
            raise ValueError(f"Missing sheet: {sheet}")
        missing = [c for c in cols if c not in sheets[sheet].columns]
        if missing:
            raise ValueError(f"Sheet '{sheet}' is missing columns: {missing}")

    locations = sheets["locations"]
    legs = sheets["network_legs"]
    scenarios = sheets["disruption_scenarios"]

    node_ids = set(locations["node_id"])
    bad = legs[~legs["from_node"].isin(node_ids) | ~legs["to_node"].isin(node_ids)]
    if not bad.empty:
        raise ValueError(f"Legs reference unknown nodes: {list(bad['leg_id'])}")
    if locations["node_id"].duplicated().any():
        raise ValueError("Duplicate node_id values in locations")
    if legs["leg_id"].duplicated().any():
        raise ValueError("Duplicate leg_id values in network_legs")

    return {"locations": locations, "legs": legs, "scenarios": scenarios}
