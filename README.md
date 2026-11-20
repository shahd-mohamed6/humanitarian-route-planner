# Gaza Humanitarian Air Bridge: Humanitarian Route Planner

CSCI218 Foundations of AI, UOWD Autumn 2026, Group Project (Topic 11, Dubai Humanitarian use case B)

Finds the fastest, cheapest, safest or best-balanced multimodal route (air, sea, road,
airdrop) from the Dubai humanitarian logistics hub to Gaza City when airports, ports,
crossings or roads are unavailable, and compares search algorithms on route cost,
delivery time and nodes explored.

## Algorithms
| Algorithm | Priority f(n) | Optimal? | Role |
|---|---|---|---|
| Dijkstra (uniform-cost search) | g(n) | Yes | Baseline |
| A* | g(n) + h(n) | Yes (consistent h) | Main method |
| Two-tier A* | tier-1 A* to Gaza entry gateways + tier-2 last-mile table | Yes | Hierarchical variant (Topic 4) |
| Weighted A* | g(n) + w·h(n), w = 1.5 | Bounded (≤ w × optimal) | Speed/quality trade-off |
| Greedy best-first | h(n) | No | Lower-bound on effort |

Heuristic: `h(n) = k × great-circle distance(n, goal)`, where `k = min over legs of
weight / straight-line distance`. It is consistent (so admissible) for every objective;
see `src/heuristics.py` for the proof.

Objectives: `time`, `robust` (P90 time), `cost`, `risk` (−ln reliability, i.e. maximise
on-time probability), `balanced` (weighted, normalised mix). A shipment-size capacity
constraint removes legs that cannot carry the load.

## Install
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py                      # interactive map + comparison (demo)
python run_demo.py                        # all scenarios, time objective
python run_demo.py --objective cost       # other objective
python run_demo.py --all-objectives       # writes results/scenario_results.csv
python experiments.py                     # scaling experiments (results/ CSV + charts)
pytest -q                                 # tests
```

## Project layout
```
app.py               Streamlit interactive map and re-planning demo
run_demo.py          Command-line scenario runner
experiments.py       Repeated-run scaling experiments
config.json          Origin, destination, shipment size, weights, seed
data/                gaza_air_bridge_dataset.xlsx (instructor-provided course dataset)
src/data_loader.py   Load + validate the Excel workbook
src/network.py       Node / Leg / Network classes, great-circle distance
src/costs.py         Cost models (objectives) and route summaries
src/heuristics.py    Admissible, consistent heuristic
src/search.py        Dijkstra, A*, weighted A*, greedy, two-tier A* (from scratch)
src/scenarios.py     Disruption scenarios S0–S10 + ad-hoc closures
src/synthetic.py     Seeded synthetic networks for scaling tests
src/planner.py       One-call wrapper used by app, demo and experiments
tests/               pytest test suite
results/             CSV results and charts
```

## Reproducibility
All randomness (synthetic networks) uses fixed seeds from `config.json`.
Data source: CSCI218 course dataset (instructor-provided, anonymised).
