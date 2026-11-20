# Team Guide: Humanitarian Route Planner

Everyone in the group must be able to explain the WHOLE system in the viva.
Read Part 1 together, then each person does their own task in Part 2.

---------------------------------------------------------------------------

## Part 1: How the system works (plain English)

### 1. The problem
Get aid from the Dubai humanitarian hub (`ORG`) to Gaza City (`GZC`). There are many
possible chains of flights, ships, trucks and airdrops. Each hop has a time, cost,
reliability and capacity. Crossings and airports close often. The planner finds
the best route for what is open **right now**, and re-plans when something closes.

### 2. The data (`data/gaza_air_bridge_dataset.xlsx`)
- **locations** (28 rows) = nodes: airports, ports, crossings, hubs. Each has lat/lon,
  a `tier` (1 = international, 2 = crossings/last mile) and `is_entry_gateway`
  (Rafah, Kerem Shalom, Zikim, airdrop zone, pier).
- **network_legs** (45 rows) = directed edges. A *leg* is one hop by one mode,
  e.g. "fly DWC→AAC". Columns: hours (typical and P90), USD/tonne, capacity,
  risk 1–5, reliability %.
- **disruption_scenarios** (S0–S10): CLOSE a leg, multiply time (TIME_MULT), or
  COMBINE scenarios.

### 3. The code, file by file
| File | What it does | Key idea to explain |
|---|---|---|
| `data_loader.py` | Reads Excel, checks sheets/columns, checks every leg points to real nodes | Validation stops silent errors |
| `network.py` | `Node`, `Leg`, `Network` (adjacency list: node → list of outgoing legs) | Graph = state space. State = current location |
| `costs.py` | Turns a leg into one number to minimise (time / robust / cost / risk / balanced) | Risk = −ln(reliability): adding these = multiplying probabilities |
| `heuristics.py` | `h(n) = k × straight-line km to Gaza` | Consistent, so A* stays optimal (proof below) |
| `search.py` | Dijkstra, A*, weighted A*, greedy, two-tier A*, all from scratch with a priority queue | Same loop; only f(n) changes |
| `scenarios.py` | Applies S0–S10 and ad-hoc closures (node, corridor, mode) | Returns a NEW network; the original is never changed |
| `synthetic.py` | Makes bigger fake networks (50–3000 nodes) with a seed | Real data is too small to show scaling |
| `planner.py` | One function `plan()` that runs the chosen algorithms | Used by app, demo, experiments |
| `app.py` | Streamlit map: pick objective, scenario, closures; compare algorithms; re-plan en route | The live demo |

### 4. Formal problem (AIMA style)
- **States:** the node where the cargo currently is.
- **Initial state:** `ORG`. **Goal test:** node == `GZC`.
- **Actions:** take any open outgoing leg whose capacity ≥ shipment tonnes.
- **Transition model:** taking leg u→v moves the cargo to v.
- **Path cost:** sum of leg weights for the chosen objective.

### 5. The algorithms
All use a priority queue (min-heap) ordered by f(n):
- **Dijkstra:** f = g (cost so far). Explores outward in all directions. Always optimal.
- **A\*:** f = g + h. h "pulls" the search toward Gaza, so it expands fewer nodes.
  Optimal if h is consistent.
- **Weighted A\*:** f = g + 1.5h. Trusts the heuristic more: faster, cost ≤ 1.5 × optimal.
- **Greedy:** f = h only. Very fast, but can return bad routes (it ignores cost so far).
- **Two-tier A\*:** first compute the exact cost from every Gaza gateway to Gaza City
  (backward Dijkstra on the small last-mile network). Then A* over the international
  network, which stops at gateways and adds the last-mile cost. Optimal, and the
  last-mile table can be reused when only international legs change.

### 6. Why the heuristic is consistent (likely viva question)
k = smallest (leg weight ÷ straight-line length) in the network.
For any leg u→v: weight ≥ k·d(u,v). Triangle inequality: d(u,G) ≤ d(u,v) + d(v,G).
So h(u) = k·d(u,G) ≤ weight(u,v) + h(v). That is consistency, and with h(G) = 0
it means h never overestimates (admissible).
For time, 1/k = the fastest effective speed seen anywhere (~211 km/h, an air leg
including handling), so h = "fly straight to Gaza at the best speed ever seen".

### 7. Key results so far (`python run_demo.py`)
- Baseline fastest route: ORG → DWC → El Arish airport → El Arish hub → Rafah → Gaza
  (about 2.4 days).
- Rafah closed (S1) or El Arish closed (S3): re-routes via Amman → Allenby → Kerem
  Shalom (about 2.8 days).
- Cheapest baseline route: by sea via Jebel Ali → Ashdod → Zikim. If the Red Sea is
  closed (S4), it switches to the Saudi land bridge.
- Dijkstra, A* and two-tier A* always return the same optimal cost; greedy is faster
  but its routes can cost up to about 3× more (e.g. $2042/t vs $598/t for the cheapest route).
- On the 28-node real network A* saves only a few expansions (the heuristic is weak
  because one air leg is very fast). On synthetic networks of 1000+ nodes A* expands
  about 15–20% fewer nodes, and weighted A* about 50% fewer.

### 8. Practice viva questions
1. Why A* over Dijkstra? Why not greedy?
2. Prove your heuristic is admissible. What happens if it overestimates?
3. Why −ln(reliability) for the risk objective?
4. What happens if Rafah closes mid-journey? (Show the Re-plan tab.)
5. What is the complexity? (O((V+E) log V) with a binary heap.) What if the network doubles?
6. Why does two-tier A* expand MORE nodes on the small real network? (It builds the
   whole last-mile table; that pays off when the table is reused or the network is large.)
7. How is the capacity constraint enforced? (Legs with capacity < tonnes are skipped
   in `neighbours()`.)

---------------------------------------------------------------------------

## Part 2: Individual tasks

Workflow for both teammates:
1. Download the repo from GitHub (green **Code** button → **Download ZIP**) and set up
   VS Code: `python -m venv venv`, activate, `pip install -r requirements.txt`.
2. Open Claude, start a new chat, attach the files listed in your prompt, and paste
   your prompt.
3. Write the code with Claude's help, step by step, until you understand every line.
4. Upload your file(s) to the GitHub repo yourself (**Add file → Upload files**), with a
   clear commit message.

### Teammate 2: Test suite (`tests/test_search.py`)

**Files to attach:** `src/network.py`, `src/costs.py`, `src/heuristics.py`,
`src/search.py`, `src/scenarios.py`, `src/planner.py`, `src/data_loader.py`,
`src/synthetic.py`, `config.json`, and this guide.

**Prompt to paste:**

```
You are my tutor for the CSCI218 (Foundations of AI, UOWD) group project. Our project
is a "Gaza Humanitarian Air Bridge Route Planner": a multimodal graph (28 locations,
45 transport legs) where we find the best route from Dubai (node ORG) to Gaza City
(node GZC) using Dijkstra, A*, two-tier A*, weighted A* and greedy best-first search,
under disruption scenarios S0-S10. I have attached our source code and team guide.

MY TASK: write the test suite tests/test_search.py using pytest. I must write it
myself with your guidance, because I will explain it in an individual viva and in my
reflection statement. So:
- Work STEP BY STEP. Give me one small step, explain it in simple terms, wait for me
  to say "done / next" before the next step. Ask me questions to check I understand.
- Explain the AI concept behind each test (optimality, admissibility, consistency,
  constraints) in simple words.
- Read the attached code carefully and use the real function names and signatures.
  Do not invent functions.

The tests should cover at least:
1. Data loading: 28 nodes, 45 legs, every leg's endpoints exist.
2. Optimality: Dijkstra, A* and two-tier A* return the SAME cost on the real network for
   every objective (time, robust, cost, risk, balanced) and every scenario S0-S10.
3. Cross-check: our Dijkstra matches networkx.dijkstra_path_length on the same weights.
4. Admissibility: for every node, h(n) <= true optimal cost from n to GZC.
5. Consistency: for every leg u->v, h(u) <= weight(u,v) + h(v).
6. Closures: a closed leg (e.g. S1 closes E32) never appears in the returned route.
7. Capacity constraint: with a 100-tonne shipment, no leg in the route has capacity < 100.
8. Edge cases: start == goal; goal unreachable (close everything into Gaza) returns
   found=False; weighted A* cost <= 1.5 x optimal; greedy is never better than optimal.
9. Synthetic networks: for seeds 1-10 and sizes 50 and 200, A* cost == Dijkstra cost.
10. Reproducibility: generate_network with the same seed gives identical networks.

Use pytest fixtures for loading the data once, and pytest.mark.parametrize for
scenarios and objectives. Use clear docstrings. At the end, help me run `pytest -v`,
fix any failures, and write a short paragraph for the report section
"Testing and validation" plus 3 bullet points I can use in my reflection statement.
Start with step 1.
```

### Teammate 3: Experiments and analysis (`experiments.py` + results)

**Files to attach:** all files in `src/`, `config.json`, `run_demo.py`, this guide.

**Prompt to paste:**

```
You are my tutor for the CSCI218 (Foundations of AI, UOWD) group project. Our project
is a "Gaza Humanitarian Air Bridge Route Planner": a multimodal graph where we find the
best route from Dubai to Gaza City with Dijkstra (baseline), A* (main method),
two-tier A*, weighted A* (w=1.5) and greedy best-first search. I have attached our
source code and team guide. src/synthetic.py generates larger seeded networks
(generate_network(n_outer, n_inner, n_gateways, k, seed) -> (net, start, goal)) and
src/planner.py has plan() and result_row().

MY TASK: write experiments.py, run it, and produce the results and analysis for the
report. The marking rubric asks for "meaningful metrics, multiple runs, mean and
standard deviation, comparison with a baseline, and threats to validity". I must write
this myself with your guidance because I will explain it in an individual viva. So:
- Work STEP BY STEP: one small step, explain it simply, wait for me to say "next".
  Check my understanding with questions.
- Read the attached code and use the real function names. Do not invent functions.

The script should run:
Experiment 1 (scaling): network sizes n_outer = 50, 100, 200, 500, 1000, 2000
  (n_inner=15, n_gateways=3), 30 random seeds per size (seeds from config.json
  random_seed + i), objective "time". Record for every algorithm: nodes_expanded,
  nodes_generated, runtime_ms, path cost and the optimality gap vs Dijkstra (%).
Experiment 2 (real network, all scenarios): every objective x scenario S0-S10 on the
  real dataset, each run repeated 30 times to get mean +- std runtime; also record
  route, time, cost, success probability, nodes expanded.
Experiment 3 (re-planning): for each disruption scenario S1-S10 on the real network,
  compare full re-plan with A* vs two-tier A* reusing a cached last-mile table
  (src.search.lastmile_table, pass table=...), counting nodes expanded.

Outputs (in results/): raw CSVs, a summary CSV with mean and std per algorithm and size,
and matplotlib charts saved as PNG with titles, axis labels and legends:
  (a) nodes expanded vs network size (error bars = std), one line per algorithm
  (b) runtime vs network size
  (c) optimality gap bar chart (weighted A*, greedy)
  (d) per-scenario delivery time for the time objective.
Use fixed seeds so results are reproducible, and a __main__ block so
`python experiments.py` runs everything.

At the end, help me write the report section "Experiments and results":
experimental setup, a results table (mean +- std), interpretation of each chart
(WHY A* expands fewer nodes, why weighted/greedy trade optimality for speed, why
two-tier helps re-planning), comparison against the Dijkstra baseline, and threats to
validity (synthetic data, small real network, timing noise). Then give me 3 bullet
points for my reflection statement. Start with step 1.
```

### You (lead developer)
Core code (`src/`, `app.py`, `run_demo.py`), integration, README, report assembly,
demo. Upload your files first so teammates can download them.
