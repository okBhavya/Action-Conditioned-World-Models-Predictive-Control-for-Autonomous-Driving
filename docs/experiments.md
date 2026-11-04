# Experimental Protocol & Benchmark Configuration

## Map Suites & Data Splits

To ensure that the learned world models generalize across road topologies, scenarios are divided into strictly disjoint seed ranges and held-out map geometries as defined in `configs/maps.yaml`.

| Split / Suite Name | Road Layouts Included | Seed Range | Traffic Density | Purpose |
|---|---|---|---|---|
| **train** | `['S', 'C', 'SC', 'CS', 'X', 'r', 'R']` | 1000 – 1999 | 0.0 | Training data collection across straights, curves, intersections, and ramps |
| **val** | `['SC', 'CS', 'X']` | 2000 – 2099 | 0.0 | Validation split for early stopping and hyperparameter tuning |
| **test_unseen_seeds** | `['S', 'C', 'SC', 'CS', 'X']` | 3000 – 3049 | 0.0 | Test Suite 1: Evaluating generalization to unseen procedural scenarios on seen layouts |
| **test_unseen_layouts** | `['O', 'T', 'CO', 'TO']` | 4000 – 4049 | 0.0 | Test Suite 2: Generalization to completely held-out layouts (Roundabouts 'O', T-junctions 'T') |
| **test_traffic** | `['SC', 'CS', 'X']` | 5000 – 5049 | 0.08 | Test Suite 3: Closed-loop collision avoidance under dynamic vehicle traffic |

> **Note on Disjoint Splits**:
> - Roundabouts (`O`) and T-junctions (`T`) are strictly excluded from the training split.
> - Vehicle dynamics remain fixed across all splits (`random_agent_model: False`) so physical transferability is preserved while topological variation is rigorously evaluated.

---

## Behavior Policy Mixture

Pure random exploration in autonomous driving results in immediate curb collisions and fails to cover normal highway or curved driving regimes. To construct a representative dataset of both nominal and off-nominal states, three behavior policies were mixed:

1. **Noisy Baseline Controller (60%)**:
   The tuned Stanley PID controller combined with an Ornstein-Uhlenbeck (OU) random walk noise process ($\rho = 0.75, \sigma = [0.12, 0.15]$), collecting smooth path tracking transitions with gentle perturbations.

2. **Smoothed Random Walk (20%)**:
   A low-pass filtered random action policy initialized at diverse speeds ($v \in [3, 8]\text{ m/s}$) exploring varied acceleration and steering regimes.

3. **Disturbance and Recovery (20%)**:
   The baseline controller with intermittent short steering impulses (4–8 steps of $\pm 0.45$ steering bias) followed by active recovery back to the lane centerline. This guarantees coverage of off-center offsets ($e_{lat} \in [0.5, 2.5]\text{ m}$) and high heading errors ($e_\psi \in [0.2, 0.8]\text{ rad}$).

---

## Ablation Study Results

### Ablation A: Impact of Explicit Safety Constraints
Evaluated on `test_unseen_seeds` and `test_traffic` (`results/final/ablation_a/summary.json`):

| Controller Variant | Unseen Seeds Success (%) | Unseen Seeds Completion (%) | Traffic Collision Rate (%) | Traffic Out-of-Road Rate (%) | Traffic Route Completion (%) |
|---|---|---|---|---|---|
| **MPC + Safety Constraints** | **40.0%** | **76.9%** | **20.0%** | **20.0%** | **63.2%** |
| **MPC (Unconstrained Cost Only)** | 40.0% | 80.3% | 20.0% | 40.0% | 58.5% |

*Key finding*: Incorporating explicit road boundary and collision constraints halves the out-of-road rate under dynamic traffic (from 40.0% to 20.0%) and improves route completion from 58.5% to 63.2%, demonstrating that cost-only optimization gambles on narrow clearances that physical collision margins forbid.

### Ablation B: Horizon ($H$) vs. Candidate Count ($N$) vs. Planning Latency
Evaluated on procedural curve map `SC` (`results/final/ablation_b_grid.json`):

| Horizon ($H$) | Candidates ($N$) | Mean CPU Planning Latency ($ms$) | Route Completion (%) | Real-Time Feasibility ($\le 100\text{ ms}$) |
|---|---|---|---|---|
| $H=5$ | $N=128$ | **13.3 ms** | 46.0% | Yes (75 Hz) |
| $H=5$ | $N=256$ | **25.7 ms** | 46.1% | Yes (39 Hz) |
| $H=10$ | $N=128$ | **32.5 ms** | 45.4% | Yes (31 Hz) |
| $H=10$ | $N=256$ | **53.0 ms** | 45.0% | Yes (19 Hz) |

*Key finding*: PyTorch batch rollouts scale linearly with $H \times N$. At $H=10, N=256$, mean latency is $53.0\text{ ms}$, comfortably within the simulator's $100\text{ ms}$ ($10\text{ Hz}$) control interval.

---

## Failure Analysis of MPC Episodes

Detailed in `results/final/failure_analysis.json`:

1. **Case 1 (Map `SC`, Seed 3002)**: Infeasible safety fallback timeout. Vehicle completed 55.5% of route at $8.61\text{ m/s}$ before braking into safety fallback due to tight road boundary clearance. [Animation: `docs/img/failure_case_1.gif` (0.72 MB)]
2. **Case 2 (Map `O`, Seed 4000)**: Held-out roundabout geometry collision. Unseen continuous curve required persistent steering beyond the 1.0s lookahead, resulting in collision at 30.7% completion. [Animation: `docs/img/failure_case_2.gif` (0.63 MB)]
3. **Case 3 (Map `CO`, Seed 4002)**: Curve-to-roundabout transition exit. Sudden inflection point in reference lane tangent caused boundary exit at 18.1% completion. [Animation: `docs/img/failure_case_3.gif` (0.61 MB)]
4. **Case 4 (Map `TO`, Seed 4003)**: Held-out T-junction collision. Obstacle interaction across intersection led to collision at 47.2% completion. [Animation: `docs/img/failure_case_4.gif` (1.15 MB)]
5. **Case 5 (Map `SC`, Seed 5003)**: Traffic clearance horizon timeout. Vehicle safely navigated dense traffic reaching 93.7% completion before the maximum step limit elapsed under conservative safety spacing. [Animation: `docs/img/failure_case_5.gif` (0.83 MB)]
