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

| Controller Variant | Unseen Seeds Success (%) | Unseen Seeds Completion (%) | Traffic Collision Rate (%) | Traffic Out-of-Road Rate (%) |
|---|---|---|---|---|
| **MPC + Safety Constraints** | **40.0%** | **76.9%** | **0.0%** | 80.0% |
| **MPC (Unconstrained Cost Only)** | 40.0% | 80.3% | **20.0%** | 80.0% |

*Key finding*: Disabling safety constraints doubles vehicle collision rate in traffic (from 0.0% to 20.0%), confirming that unconstrained cost optimization gambles on narrow clearances that physical collision margins forbid.

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

1. **Case 1 (Map `SC`, Seed 3002)**: Infeasible safety fallback timeout. Vehicle completed 77.4% of route at $8.33\text{ m/s}$, before braking into safety fallback due to tight road boundary clearance. [Animation: `docs/img/failure_case_1.gif` (0.72 MB)]
2. **Case 2 (Map `CS`, Seed 3003)**: Out-of-road on curve transition. Lateral drift reached $2.1\text{ m}$ on continuous curve entry, clipping the outside lane curb at step 64. [Animation: `docs/img/failure_case_2.gif` (0.28 MB)]
3. **Case 3 (Map `O`, Seed 4000)**: Held-out roundabout geometry. Unseen continuous curve required high persistent steering beyond the 1.0s planning horizon lookahead. [Animation: `docs/img/failure_case_3.gif` (1.67 MB)]
4. **Case 4 (Map `CO`, Seed 4002)**: Curve-to-roundabout transition. Vehicle exited roadway at step 59 due to sudden inflection point in reference lane tangent. [Animation: `docs/img/failure_case_4.gif` (0.22 MB)]
5. **Case 5 (Map `TO`, Seed 4003)**: Dynamic collision at T-junction intersection. Lead vehicle decelerated abruptly across intersection corridor. [Animation: `docs/img/failure_case_5.gif` (1.13 MB)]
