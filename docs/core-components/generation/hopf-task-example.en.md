# Task Example: SINDy System Identification

This example describes the inputs, program interface and outputs of a Harbor scientific-computing task. It uses synthetic observations to assess recovery of dynamical equations from data.

## Objective

Recover the coefficient matrix of a parameterized Hopf system from noisy state and derivative observations, then predict growth rates, stable amplitudes and a bifurcation location:

```text
xdot = (b0 + b1*mu)*x - omega*y - a*(x² + y²)*x
ydot = omega*x + (b0 + b1*mu)*y - a*(x² + y²)*y
```

The constants are unknown, with `a > 0`, `b1 > 0` and `omega > 0`. The library contains all 20 monomials in `x`, `y`, `mu` of total degree at most three.

## Inputs

| File | Content |
| --- | --- |
| `samples.csv` | 900 observations with columns `x,y,mu,xdot,ydot`, including small Gaussian derivative noise |
| `library.json` | Exponents and column order for the 20 monomials |
| `queries.json` | Four query `mu` values |
| `README.md` | Sign conventions, formats and output requirements |

Example observations:

```csv
x,y,mu,xdot,ydot
0.25155190122069637,1.2631956772361304,-0.80225488734348971,-2.7852517274645363,-2.3684885798716087
-0.049753970454211593,0.037702180145090303,-0.80225488734348971,-0.023963676199332287,-0.11981683825047357
```

Query structure:

```json
{"query_mu": [-0.5496858751490484, -0.18767029100368254, 0.2080211614342755, 0.5447798443601972]}
```

## Program Interface

The agent implements `/app/solver.py`:

```bash
python /app/solver.py --case-dir /app/public_case --output /app/output
```

It writes `/app/output/solution.json`:

| Field | Requirement |
| --- | --- |
| `coefficients` | 2×20 numeric matrix, rows for `xdot` and `ydot`, columns ordered by `library.json` |
| `query_mu` | Input query parameters in their original order |
| `sigma` | Growth rates, `b0 + b1*mu` |
| `amplitude` | Stable amplitudes, `sqrt(sigma/a)` when positive, otherwise zero |
| `mu_c` | Bifurcation location, `-b0/b1` |

Derive predictions from the recovered coefficients. Correct least-squares or thresholded-regression approaches are accepted; grading does not prescribe an algorithm.

## Package Layout

```text
hopf-sindy-identification/
├── instruction.md
├── task.toml
├── environment/
│   ├── Dockerfile
│   ├── starter/
│   └── data/
├── generation_data.py
├── generation/Dockerfile
├── solution/
└── tests/
    ├── Dockerfile
    ├── test_outputs.py
    ├── support/
    └── data/
```

Public inputs live under `environment/data/`. Reference solutions and hidden truth remain under `solution/` and `tests/data/`, outside the contestant image. The starter provides an unimplemented CLI scaffold.

The runtime uses Python 3.12, NumPy 1.26.4, two CPUs and 2048 MB RAM. Data are generated locally without external dataset downloads.

## Grading

Three hidden cases use different coefficients and random seeds. All coefficients, growth rates, amplitudes and bifurcation locations must pass:

- Coefficient tolerance: `0.01 + 0.02 × |truth|`.
- Derived-value tolerance: `0.01 + 0.03 × |truth|`.
- Every field in every case must pass for reward 1; otherwise reward 0.

Production checks cover task format, deterministic data generation, reference-solution acceptance and no-op rejection. These checks do not calibrate task difficulty.

See [Graph usage](graph-task-producer.md) for task production and [Task Production Protocols](task-protocols.md) for the shared interfaces.
