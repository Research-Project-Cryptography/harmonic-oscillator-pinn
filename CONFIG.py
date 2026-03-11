"""
Physical parameter sets for the underdamped harmonic oscillator datasets.

Each entry defines a unique physical regime.  Computed fields:
    mu_true = 2 * d
    k       = w0 ** 2
    m       = 1  (fixed throughout the project)

The first two (D1, D2) are the original paper datasets.
D3 and D4 extend coverage to higher damping / frequency regimes.
"""

OSCILLATION_PARAMS = [
    {
        "name": "D1_d2_w20",
        "d": 2,
        "w0": 20,
        "mu_true": 4,
        "k": 400,
        "m": 1,
    },
    {
        "name": "D2_d1.5_w30",
        "d": 1.5,
        "w0": 30,
        "mu_true": 3,
        "k": 900,
        "m": 1,
    },
    {
        "name": "D3_d3_w30",
        "d": 3,
        "w0": 30,
        "mu_true": 6,
        "k": 900,
        "m": 1,
    },
    {
        "name": "D4_d4_w40",
        "d": 4,
        "w0": 40,
        "mu_true": 8,
        "k": 1600,
        "m": 1,
    },
]
