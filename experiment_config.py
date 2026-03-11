"""
Central experiment configuration for the PIQML harmonic oscillator project.

Every tuneable hyperparameter lives here so that experiments are fully
reproducible from a single config object.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DatasetConfig:
    """Physical parameters and data-generation settings for one dataset.

    Training data strategy (matching the original paper):
        1. Generate n_points over [t_min, t_max] with Gaussian noise.
        2. Keep only points where t <= t_min + train_fraction*(t_max-t_min).
           This creates a hard cutoff so the model must *extrapolate* the tail.
        3. Subsample every train_subsample-th point from that window.

    Example with defaults (300 pts, fraction=0.55, subsample=5):
        window  = t in [0.0, 0.55]  -> 165 points
        sampled = every 5th         ->  33 training points
        model must generalise t in (0.55, 1.0] without any training data
    """
    name: str
    d: float
    w0: float
    n_points: int = 300
    noise_std: float = 0.02
    train_fraction: float = 0.55   # fraction of time domain used for training
    train_subsample: int = 5       # keep every Nth point in the training window
    t_min: float = 0.0
    t_max: float = 1.0
    data_seed: int = 123

    @property
    def mu_true(self) -> float:
        return 2 * self.d

    @property
    def k(self) -> float:
        return self.w0 ** 2

    @property
    def m(self) -> float:
        return 1.0


@dataclass
class ModelConfig:
    """Architecture specification for one model."""
    name: str
    model_type: Literal["hybrid_qn", "fcn"]

    n_qubits: int = 6
    n_circuit_layers: int = 5
    rotation: Literal["Ry", "Rxyz"] = "Ry"

    n_hidden: int = 4
    n_mlp_layers: int = 6

    input_dim: int = 1
    output_dim: int = 1


@dataclass
class TrainingConfig:
    """Optimiser, loss weights, and iteration settings."""
    iterations: int = 50_000
    learning_rate: float = 1e-2
    optimizer: Literal["adam", "sgd", "lbfgs"] = "adam"

    lambda1: float = 1e5
    lambda2: float = 1e5
    lambda3: float = 1.0
    lambda4: float = 1e5

    log_interval: int = 1000


@dataclass
class ExperimentConfig:
    """Top-level config that bundles dataset, model, and training settings."""
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "results"


# ---------------------------------------------------------------------------
# Preset dataset configs
# ---------------------------------------------------------------------------

DATASET_D1 = DatasetConfig(name="D1_d2_w20", d=2.0, w0=20.0)
DATASET_D2 = DatasetConfig(name="D2_d1.5_w30", d=1.5, w0=30.0)
DATASET_D3 = DatasetConfig(name="D3_d3_w30", d=3.0, w0=30.0)
DATASET_D4 = DatasetConfig(name="D4_d4_w40", d=4.0, w0=40.0)

ALL_DATASETS = [DATASET_D1, DATASET_D2, DATASET_D3, DATASET_D4]

# ---------------------------------------------------------------------------
# Preset model configs
# ---------------------------------------------------------------------------

MODEL_PIQML_109 = ModelConfig(
    name="PIQML_109",
    model_type="hybrid_qn",
    n_qubits=6,
    n_circuit_layers=5,
    rotation="Rxyz",
    input_dim=1,
    output_dim=1,
)

MODEL_PIML_113 = ModelConfig(
    name="PIML_113",
    model_type="fcn",
    n_hidden=4,
    n_mlp_layers=6,
    input_dim=1,
    output_dim=1,
)

MODEL_PIML_2209 = ModelConfig(
    name="PIML_2209",
    model_type="fcn",
    n_hidden=32,
    n_mlp_layers=3,
    input_dim=1,
    output_dim=1,
)

ALL_MODELS = [MODEL_PIQML_109, MODEL_PIML_113, MODEL_PIML_2209]

# Sweep order: run classical first, quantum last (quantum is slowest).
MODEL_PRIORITY = [MODEL_PIML_2209, MODEL_PIML_113, MODEL_PIQML_109]

# ---------------------------------------------------------------------------
# Default training config
# ---------------------------------------------------------------------------

DEFAULT_TRAINING = TrainingConfig()

# ---------------------------------------------------------------------------
# Convenience: build a full ExperimentConfig from presets
# ---------------------------------------------------------------------------

def make_experiment(
    model: ModelConfig = MODEL_PIQML_109,
    dataset: DatasetConfig = DATASET_D1,
    training: TrainingConfig = DEFAULT_TRAINING,
    seed: int = 42,
    device: str = "cpu",
    output_dir: str = "results",
) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset,
        model=model,
        training=training,
        seed=seed,
        device=device,
        output_dir=output_dir,
    )
