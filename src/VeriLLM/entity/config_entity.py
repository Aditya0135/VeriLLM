"""
Frozen config objects, one per stage.

Every component takes exactly one of these and reads nothing else from disk.
That is what keeps a component testable: hand it a dataclass built in memory
and it runs without config/config.yaml existing at all.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    config_name: str


@dataclass(frozen=True)
class ReferenceRetrievalConfig:
    root_dir: Path
    reference_map_path: Path
    search_depth: str
    include_answer: str
    translator_model: str


@dataclass(frozen=True)
class AnnotationConfig:
    root_dir: Path
    annotations_path: Path
    log_path: Path
    models: dict
    temperature: float
    max_retries: int
    retry_wait_seconds: int
    rpm_sleep_seconds: float


@dataclass(frozen=True)
class LabelAggregationConfig:
    root_dir: Path
    dataset_path: Path
    annotations_path: Path
    min_valid_votes: int
    hard_label_threshold: float


@dataclass(frozen=True)
class TokenizationConfig:
    model_name: str
    max_length: int
    use_reference: bool


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    checkpoint_dir: Path
    final_model_dir: Path
    dataset_path: Path
    model_name: str
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    num_train_epochs: int
    weight_decay: float
    save_total_limit: int
    seed: int
    fp16: bool
    use_soft_labels: bool
    metric_for_best_model: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    metrics_path: Path
    plots_dir: Path
    final_model_dir: Path
    predict_batch_size: int
    betas: tuple
    cutoff_sweep: tuple
    dataset_name: str
    config_name: str
