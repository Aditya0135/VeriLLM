"""
Reads config/config.yaml plus params.yaml and hands out one frozen config per
stage. The only place in the package that touches those two files.
"""

from pathlib import Path

from VeriLLM.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from VeriLLM.entity.config_entity import (
    AnnotationConfig,
    DataIngestionConfig,
    LabelAggregationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    ReferenceRetrievalConfig,
    TokenizationConfig,
)
from VeriLLM.utils.common import create_directories, read_yaml


class ConfigurationManager:
    """
    Build stage configs from the two YAML files.

    Args:
        config_filepath (Path): paths file.
        params_filepath (Path): hyperparameters file.

    Example:
        >>> config = ConfigurationManager()
        >>> config.get_annotation_config().max_retries
        5
    """

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]
        params = self.params["data"]

        create_directories([config["root_dir"]])

        return DataIngestionConfig(
            root_dir=Path(config["root_dir"]),
            dataset_name=params["dataset_name"],
            config_name=params["config_name"],
        )

    def get_reference_retrieval_config(self) -> ReferenceRetrievalConfig:
        config = self.config["reference_retrieval"]
        params = self.params["reference_retrieval"]

        create_directories([config["root_dir"]])

        return ReferenceRetrievalConfig(
            root_dir=Path(config["root_dir"]),
            reference_map_path=Path(config["reference_map_path"]),
            search_depth=params["search_depth"],
            include_answer=params["include_answer"],
            translator_model=params["translator_model"],
        )

    def get_annotation_config(self) -> AnnotationConfig:
        config = self.config["annotation"]
        params = self.params["annotation"]

        create_directories([config["root_dir"], Path(config["log_path"]).parent])

        return AnnotationConfig(
            root_dir=Path(config["root_dir"]),
            annotations_path=Path(config["annotations_path"]),
            log_path=Path(config["log_path"]),
            models=params["models"],
            temperature=params["temperature"],
            max_retries=params["max_retries"],
            retry_wait_seconds=params["retry_wait_seconds"],
            rpm_sleep_seconds=params["rpm_sleep_seconds"],
        )

    def get_label_aggregation_config(self) -> LabelAggregationConfig:
        config = self.config["label_aggregation"]
        params = self.params["label_aggregation"]

        create_directories([config["root_dir"]])

        return LabelAggregationConfig(
            root_dir=Path(config["root_dir"]),
            dataset_path=Path(config["dataset_path"]),
            annotations_path=Path(self.config["annotation"]["annotations_path"]),
            min_valid_votes=params["min_valid_votes"],
            hard_label_threshold=params["hard_label_threshold"],
        )

    def get_tokenization_config(self) -> TokenizationConfig:
        params = self.params["tokenizer"]

        return TokenizationConfig(
            model_name=params["model_name"],
            max_length=params["max_length"],
            use_reference=params["use_reference"],
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config["model_trainer"]
        params = self.params["training"]

        create_directories([config["root_dir"]])

        return ModelTrainerConfig(
            root_dir=Path(config["root_dir"]),
            checkpoint_dir=Path(config["checkpoint_dir"]),
            final_model_dir=Path(config["final_model_dir"]),
            dataset_path=Path(self.config["label_aggregation"]["dataset_path"]),
            model_name=self.params["tokenizer"]["model_name"],
            learning_rate=float(params["learning_rate"]),
            train_batch_size=params["train_batch_size"],
            eval_batch_size=params["eval_batch_size"],
            num_train_epochs=params["num_train_epochs"],
            weight_decay=params["weight_decay"],
            save_total_limit=params["save_total_limit"],
            seed=params["seed"],
            fp16=params["fp16"],
            use_soft_labels=params["use_soft_labels"],
            metric_for_best_model=params["metric_for_best_model"],
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config["model_evaluation"]
        params = self.params["evaluation"]

        create_directories([config["root_dir"], config["plots_dir"]])

        return ModelEvaluationConfig(
            root_dir=Path(config["root_dir"]),
            metrics_path=Path(config["metrics_path"]),
            plots_dir=Path(config["plots_dir"]),
            final_model_dir=Path(self.config["model_trainer"]["final_model_dir"]),
            predict_batch_size=params["predict_batch_size"],
            betas=tuple(params["betas"]),
            cutoff_sweep=tuple(params["cutoff_sweep"]),
            dataset_name=self.params["data"]["dataset_name"],
            config_name=self.params["data"]["config_name"],
        )
