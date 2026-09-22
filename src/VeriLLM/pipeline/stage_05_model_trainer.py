"""
Stage 05 - fine-tune the token classifier on the silver labels.

Free (no API). Train on silver `train_unlabeled`, select the checkpoint on
gold `validation`. The gold `test` split is not touched here.
"""

from datasets import load_from_disk
from transformers import AutoTokenizer

from VeriLLM import logger
from VeriLLM.components.model_trainer import ModelTrainer
from VeriLLM.components.plots import plot_loss_curves
from VeriLLM.components.tokenization import Tokenizer
from VeriLLM.config.configuration import ConfigurationManager

STAGE_NAME = "Model Trainer"


class ModelTrainerPipeline:
    """Tokenize, train and save the detector."""

    def main(self, val, silver=None):
        """
        Args:
            val (datasets.Dataset): the gold validation split, raw.
            silver (datasets.Dataset | None): silver rows; loaded from disk
                when omitted.

        Returns:
            transformers.Trainer: the fitted trainer.
        """
        manager = ConfigurationManager()

        trainer_config = manager.get_model_trainer_config()
        tokenization_config = manager.get_tokenization_config()
        evaluation_config = manager.get_model_evaluation_config()

        if silver is None:
            silver = load_from_disk(str(trainer_config.dataset_path))
            logger.info(f"loaded silver dataset: {len(silver)} rows")

        hf_tokenizer = AutoTokenizer.from_pretrained(tokenization_config.model_name)
        encoder = Tokenizer(tokenization_config, hf_tokenizer)

        train_dataset = encoder.tokenize_dataset(silver, "silver train")
        eval_dataset = encoder.tokenize_dataset(val, "gold validation")

        trainer = ModelTrainer(trainer_config, hf_tokenizer).train(
            train_dataset, eval_dataset
        )

        plot_loss_curves(
            trainer, evaluation_config.plots_dir / "loss_curves.png"
        )

        return trainer


if __name__ == "__main__":
    from VeriLLM.pipeline.stage_01_data_ingestion import DataIngestionPipeline

    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    _, val, _ = DataIngestionPipeline().main()

    ModelTrainerPipeline().main(val)

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
