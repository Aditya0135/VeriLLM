"""
Stage 06 - official IoU / rho evaluation against the gold splits.

Free (no API). Loads the saved checkpoint from disk, so it is self-contained:
it does not need stage 05 to have run in the same process.
"""

from transformers import AutoModelForTokenClassification, AutoTokenizer

from VeriLLM import logger
from VeriLLM.components.model_evaluation import ModelEvaluation
from VeriLLM.components.plots import plot_language_scores
from VeriLLM.components.prediction import Predictor, build_reference_records
from VeriLLM.components.tokenization import Tokenizer
from VeriLLM.config.configuration import ConfigurationManager

STAGE_NAME = "Model Evaluation"


class ModelEvaluationPipeline:
    """Score the trained detector on gold validation and test."""

    def main(self, val, test):
        """
        Args:
            val (datasets.Dataset): gold validation, raw.
            test (datasets.Dataset): gold test, raw.

        Returns:
            dict: the metrics.
        """
        manager = ConfigurationManager()

        evaluation_config = manager.get_model_evaluation_config()
        tokenization_config = manager.get_tokenization_config()

        model_dir = str(evaluation_config.final_model_dir)

        hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)

        encoder = Tokenizer(tokenization_config, hf_tokenizer)

        predictor = Predictor(
            model,
            encoder,
            batch_size=evaluation_config.predict_batch_size,
        )

        val_refs = build_reference_records(val)
        test_refs = build_reference_records(test)

        metrics = ModelEvaluation(evaluation_config, predictor).run(
            val_refs, test_refs
        )

        plot_language_scores(
            metrics["test_by_language"],
            evaluation_config.plots_dir / "test_by_language.png",
            title="test: IoU and rho per language",
        )

        return metrics


if __name__ == "__main__":
    from VeriLLM.pipeline.stage_01_data_ingestion import DataIngestionPipeline

    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    _, val, test = DataIngestionPipeline().main()

    ModelEvaluationPipeline().main(val, test)

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
