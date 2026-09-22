"""
The supervised-learning half: stages 05-06.

    tokenize -> train -> official evaluation

**Zero API calls.** Nothing in this pipeline can touch `annotations.jsonl`,
`annotations_strict.jsonl` or `reference_map.json`, so it is safe to re-run as
often as you like - a different encoder, a different threshold, a different
number of epochs. That is the whole reason the split falls here.

    python -m VeriLLM.pipeline.supervised_pipeline
    python -m VeriLLM.pipeline.supervised_pipeline --eval-only
"""

import argparse

from VeriLLM import logger
from VeriLLM.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from VeriLLM.pipeline.stage_05_model_trainer import ModelTrainerPipeline
from VeriLLM.pipeline.stage_06_model_evaluation import ModelEvaluationPipeline


class SupervisedPipeline:
    """Train the detector on silver labels and score it against gold."""

    def main(self, silver=None, val=None, test=None, train_model=True):
        """
        Args:
            silver (datasets.Dataset | None): silver rows; loaded from disk
                when omitted.
            val (datasets.Dataset | None): gold validation; loaded when omitted.
            test (datasets.Dataset | None): gold test; loaded when omitted.
            train_model (bool): False scores the saved checkpoint as-is.

        Returns:
            dict: the metrics.
        """
        if val is None or test is None:
            _, val, test = DataIngestionPipeline().main()

        if train_model:
            ModelTrainerPipeline().main(val, silver=silver)
        else:
            logger.info("skipping training - scoring the saved checkpoint")

        return ModelEvaluationPipeline().main(val, test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-only", action="store_true",
                        help="score the saved checkpoint without retraining")
    args = parser.parse_args()

    logger.info(">>>>>> SUPERVISED PIPELINE started <<<<<<")

    metrics = SupervisedPipeline().main(train_model=not args.eval_only)

    logger.info(
        f">>>>>> SUPERVISED PIPELINE completed - "
        f"test IoU {metrics['test']['iou']:.3f}, "
        f"rho {metrics['test']['cor']:.3f} <<<<<<"
    )
