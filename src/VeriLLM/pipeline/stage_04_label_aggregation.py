"""
Stage 04 - aggregate annotator votes into official-format silver labels.

Free and re-runnable. Changing `hard_label_threshold` in params.yaml and
re-running this stage is the cheapest experiment in the project: with 3 votes,
0.5 means 2-of-3 (majority) and 0.34 means 1-of-3 (union), which is the knob
to reach for if the silver positive rate lands far from gold's ~40%.
"""

from VeriLLM import logger
from VeriLLM.components.label_aggregation import LabelAggregation
from VeriLLM.config.configuration import ConfigurationManager

STAGE_NAME = "Label Aggregation"


class LabelAggregationPipeline:
    """Build the silver dataset and save it to disk."""

    def main(self, train, save=True):
        """
        Args:
            train (datasets.Dataset): rows with references attached.
            save (bool): write the result to `config.dataset_path`.

        Returns:
            datasets.Dataset: the silver-labelled rows.
        """
        config = ConfigurationManager().get_label_aggregation_config()

        silver = LabelAggregation(config).run(train)

        if save:
            silver.save_to_disk(str(config.dataset_path))
            logger.info(f"silver dataset saved to {config.dataset_path}")

        return silver


if __name__ == "__main__":
    from VeriLLM.pipeline.stage_01_data_ingestion import DataIngestionPipeline
    from VeriLLM.pipeline.stage_02_reference_retrieval import (
        ReferenceRetrievalPipeline,
    )

    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    train, _, _ = DataIngestionPipeline().main()
    train = ReferenceRetrievalPipeline().main(train)

    silver = LabelAggregationPipeline().main(train)

    logger.info(f">>>>>> stage {STAGE_NAME} completed - {len(silver)} rows <<<<<<")
