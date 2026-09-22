"""
Stage 01 - load the three Mu-SHROOM splits.

Free. Downloads are cached by `datasets` after the first run.
"""

from VeriLLM import logger
from VeriLLM.components.data_ingestion import DataIngestion
from VeriLLM.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion"


class DataIngestionPipeline:
    """Load `train_unlabeled`, `validation` and `test`."""

    def main(self):
        """
        Returns:
            tuple[Dataset, Dataset, Dataset]: train, validation, test.
        """
        config = ConfigurationManager().get_data_ingestion_config()

        return DataIngestion(config).load_splits()


if __name__ == "__main__":
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    train, val, test = DataIngestionPipeline().main()

    logger.info(
        f">>>>>> stage {STAGE_NAME} completed - "
        f"{len(train)}/{len(val)}/{len(test)} rows <<<<<<"
    )
