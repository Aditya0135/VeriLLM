"""
Stage 03 - run the strict-prompt annotator ensemble.

**This is the only stage that spends Groq quota.** 3 models x 3,351 rows =
about 10,053 calls for a full pass. It is resumable and append-only, so an
interrupted run resumes exactly where it stopped.

Run with `--limit 5` first. A parse bug found at row 2,000 costs far more
quota than five calls do.

    python -m VeriLLM.pipeline.stage_03_annotation --limit 5
    python -m VeriLLM.pipeline.stage_03_annotation
"""

import argparse

from VeriLLM import logger
from VeriLLM.components.annotation import Annotator
from VeriLLM.config.configuration import ConfigurationManager

STAGE_NAME = "Annotation"


class AnnotationPipeline:
    """Collect three strict-prompt votes per sample."""

    def main(self, train, limit=None):
        """
        Args:
            train (datasets.Dataset): rows with `id`, `model_input`,
                `model_output_text` and `reference_answer`.
            limit (int | None): stop after this many newly annotated rows.

        Returns:
            int: number of rows newly annotated.
        """
        config = ConfigurationManager().get_annotation_config()

        return Annotator(config).run(train, limit=limit)


if __name__ == "__main__":
    from VeriLLM.pipeline.stage_01_data_ingestion import DataIngestionPipeline
    from VeriLLM.pipeline.stage_02_reference_retrieval import (
        ReferenceRetrievalPipeline,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many newly annotated rows (smoke test)",
    )
    parser.add_argument(
        "--allow-search",
        action="store_true",
        help="permit Tavily calls for uncached questions",
    )
    args = parser.parse_args()

    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    train, _, _ = DataIngestionPipeline().main()
    train = ReferenceRetrievalPipeline().main(train, allow_search=args.allow_search)

    written = AnnotationPipeline().main(train, limit=args.limit)

    logger.info(f">>>>>> stage {STAGE_NAME} completed - {written} new rows <<<<<<")
