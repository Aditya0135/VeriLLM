"""
The weak-supervision half: stages 01-04.

    ingestion -> reference retrieval -> annotation -> label aggregation

**Every API call in the project lives in this pipeline.** That is the point of
the split: `supervised_pipeline` can be re-run without limit, while this one
spends quota that does not come back (CLAUDE.md section 0).

Stage 03 is the expensive step - 3 models x 3,351 rows, about 10,000 Groq
calls - and it is skipped by default. Pass `annotate=True` only when you mean
it. Everything else here is free, and the annotation loop is resumable, so a
run that stops halfway loses nothing.

    python -m VeriLLM.pipeline.silver_pipeline               # aggregate only
    python -m VeriLLM.pipeline.silver_pipeline --annotate    # full, ~10k calls
"""

import argparse

from VeriLLM import logger
from VeriLLM.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from VeriLLM.pipeline.stage_02_reference_retrieval import ReferenceRetrievalPipeline
from VeriLLM.pipeline.stage_03_annotation import AnnotationPipeline
from VeriLLM.pipeline.stage_04_label_aggregation import LabelAggregationPipeline


class SilverPipeline:
    """Build the silver-labelled training set."""

    def main(self, annotate=False, limit=None, allow_search=False):
        """
        Args:
            annotate (bool): run stage 03. Costs Groq quota.
            limit (int | None): cap on newly annotated rows (smoke test).
            allow_search (bool): permit Tavily calls for uncached questions.

        Returns:
            tuple: (silver dataset, validation split, test split).
        """
        train, val, test = DataIngestionPipeline().main()

        train = ReferenceRetrievalPipeline().main(
            train, allow_search=allow_search
        )

        if annotate:
            AnnotationPipeline().main(train, limit=limit)
        else:
            logger.info(
                "skipping annotation (stage 03) - pass --annotate to run it"
            )

        silver = LabelAggregationPipeline().main(train)

        return silver, val, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotate", action="store_true",
                        help="run the annotation stage (costs Groq quota)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap newly annotated rows")
    parser.add_argument("--allow-search", action="store_true",
                        help="permit Tavily calls for uncached questions")
    args = parser.parse_args()

    logger.info(">>>>>> SILVER PIPELINE started <<<<<<")

    silver, _, _ = SilverPipeline().main(
        annotate=args.annotate,
        limit=args.limit,
        allow_search=args.allow_search,
    )

    logger.info(f">>>>>> SILVER PIPELINE completed - {len(silver)} rows <<<<<<")
