"""
Stage 02 - attach a reference answer to every training row, then normalize.

Effectively free: all 3,351 rows already have a reference in the six-variant
annotations file, and the corpus contains only 201 distinct questions. Tavily
is touched only if `allow_search` is on AND something is genuinely uncached.

Normalization runs **after** the reference is attached and on the training
split only. `normalize_split` refuses the gold splits outright, because their
human character offsets index the raw string (CLAUDE.md section 5.2).
"""

from VeriLLM import logger
from VeriLLM.components.reference_retrieval import ReferenceRetrieval
from VeriLLM.components.text_normalization import normalize_split
from VeriLLM.config.configuration import ConfigurationManager

STAGE_NAME = "Reference Retrieval"


class ReferenceRetrievalPipeline:
    """Attach references and normalize the training split."""

    def main(self, train, allow_search=False, normalize=True):
        """
        Args:
            train (datasets.Dataset): the `train_unlabeled` split, with ids.
            allow_search (bool): permit Tavily calls for uncached questions.
            normalize (bool): apply Unicode normalization afterwards.

        Returns:
            datasets.Dataset: train with `reference_answer`, normalized.
        """
        config = ConfigurationManager().get_reference_retrieval_config()

        train = ReferenceRetrieval(config).attach_references(
            train, allow_search=allow_search
        )

        if normalize:
            train = normalize_split(train, "train_unlabeled")

        return train


if __name__ == "__main__":
    from VeriLLM.pipeline.stage_01_data_ingestion import DataIngestionPipeline

    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    train, _, _ = DataIngestionPipeline().main()
    train = ReferenceRetrievalPipeline().main(train)

    with_reference = sum(1 for row in train if row["reference_answer"])

    logger.info(
        f">>>>>> stage {STAGE_NAME} completed - "
        f"{with_reference}/{len(train)} rows have a reference <<<<<<"
    )
