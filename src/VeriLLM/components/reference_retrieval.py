"""
Reference answers for the annotator ensemble.

Three sources, cheapest first. Nothing here should ever spend Tavily quota on
a question that has already been answered once (CLAUDE.md section 0).

1. **`data/annotations.jsonl`** - the six-variant run stored the retrieved
   reference alongside every sample. All 3,351 rows have one, keyed by
   `sample_id`. This is the primary source and it is free.
2. **`data/reference_map.json`** - `{question: reference}`, 216 entries
   covering the 201 unique questions in the corpus.
3. **Tavily** - only for a question neither cache knows. Written back to the
   map immediately after each successful search, so an interrupted run never
   pays for the same question twice.

Note the shape of this data: 3,351 rows share just **201 distinct questions**,
so retrieval was always a 201-call problem, not a 3,351-call one.
"""

import os

from VeriLLM import logger
from VeriLLM.entity.config_entity import ReferenceRetrievalConfig
from VeriLLM.utils.common import load_json, load_jsonl, save_json


class ReferenceRetrieval:
    """
    Attach a `reference_answer` to every row, spending as little quota as possible.

    Args:
        config (ReferenceRetrievalConfig): cache path and search settings.
        prior_annotations_path (str | None): a previous annotations file to
            harvest references from. Defaults to `data/annotations.jsonl`.

    Usage::

        retrieval = ReferenceRetrieval(config)
        train = retrieval.attach_references(train)
    """

    def __init__(self, config: ReferenceRetrievalConfig,
                 prior_annotations_path="data/annotations.jsonl"):
        self.config = config
        self.prior_annotations_path = prior_annotations_path

        self.reference_map = {}
        self.by_sample_id = {}

        self._load_caches()

    def _load_caches(self):
        """Populate both lookup tables from disk."""
        if os.path.exists(self.config.reference_map_path):
            self.reference_map = load_json(self.config.reference_map_path)
            logger.info(
                f"reference_map: {len(self.reference_map)} cached questions"
            )

        if self.prior_annotations_path and os.path.exists(self.prior_annotations_path):

            for record in load_jsonl(self.prior_annotations_path):

                reference = record.get("reference_answer")

                if reference:
                    self.by_sample_id[record["sample_id"]] = reference

                    # a prior reference also fills a gap in the question map
                    question = record.get("model_input")

                    if question and question not in self.reference_map:
                        self.reference_map[question] = reference

            logger.info(
                f"prior annotations: references for {len(self.by_sample_id)} samples"
            )

    def lookup(self, sample_id, question):
        """
        Return a cached reference, or `None` if neither cache has one.

        Args:
            sample_id: the row id.
            question (str): `model_input`.

        Returns:
            str | None
        """
        reference = self.by_sample_id.get(sample_id)

        if reference:
            return reference

        return self.reference_map.get(question)

    def search(self, question):
        """
        Query Tavily for one question and cache the answer immediately.

        Only called for questions no cache knows. The write happens after every
        successful search, not at the end, so an interrupted run keeps what it
        paid for.

        Args:
            question (str): the query.

        Returns:
            str: the reference answer, or `""` if the search failed.
        """
        from tavily import TavilyClient

        client = TavilyClient(os.getenv("API_Tavily"))

        try:
            response = client.search(
                query=question,
                include_answer=self.config.include_answer,
                search_depth=self.config.search_depth,
            )

            answer = response.get("answer") or ""

        except Exception as error:
            logger.warning(f"Tavily search failed for {question!r}: {error}")
            return ""

        if answer:
            self.reference_map[question] = answer
            save_json(self.config.reference_map_path, self.reference_map)

        return answer

    def attach_references(self, dataset, allow_search=False):
        """
        Add a `reference_answer` column to a split.

        Args:
            dataset (datasets.Dataset): rows with `id` and `model_input`.
            allow_search (bool): permit Tavily calls for uncached questions.
                Defaults to False so a routine run can never spend quota by
                accident; the pipeline stage turns it on deliberately.

        Returns:
            datasets.Dataset: the same rows plus `reference_answer`.
        """
        missing = []

        def add_reference(example):
            reference = self.lookup(example["id"], example["model_input"])

            if reference is None:
                missing.append(example["model_input"])
                reference = ""

            example["reference_answer"] = reference
            return example

        # load_from_cache_file=False: this closure records misses as a side
        # effect, and a cached map would skip it and report none.
        dataset = dataset.map(add_reference, load_from_cache_file=False)

        unique_missing = sorted(set(missing))

        if unique_missing:
            logger.info(
                f"{len(missing)} rows ({len(unique_missing)} unique questions) "
                f"have no cached reference"
            )

            if allow_search:
                logger.info(f"searching Tavily for {len(unique_missing)} questions")

                for question in unique_missing:
                    self.search(question)

                missing.clear()
                dataset = dataset.map(add_reference, load_from_cache_file=False)

                if missing:
                    logger.warning(
                        f"{len(set(missing))} questions still have no reference "
                        f"after searching; those rows carry an empty one"
                    )

        else:
            logger.info("every row has a cached reference - no search needed")

        return dataset
