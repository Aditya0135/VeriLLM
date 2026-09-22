"""
VeriLLM entry point.

Runs the silver pipeline (stages 01-04) and then the supervised one
(stages 05-06).

Annotation is **off by default**: it is the only stage that spends Groq quota,
it takes hours, and the cache already holds the result of the last run. Pass
`--annotate` to include it.

    python main.py                  # aggregate cached votes, train, evaluate
    python main.py --annotate       # ... including a fresh annotation pass
    python main.py --eval-only      # score the saved checkpoint, nothing else
"""

import argparse
import sys

sys.path.insert(0, "src")

from VeriLLM import logger  # noqa: E402
from VeriLLM.pipeline.silver_pipeline import SilverPipeline  # noqa: E402
from VeriLLM.pipeline.supervised_pipeline import SupervisedPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotate", action="store_true",
                        help="run the annotation stage (costs Groq quota)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap newly annotated rows (smoke test)")
    parser.add_argument("--allow-search", action="store_true",
                        help="permit Tavily calls for uncached questions")
    parser.add_argument("--eval-only", action="store_true",
                        help="score the saved checkpoint without retraining")
    args = parser.parse_args()

    logger.info(">>>>>> VeriLLM started <<<<<<")

    silver, val, test = SilverPipeline().main(
        annotate=args.annotate,
        limit=args.limit,
        allow_search=args.allow_search,
    )

    metrics = SupervisedPipeline().main(
        silver=silver,
        val=val,
        test=test,
        train_model=not args.eval_only,
    )

    logger.info(
        f">>>>>> VeriLLM completed - test IoU {metrics['test']['iou']:.3f}, "
        f"rho {metrics['test']['cor']:.3f} <<<<<<"
    )


if __name__ == "__main__":
    main()
