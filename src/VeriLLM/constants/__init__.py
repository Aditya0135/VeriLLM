from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# The shared task's scoring rule: a character counts as predicted-hallucinated
# when its probability is strictly greater than 0.5 (scorer.py,
# recompute_hard_labels). This is NOT the same decision as the threshold that
# binarises our silver *training* labels - that one is a vote over annotators.
OFFICIAL_CUTOFF = 0.5

# Fixed by the HuggingFace convention: positions the loss must ignore.
LABEL_IGNORE_INDEX = -100

# Test-only languages with no train or validation data; every score on these
# rows is zero-shot transfer.
SURPRISE_LANGUAGES = ("CA", "CS", "EU", "FA")
