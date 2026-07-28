from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
SCHEMA_FILE_PATH = Path("schema.yaml")

# A character is predicted hallucinated when its probability is > 0.5.
# Here probability is the BERT model's logit for every word.
# if those logits are > 0.5 => hallucinated
OFFICIAL_CUTOFF = 0.5

# Fixed by the HuggingFace convention: positions the loss must ignore.
LABEL_IGNORE_INDEX = -100
