"""
Load the Mu-SHROOM splits.

No module-level side effects: importing this must not download 2 GB. Everything
happens inside `DataIngestion.load_splits()`.
"""

from datasets import load_dataset

from VeriLLM import logger
from VeriLLM.entity.config_entity import DataIngestionConfig


def add_id(example, idx):
    """
    Attach a sequential integer id to a dataset row.

    `train_unlabeled` ships with `id = None` on every row, so an id has to be
    assigned before annotations can be matched back to their samples.

    Args:
        example (dict): one dataset row.
        idx (int): the row index, supplied by `Dataset.map(with_indices=True)`.

    Returns:
        dict: the same row with `example["id"]` set to `idx`.

    Example:
        >>> add_id({"lang": "EN"}, 7)
        {'lang': 'EN', 'id': 7}

    Usage::

        train = train.map(add_id, with_indices=True)
    """
    example["id"] = idx
    return example


class DataIngestion:
    """
    Fetch the three Mu-SHROOM splits and index the unlabeled one.

    Args:
        config (DataIngestionConfig): dataset name and config name.

    Usage::

        train, val, test = DataIngestion(config).load_splits()
        # -> 3351, 499, 1902
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_splits(self):
        """
        Return `(train_unlabeled, validation, test)`.

        Only `train_unlabeled` is given ids. The gold splits already carry
        string ids like `val-en-1`, and overwriting them would break the join
        between reference records and prediction records at scoring time.

        The gold splits are returned **raw**: no normalization of any kind.
        Every gold offset indexes the untouched string (CLAUDE.md section 5.2).

        Returns:
            tuple[Dataset, Dataset, Dataset]
        """
        dataset = load_dataset(self.config.dataset_name, self.config.config_name)

        train = dataset["train_unlabeled"].map(add_id, with_indices=True)
        val = dataset["validation"]
        test = dataset["test"]

        logger.info(
            f"loaded splits - train_unlabeled: {len(train)}, "
            f"validation: {len(val)}, test: {len(test)}"
        )

        return train, val, test
