"""
Training curves and per-language score charts.

Matplotlib only, saved to disk - nothing here needs a notebook or a display.
`matplotlib.use("Agg")` is set before pyplot is imported so this works headless.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from VeriLLM import logger  # noqa: E402


def get_loss_history(trainer):
    """
    Pull train and eval loss out of the Trainer's log history.

    The two are logged at different cadences - train loss every
    `logging_steps`, eval loss once per epoch - so they come back as separate
    (epoch, value) series rather than one aligned table.

    Args:
        trainer (transformers.Trainer): a trainer that has run.

    Returns:
        dict: `{"train": [(epoch, loss)], "eval": [(epoch, loss)],
                "metrics": {name: [(epoch, value)]}}`
    """
    train = []
    evaluation = []
    metrics = {"precision": [], "recall": [], "f1": []}

    for entry in trainer.state.log_history:

        epoch = entry.get("epoch")

        if epoch is None:
            continue

        if "loss" in entry:
            train.append((epoch, entry["loss"]))

        if "eval_loss" in entry:
            evaluation.append((epoch, entry["eval_loss"]))

        for name in metrics:
            key = f"eval_{name}"
            if key in entry:
                metrics[name].append((epoch, entry[key]))

    return {"train": train, "eval": evaluation, "metrics": metrics}


def plot_loss_curves(trainer, save_path):
    """
    Plot training/eval loss and the eval metrics side by side.

    Worth reading together: on this task `eval_loss` rises while f1 improves.
    Cross-entropy punishes confidence, f1 scores decisions - which is why
    checkpoint selection uses f1 and not loss.

    Args:
        trainer (transformers.Trainer): a trainer that has run.
        save_path (str | Path): where to write the PNG.

    Returns:
        str: the path written.
    """
    history = get_loss_history(trainer)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if history["train"]:
        axes[0].plot(*zip(*history["train"]), label="train", linewidth=1.5)

    if history["eval"]:
        axes[0].plot(
            *zip(*history["eval"]),
            label="eval (gold validation)",
            marker="o",
            linewidth=1.5,
        )

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for name, series in history["metrics"].items():
        if series:
            axes[1].plot(*zip(*series), label=name, marker="o", linewidth=1.5)

    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("token-level metrics on gold validation")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)

    logger.info(f"loss curves written to {save_path}")

    return str(save_path)


def plot_language_scores(results_by_language, save_path, title="per-language IoU"):
    """
    Bar chart of IoU and rho per language, with the macro average marked.

    Args:
        results_by_language (dict): output of `evaluate_by_language`.
        save_path (str | Path): where to write the PNG.
        title (str): chart title.

    Returns:
        str: the path written.
    """
    languages = sorted(k for k in results_by_language if k != "macro")

    ious = [results_by_language[lang]["iou"] for lang in languages]
    cors = [results_by_language[lang]["cor"] for lang in languages]

    positions = range(len(languages))
    width = 0.4

    figure, axis = plt.subplots(figsize=(max(8, len(languages) * 0.8), 4.5))

    axis.bar([p - width / 2 for p in positions], ious, width, label="IoU")
    axis.bar([p + width / 2 for p in positions], cors, width, label="rho")

    macro = results_by_language["macro"]

    axis.axhline(
        macro["iou"],
        linestyle="--",
        linewidth=1,
        label=f"macro IoU {macro['iou']:.3f}",
    )

    axis.set_xticks(list(positions))
    axis.set_xticklabels(languages)
    axis.set_ylabel("score")
    axis.set_title(title)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)

    logger.info(f"language chart written to {save_path}")

    return str(save_path)
