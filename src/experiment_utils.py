"""Shared helpers for ViCTSD label comparison and model notebooks."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


TEXT_COL = "Comment"
LABEL_COL = "Constructiveness"
ID_COL = "Unnamed: 0"
LABEL_NAMES = ["Non-constructive", "Constructive"]
DATASET_LABELS = ("Original labels", "AI re-annotated labels")


def ensure_project_on_path(repo_url: str = "https://github.com/lechihoang/multi-agent-annotation.git") -> Path:
    """Make notebooks runnable from local, Colab, or Kaggle environments."""
    cwd = Path.cwd()
    candidates = [cwd, *cwd.parents]
    root = next((path for path in candidates if (path / "src").exists() and (path / "data").exists()), None)

    if root is None:
        if not Path("multi-agent-annotation").exists():
            subprocess.run(["git", "clone", repo_url], check=True)
        root = Path("multi-agent-annotation").resolve()
        os.chdir(root / "notebooks")
        root = root.resolve()

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def get_data_dir(root: Path | None = None) -> Path:
    root = root or ensure_project_on_path()
    candidates = [root / "data", Path("../data"), Path("data")]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Cannot find data directory.")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).lower().split())


def load_victsd_splits(
    data_dir: str | Path | None = None,
    *,
    drop_unlabeled_reannotated: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load original, re-annotated, and test splits with consistent null handling."""
    data_path = Path(data_dir) if data_dir is not None else get_data_dir()
    files = {
        "train_orig": "ViCTSD_train.csv",
        "valid_orig": "ViCTSD_valid.csv",
        "test": "ViCTSD_test.csv",
        "train_new": "ViCTSD_train_reannotated.csv",
        "valid_new": "ViCTSD_valid_reannotated.csv",
    }
    splits = {name: pd.read_csv(data_path / filename).fillna("") for name, filename in files.items()}

    if drop_unlabeled_reannotated:
        for name in ("train_new", "valid_new"):
            splits[name] = filter_valid_labels(splits[name]).reset_index(drop=True)
    else:
        for name, frame in splits.items():
            validate_binary_labels(frame, dataset_name=name)

    print(
        "Loaded splits: "
        f"train_orig={len(splits['train_orig'])}, train_new={len(splits['train_new'])}, "
        f"valid_orig={len(splits['valid_orig'])}, valid_new={len(splits['valid_new'])}, "
        f"test={len(splits['test'])}"
    )
    return splits


def filter_valid_labels(df: pd.DataFrame, label_col: str = LABEL_COL) -> pd.DataFrame:
    labels = pd.to_numeric(df[label_col], errors="coerce")
    return df[labels.isin([0, 1])].copy()


def validate_binary_labels(
    df: pd.DataFrame,
    *,
    dataset_name: str = "dataset",
    label_col: str = LABEL_COL,
) -> None:
    labels = pd.to_numeric(df[label_col], errors="coerce")
    invalid = sorted(labels[~labels.isin([0, 1])].dropna().unique().tolist())
    missing = int(labels.isna().sum())
    if invalid or missing:
        raise ValueError(
            f"{dataset_name} has non-binary labels in {label_col}: "
            f"invalid={invalid}, missing={missing}. "
            "Use drop_unlabeled_reannotated=True only for incomplete generated datasets."
        )


def align_label_frames(
    original: pd.DataFrame,
    reannotated: pd.DataFrame,
    *,
    text_col: str = TEXT_COL,
    label_col: str = LABEL_COL,
    id_col: str = ID_COL,
) -> pd.DataFrame:
    """Align two label frames robustly; prefer row id, then fall back to text."""
    original = filter_valid_labels(original, label_col)
    reannotated = filter_valid_labels(reannotated, label_col)
    join_col = id_col if id_col in original.columns and id_col in reannotated.columns else text_col

    left = original[[join_col, text_col, label_col]].rename(columns={label_col: "original_label"})
    right_cols = [join_col, label_col] if join_col == text_col else [join_col, text_col, label_col]
    right = reannotated[right_cols].rename(columns={label_col: "reannotated_label"})

    aligned = left.merge(right, on=join_col, how="inner", suffixes=("", "_reannotated"))
    if f"{text_col}_reannotated" in aligned.columns:
        aligned = aligned.drop(columns=[f"{text_col}_reannotated"])
    aligned["original_label"] = aligned["original_label"].astype(int)
    aligned["reannotated_label"] = aligned["reannotated_label"].astype(int)
    return aligned


def summarize_label_changes(aligned: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    cm = confusion_matrix(
        aligned["original_label"],
        aligned["reannotated_label"],
        labels=[0, 1],
    )
    total = int(cm.sum())
    summary = pd.DataFrame(
        [
            ("same_0", cm[0, 0], cm[0, 0] / total),
            ("same_1", cm[1, 1], cm[1, 1] / total),
            ("0_to_1", cm[0, 1], cm[0, 1] / total),
            ("1_to_0", cm[1, 0], cm[1, 0] / total),
            ("agreement", cm[0, 0] + cm[1, 1], (cm[0, 0] + cm[1, 1]) / total),
            ("disagreement", cm[0, 1] + cm[1, 0], (cm[0, 1] + cm[1, 0]) / total),
        ],
        columns=["case", "count", "rate"],
    )
    return summary, cm


def print_label_change_summary(summary: pd.DataFrame) -> None:
    printable = summary.copy()
    printable["rate"] = printable["rate"].map(lambda value: f"{value:.2%}")
    print(printable.to_string(index=False))


def plot_label_confusion(cm: np.ndarray, title: str = "Original vs AI re-annotated labels") -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"{name} ({idx})" for idx, name in enumerate(LABEL_NAMES)],
        yticklabels=[f"{name} ({idx})" for idx, name in enumerate(LABEL_NAMES)],
    )
    plt.xlabel("AI re-annotated label")
    plt.ylabel("Original label")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def changed_label_examples(
    aligned: pd.DataFrame,
    *,
    original_label: int,
    reannotated_label: int,
    n: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    mask = (aligned["original_label"] == original_label) & (
        aligned["reannotated_label"] == reannotated_label
    )
    subset = aligned.loc[mask, [TEXT_COL, "original_label", "reannotated_label"]]
    if subset.empty:
        return subset
    return subset.sample(min(n, len(subset)), random_state=random_state)


def make_text_xy(df: pd.DataFrame, *, text_col: str = TEXT_COL, label_col: str = LABEL_COL) -> tuple[pd.Series, pd.Series]:
    valid = filter_valid_labels(df, label_col)
    return valid[text_col].map(normalize_text), valid[label_col].astype(int)


def find_or_download_phow2v(
    filename: str = "word2vec_vi_syllables_300dims.txt",
    *,
    data_dir: str | Path | None = None,
) -> Path:
    """Find PhoW2V in common Kaggle/local paths; download the Kaggle dataset if needed."""
    search_roots = [Path.cwd(), Path("../data"), Path("/kaggle/input")]
    if data_dir is not None:
        search_roots.insert(0, Path(data_dir))

    for root in search_roots:
        if root.exists():
            matches = list(root.glob(f"**/{filename}"))
            if matches:
                return matches[0]

    archive = Path("phow2v.zip")
    subprocess.run(
        ["curl", "-L", "-o", str(archive), "https://www.kaggle.com/api/v1/datasets/download/toxuandong/phow2v"],
        check=True,
    )
    with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extractall(".")
    matches = list(Path.cwd().glob(f"**/{filename}"))
    if not matches:
        raise FileNotFoundError(f"Downloaded PhoW2V archive but could not find {filename}.")
    return matches[0]


def prepare_keras_text_data(
    splits: dict[str, pd.DataFrame],
    *,
    max_words: int = 10000,
    max_len: int = 100,
):
    """Build one tokenizer and sequence arrays shared by original/re-annotated runs."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=max_words, oov_token="<UNK>")
    tokenizer.fit_on_texts(splits["train_orig"][TEXT_COL].map(normalize_text))

    def seq(frame: pd.DataFrame) -> np.ndarray:
        texts = frame[TEXT_COL].map(normalize_text)
        return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len)

    return {
        "tokenizer": tokenizer,
        "X_train_original": seq(splits["train_orig"]),
        "y_train_original": splits["train_orig"][LABEL_COL].astype(int).values,
        "X_train_reannotated": seq(splits["train_new"]),
        "y_train_reannotated": splits["train_new"][LABEL_COL].astype(int).values,
        "X_test": seq(splits["test"]),
        "y_test": splits["test"][LABEL_COL].astype(int).values,
        "test_texts": splits["test"][TEXT_COL].map(normalize_text).values,
    }


def build_embedding_matrix(
    tokenizer,
    *,
    max_words: int = 10000,
    embedding_dim: int = 300,
    phow2v_path: str | Path | None = None,
    random_state: int = 42,
) -> np.ndarray:
    from gensim.models import KeyedVectors

    rng = np.random.default_rng(random_state)
    path = Path(phow2v_path) if phow2v_path is not None else find_or_download_phow2v()
    print(f"Loading PhoW2V from: {path}")
    w2v_model = KeyedVectors.load_word2vec_format(path, binary=False)

    matrix = np.zeros((max_words, embedding_dim))
    found = 0
    for word, index in tokenizer.word_index.items():
        if index >= max_words:
            continue
        if word in w2v_model:
            matrix[index] = w2v_model[word]
            found += 1
        else:
            matrix[index] = rng.normal(scale=0.6, size=(embedding_dim,))

    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    print(f"Matched {found}/{vocab_size} tokenizer entries with PhoW2V.")
    return matrix


def train_predict_keras_pair(
    *,
    build_model: Callable[[], object],
    X_train_original,
    y_train_original,
    X_train_reannotated,
    y_train_reannotated,
    X_test,
    y_test,
    model_name: str,
    epochs: int = 5,
    batch_size: int = 32,
    validation_split: float = 0.1,
):
    print(f"Training {model_name} on original labels...")
    model_original = build_model()
    model_original.fit(
        X_train_original,
        y_train_original,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )
    y_pred_original = (model_original.predict(X_test) > 0.5).astype(int).flatten()
    evaluate_predictions(f"{model_name} ({DATASET_LABELS[0]})", y_test, y_pred_original)

    print(f"\nTraining {model_name} on AI re-annotated labels...")
    model_reannotated = build_model()
    model_reannotated.fit(
        X_train_reannotated,
        y_train_reannotated,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )
    y_pred_reannotated = (model_reannotated.predict(X_test) > 0.5).astype(int).flatten()
    evaluate_predictions(f"{model_name} ({DATASET_LABELS[1]})", y_test, y_pred_reannotated)

    comparison = compare_predictions(model_name, y_test, y_pred_original, y_pred_reannotated)
    return model_original, model_reannotated, y_pred_original, y_pred_reannotated, comparison


def prediction_metrics(y_true, y_pred, *, prefix: str = "") -> dict[str, float]:
    prefix = f"{prefix}_" if prefix else ""
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_predictions(model_name: str, y_true, y_pred) -> dict[str, float]:
    metrics = prediction_metrics(y_true, y_pred)
    print(f"--- {model_name} ---")
    print(
        f"Accuracy={metrics['accuracy']:.4f} | "
        f"F1 macro={metrics['f1_macro']:.4f} | "
        f"Precision macro={metrics['precision_macro']:.4f} | "
        f"Recall macro={metrics['recall_macro']:.4f}"
    )
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))
    return metrics


def compare_predictions(
    model_name: str,
    y_true,
    y_pred_original,
    y_pred_reannotated,
    *,
    plot: bool = True,
) -> pd.DataFrame:
    metrics = {
        "accuracy": "Accuracy",
        "f1_weighted": "F1 weighted",
        "f1_macro": "F1 macro",
        "precision_macro": "Precision macro",
        "recall_macro": "Recall macro",
    }
    original = prediction_metrics(y_true, y_pred_original)
    reannotated = prediction_metrics(y_true, y_pred_reannotated)
    rows = [
        {
            "Metric": label,
            DATASET_LABELS[0]: original[metric],
            DATASET_LABELS[1]: reannotated[metric],
            "Delta": reannotated[metric] - original[metric],
        }
        for metric, label in metrics.items()
    ]
    df = pd.DataFrame(rows)

    title = f"Compare result: {model_name}"
    print(f"\n{title}")
    try:
        from IPython.display import display

        display(df.style.set_caption(title).format(precision=4))
    except Exception:
        print(df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if plot:
        plot_metric_comparison(df, model_name)
    return df


def plot_metric_comparison(results_df: pd.DataFrame, model_name: str) -> None:
    metric_df = results_df.melt(
        id_vars=["Metric"],
        value_vars=list(DATASET_LABELS),
        var_name="Label set",
        value_name="Score",
    )
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=metric_df,
        x="Metric",
        y="Score",
        hue="Label set",
        palette=["#2f80ed", "#eb5757"],
    )
    ax.set_title(f"Compare result: {model_name}")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()


def train_predict_pair(
    *,
    build_model: Callable[[], object],
    X_train_original,
    y_train_original,
    X_train_reannotated,
    y_train_reannotated,
    X_test,
    y_test,
    model_name: str,
    predict_fn: Callable[[object, object], np.ndarray] | None = None,
    plot: bool = True,
) -> tuple[object, object, np.ndarray, np.ndarray, pd.DataFrame]:
    """Train one model family on two label sets and compare both predictions."""
    predict_fn = predict_fn or (lambda model, X: model.predict(X))

    print(f"Training {model_name} on original labels...")
    model_original = build_model()
    model_original.fit(X_train_original, y_train_original)
    y_pred_original = predict_fn(model_original, X_test)
    evaluate_predictions(f"{model_name} ({DATASET_LABELS[0]})", y_test, y_pred_original)

    print(f"\nTraining {model_name} on AI re-annotated labels...")
    model_reannotated = build_model()
    model_reannotated.fit(X_train_reannotated, y_train_reannotated)
    y_pred_reannotated = predict_fn(model_reannotated, X_test)
    evaluate_predictions(f"{model_name} ({DATASET_LABELS[1]})", y_test, y_pred_reannotated)

    comparison = compare_predictions(model_name, y_test, y_pred_original, y_pred_reannotated, plot=plot)
    return model_original, model_reannotated, y_pred_original, y_pred_reannotated, comparison


def compare_prediction_disagreements(texts, y_true, y_pred_original, y_pred_reannotated) -> pd.DataFrame:
    diff_mask = np.asarray(y_pred_original) != np.asarray(y_pred_reannotated)
    diff_df = pd.DataFrame(
        {
            "text": np.asarray(texts)[diff_mask],
            "true_label": np.asarray(y_true)[diff_mask],
            "pred_original": np.asarray(y_pred_original)[diff_mask],
            "pred_reannotated": np.asarray(y_pred_reannotated)[diff_mask],
        }
    )
    print(f"Prediction disagreements: {len(diff_df)} / {len(y_true)}")
    return diff_df
