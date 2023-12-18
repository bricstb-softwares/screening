import logging
import typing as T
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, confusion_matrix, roc_curve
from utils import convnets


def report_training(results_path, output_name):
    # type: (Path, str) -> None
    filepath = results_path / "history.pkl"
    if not filepath.is_file():
        raise FileNotFoundError(f"File {filepath} not found.")
    history = pd.read_pickle(filepath)
    output_path = results_path / f"{output_name}.png"
    output_path.parent.mkdir(exist_ok=True)
    plot_training_history(history, output_path)


def calculate_performance_metrics(data, results_path):
    # type: (pd.DataFrame, Path) -> dict[str, T.Any]
    filepath = results_path / "parameters.pkl"
    if not filepath.is_file():
        raise FileNotFoundError(f"File {filepath} not found.")
    params = pd.read_pickle(filepath)
    model = convnets.create_cnn(params["image_shape"])

    filepath = results_path / "model_weights.pkl"
    if not filepath.is_file():
        raise FileNotFoundError(f"File {filepath} not found.")
    model_weights = pd.read_pickle(filepath)
    model.set_weights(model_weights)

    ds = convnets.build_dataset(data, params["image_shape"], batch_size=128)
    metrics = calculate_metrics(ds, data, model)

    return metrics


def get_test_dataframe(metadata_path):
    # type: (str) -> pd.DataFrame
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata["dataset"] == "test"]
    metadata = metadata.rename(
        columns={
            "raw_image_path": "path",
            "raw_image_md5": "hash",
            "target": "label",
            "test": "fold",
        }
    )
    cols = ["hash", "path", "label", "age", "sex", "comment", "fold"]
    metadata = metadata[cols].drop_duplicates(ignore_index=True)

    return metadata


def plot_training_history(history, output_path):
    # type: (dict[str, list], Path) -> None
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    axs.plot(history["loss"], label="Train")
    axs.plot(history["val_loss"], label="Valid")
    axs.axvline(
        history["best_epoch"], color="r", alpha=0.6, linestyle="--", label="Best Epoch"
    )
    axs.legend()
    axs.set_xlabel("Iterations")
    axs.set_ylabel("Cross Entropy Loss")
    axs.spines["left"].set_position(("outward", 10))
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.get_xaxis().tick_bottom()
    axs.get_yaxis().tick_left()

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()



def calculate_metrics(ds, df, model, label='', min_sensitivity=0.9, min_specificity=0.7):
    # type: (tf.data.Dataset, pd.DataFrame, tf.python.keras.models.Sequential) -> dict[str, T.Any]
    metrics = dict()
    aux_metrics = dict()

    y_true = df["label"].values.astype(int)
    y_prob = model.predict(ds).squeeze()
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    metrics["fpr"+label] = fpr
    metrics["tpr"+label] = tpr

    # calculate the total SP & AUC
    sp_values = np.sqrt(np.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))
    sp_argmax = np.argmax(sp_values)
    metrics["SP"+label] = sp_values[sp_argmax]
    metrics["AUC"+label] = auc(fpr, tpr)

    # calculate metrics inside the WHO area
    who_selection = (tpr >= min_sensitivity) & ((1 - fpr) >= min_specificity)
    if np.any(who_selection):
        metrics["WHO"+label] = True
        sp_values = sp_values[who_selection]
        sp_argmax = np.argmax(sp_values)
        metrics["SP_WHO"+label] = sp_values[sp_argmax]
        if sum(who_selection) <= 1:
            metrics["AUC_WHO"+label] = None
        else:
            metrics["AUC_WHO"+label] = auc(fpr[who_selection], tpr[who_selection])
        metrics["threshold"+label] = thresholds[who_selection][sp_argmax]
    else:
        metrics["WHO"+label] = False
        metrics["SP_WHO"+label] = 0.0
        metrics["AUC_WHO"+label] = 0.0
        metrics["threshold"+label] = thresholds[sp_argmax]

    # make predictions using the WHO threshold
    y_pred = (y_prob >= metrics["threshold"+label]).astype(int)
    y_true = df["label"].values.astype(int)

    # calculate predictions
    aux_metrics["image_name"] = df["path"].apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )
    aux_metrics["y_prob"] = y_prob
    aux_metrics["y_pred"] = y_pred
    metrics["predictions"] = pd.DataFrame.from_dict(aux_metrics)

    # confusion matrix and metrics
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    metrics["sensitivity"+label] = tp / (tp + fn)
    metrics["specificity"+label] = tn / (tn + fp)
    metrics["precision"+label] = tp / (tp + fp)
    metrics["recall"+label] = tp / (tp + fn)

    return metrics



def build_cross_val_metrics(cv_metrics):
    # type: (dict) -> pd.DataFrame
    results = {
        "SP": [],
        "SP_WHO": [],
        "AUC": [],
        "AUC_WHO": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "recall": [],
        "WHO": [],
        "sort": [],
    }
    for metric in results:
        for fold in cv_metrics:
            results[metric].append(cv_metrics[fold][metric])

    return pd.DataFrame(results)


def plot_roc_curve(true_positive_rate, upper_error, lower_error, output_path):
    # type: (list[float], list[float], list[float], str) -> None
    mean_fpr = np.linspace(0, 1, 100)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    plt.fill_between(
        mean_fpr, lower_error, upper_error, label="Uncertainty", color="lightblue"
    )
    plt.plot(mean_fpr, true_positive_rate, color="blue", label="Mean ROC", alpha=0.8)
    plt.vlines(0.3, 0, 1, colors="red", alpha=0.6, linestyles="--")
    plt.hlines(0.9, 0, 1, colors="red", alpha=0.6, linestyles="--")
    axs.spines["left"].set_position(("outward", 10))
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.get_xaxis().tick_bottom()
    axs.get_yaxis().tick_left()
    plt.ylabel("TPR (Sensitivity)")
    plt.xlabel("FPR (1 - Specificity)")
    plt.legend(loc="lower right")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()





