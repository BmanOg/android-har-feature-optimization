#!/usr/bin/env python3
"""Run the development experiments for Elijah Ogunniran's COSC 750 Stage 3 project.

Every reported tuning and feature-selection choice stays inside nested,
participant-grouped development folds. The default command does not load or
score the official HAPT test partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from joblib import Memory
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/341/"
    "smartphone%2Bbased%2Brecognition%2Bof%2Bhuman%2Bactivities%2Band%2B"
    "postural%2Btransitions.zip"
)
DATASET_SHA256 = "4ac4ae064227c07045a99551876b54d204837e995e298b6488559e209b3deb09"
LABELS = np.arange(1, 13, dtype=int)
DAILY_LABELS = np.arange(1, 7, dtype=int)
TRANSITION_LABELS = np.arange(7, 13, dtype=int)
ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
    7: "STAND_TO_SIT",
    8: "SIT_TO_STAND",
    9: "SIT_TO_LIE",
    10: "LIE_TO_SIT",
    11: "STAND_TO_LIE",
    12: "LIE_TO_STAND",
}
OUTER_SEED = 2026
INNER_SEED_BASE = 7300
MI_SEED = 2026
PRIMARY_NONINFERIORITY_MARGIN = 0.03
SENSITIVITY_MARGIN = 0.02
DIAGNOSTIC_LINEAR_C = 0.01
FEATURE_SWEEP_C_GRID = [0.001, 0.01, 0.1]
BOOTSTRAP_SEED = 2026
BOOTSTRAP_REPLICATES = 10_000
OFFICIAL_TEST_LOCK_TOKEN = "FROZEN_PARETO_SOLUTIONS"


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    subject_train: np.ndarray
    x_test: np.ndarray | None
    y_test: np.ndarray | None
    subject_test: np.ndarray | None
    feature_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/hapt"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Cache deterministic pipeline transforms during grid search.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract the official UCI archive when data files are absent.",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller grids for a pipeline smoke test. Do not cite quick-mode results.",
    )
    parser.add_argument(
        "--unlock-official-test",
        action="store_true",
        help=(
            "Later-stage descriptive use only: load and evaluate the official test "
            "partition. Requires --test-lock-token after Pareto representatives are frozen."
        ),
    )
    parser.add_argument(
        "--test-lock-token",
        default="",
        help="Required token for final-stage official-test access.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset(data_dir: Path, allow_download: bool, include_test: bool = False) -> None:
    required = [
        data_dir / "Train" / "X_train.txt",
        data_dir / "Train" / "y_train.txt",
        data_dir / "Train" / "subject_id_train.txt",
        data_dir / "features.txt",
        data_dir / "activity_labels.txt",
    ]
    if include_test:
        required.extend(
            [
                data_dir / "Test" / "X_test.txt",
                data_dir / "Test" / "y_test.txt",
                data_dir / "Test" / "subject_id_test.txt",
            ]
        )
    if all(path.is_file() for path in required):
        return
    if not allow_download:
        missing = "\n".join(str(path) for path in required if not path.is_file())
        raise FileNotFoundError(
            "HAPT files are missing. Re-run with --download or place the official files here:\n"
            + missing
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir.parent / "hapt-official.zip"
    print(f"Downloading official HAPT archive to {archive}")
    with urllib.request.urlopen(DATASET_URL, timeout=300) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out)
    actual = sha256(archive)
    if actual != DATASET_SHA256:
        raise RuntimeError(f"Dataset checksum mismatch: expected {DATASET_SHA256}, got {actual}")
    wanted_prefixes = ("Train/", "Test/")
    wanted_files = {"features.txt", "activity_labels.txt", "features_info.txt", "README.txt"}
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.namelist():
            if member.startswith(wanted_prefixes) or member in wanted_files:
                zipped.extract(member, data_dir)


def _load_vector(path: Path, dtype: Any) -> np.ndarray:
    return np.loadtxt(path, dtype=dtype)


def load_dataset(data_dir: Path, include_test: bool = False) -> Dataset:
    x_train = np.loadtxt(data_dir / "Train" / "X_train.txt", dtype=np.float32)
    y_train = _load_vector(data_dir / "Train" / "y_train.txt", np.int16)
    subject_train = _load_vector(data_dir / "Train" / "subject_id_train.txt", np.int16)
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    subject_test: np.ndarray | None = None
    if include_test:
        x_test = np.loadtxt(data_dir / "Test" / "X_test.txt", dtype=np.float32)
        y_test = _load_vector(data_dir / "Test" / "y_test.txt", np.int16)
        subject_test = _load_vector(data_dir / "Test" / "subject_id_test.txt", np.int16)
    feature_names: list[str] = []
    for line in (data_dir / "features.txt").read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        if len(parts) == 2 and parts[0].isdigit():
            if int(parts[0]) != len(feature_names) + 1:
                raise ValueError("Feature indices are not consecutive")
            feature_names.append(parts[1])
        else:
            feature_names.append(parts[0])
    if x_train.shape != (7767, 561):
        raise ValueError(f"Unexpected training matrix shape: {x_train.shape}")
    if include_test:
        assert x_test is not None and subject_test is not None
        if x_test.shape != (3162, 561):
            raise ValueError(f"Unexpected test matrix shape: {x_test.shape}")
        if set(np.unique(subject_train)) & set(np.unique(subject_test)):
            raise ValueError("Official train and test participant IDs overlap")
    return Dataset(
        x_train=x_train,
        y_train=y_train,
        subject_train=subject_train,
        x_test=x_test,
        y_test=y_test,
        subject_test=subject_test,
        feature_names=feature_names,
    )


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
        "daily_macro_f1": float(
            f1_score(y_true, y_pred, labels=DAILY_LABELS, average="macro", zero_division=0)
        ),
        "transition_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=TRANSITION_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }


def participant_f1(
    y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray
) -> dict[str, float]:
    values: list[float] = []
    for subject in sorted(np.unique(subjects)):
        mask = subjects == subject
        present = np.unique(y_true[mask])
        values.append(
            float(
                f1_score(
                    y_true[mask],
                    y_pred[mask],
                    labels=present,
                    average="macro",
                    zero_division=0,
                )
            )
        )
    return {
        "participant_macro_f1_mean": float(np.mean(values)),
        "participant_macro_f1_std": float(np.std(values, ddof=1)),
        "participant_macro_f1_worst": float(np.min(values)),
    }


def participant_metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
    **identifiers: Any,
) -> list[dict[str, Any]]:
    """Return one comparable metric row per held-out participant."""
    rows: list[dict[str, Any]] = []
    for subject in sorted(np.unique(subjects)):
        mask = subjects == subject
        present = np.unique(y_true[mask])
        present_daily = np.intersect1d(present, DAILY_LABELS)
        present_transition = np.intersect1d(present, TRANSITION_LABELS)

        def score(labels: np.ndarray) -> float:
            if len(labels) == 0:
                return float("nan")
            return float(
                f1_score(
                    y_true[mask],
                    y_pred[mask],
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            )

        rows.append(
            {
                **identifiers,
                "participant_id": int(subject),
                "windows": int(np.sum(mask)),
                "participant_macro_f1": score(present),
                "participant_daily_macro_f1": score(present_daily),
                "participant_transition_macro_f1": score(present_transition),
            }
        )
    return rows


def macro_scorer() -> Any:
    return make_scorer(
        f1_score,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )


def make_models(
    quick: bool, cache_dir: Path
) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    linear_grid = [0.01] if quick else [0.001, 0.01, 0.1]
    rbf_c = [1.0] if quick else [1.0, 10.0]
    rbf_gamma: list[Any] = ["scale"] if quick else ["scale", 0.01]
    pca_components = [60] if quick else [60, 120]
    mi_k = [60] if quick else [20, 40, 60, 100, 200]
    mi_c = [0.01] if quick else [0.01, 0.1]
    mi_func = partial(mutual_info_classif, random_state=MI_SEED)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pipeline_memory = Memory(location=cache_dir / "pipeline", verbose=0)
    return {
        "majority": (
            Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))]),
            {},
        ),
        "linear_svm": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LinearSVC(
                            class_weight="balanced",
                            dual=False,
                            max_iter=3000,
                            tol=1e-3,
                            random_state=OUTER_SEED,
                        ),
                    ),
                ]
            ),
            {"classifier__C": linear_grid},
        ),
        "pca_svm": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "pca",
                        PCA(
                            svd_solver="randomized",
                            iterated_power=3,
                            random_state=OUTER_SEED,
                        ),
                    ),
                    (
                        "classifier",
                        LinearSVC(
                            class_weight="balanced",
                            dual=False,
                            max_iter=3000,
                            tol=1e-3,
                            random_state=OUTER_SEED,
                        ),
                    ),
                ],
                memory=pipeline_memory,
            ),
            {
                "pca__n_components": pca_components,
                "classifier__C": linear_grid,
            },
        ),
        "rbf_svm": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        SVC(
                            kernel="rbf",
                            class_weight="balanced",
                            cache_size=1500,
                            decision_function_shape="ovr",
                            random_state=OUTER_SEED,
                        ),
                    ),
                ]
            ),
            {"classifier__C": rbf_c, "classifier__gamma": rbf_gamma},
        ),
        "mutual_information_svm": (
            Pipeline(
                [
                    ("select", SelectKBest(score_func=mi_func)),
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LinearSVC(
                            class_weight="balanced",
                            dual=False,
                            max_iter=3000,
                            tol=1e-3,
                            random_state=OUTER_SEED,
                        ),
                    ),
                ],
                memory=pipeline_memory,
            ),
            {"select__k": mi_k, "classifier__C": mi_c},
        ),
    }


def nested_grouped_cv(
    dataset: Dataset,
    models: dict[str, tuple[Pipeline, dict[str, list[Any]]]],
    n_jobs: int,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    splits = list(outer.split(dataset.x_train, dataset.y_train, dataset.subject_train))
    fold_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    for model_name, (base_pipeline, grid) in models.items():
        print(f"Nested grouped CV: {model_name}")
        for fold, (train_idx, validation_idx) in enumerate(splits, start=1):
            x_fit, y_fit = dataset.x_train[train_idx], dataset.y_train[train_idx]
            groups_fit = dataset.subject_train[train_idx]
            x_val, y_val = dataset.x_train[validation_idx], dataset.y_train[validation_idx]
            val_subjects = dataset.subject_train[validation_idx]
            estimator = clone(base_pipeline)
            started = time.perf_counter()
            if grid:
                inner = StratifiedGroupKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=INNER_SEED_BASE + fold,
                )
                search = GridSearchCV(
                    estimator,
                    grid,
                    scoring=macro_scorer(),
                    cv=inner,
                    n_jobs=n_jobs,
                    refit=True,
                    return_train_score=False,
                    error_score="raise",
                )
                search.fit(x_fit, y_fit, groups=groups_fit)
                fitted = search.best_estimator_
                best_params = search.best_params_
                tuning_rows.append(
                    {
                        "model": model_name,
                        "outer_fold": fold,
                        "inner_best_macro_f1": float(search.best_score_),
                        "best_params": json.dumps(best_params, sort_keys=True),
                    }
                )
            else:
                fitted = estimator.fit(x_fit, y_fit)
                best_params = {}
            predictions = fitted.predict(x_val)
            elapsed = time.perf_counter() - started
            row: dict[str, Any] = {
                "model": model_name,
                "outer_fold": fold,
                "validation_subjects": ",".join(map(str, sorted(np.unique(val_subjects)))),
                "n_train_windows": int(len(train_idx)),
                "n_validation_windows": int(len(validation_idx)),
                "fit_and_tune_seconds": elapsed,
                "best_params": json.dumps(best_params, sort_keys=True),
            }
            row.update(metric_bundle(y_val, predictions))
            row.update(participant_f1(y_val, predictions, val_subjects))
            fold_rows.append(row)
            participant_rows.extend(
                participant_metric_rows(
                    y_val,
                    predictions,
                    val_subjects,
                    model=model_name,
                    outer_fold=fold,
                )
            )
    folds = pd.DataFrame(fold_rows)
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "daily_macro_f1",
        "transition_macro_f1",
        "participant_macro_f1_mean",
        "participant_macro_f1_worst",
    ]
    summary_rows: list[dict[str, Any]] = []
    for model_name, group in folds.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model_name, "folds": int(len(group))}
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        row["fit_and_tune_seconds_total"] = float(group["fit_and_tune_seconds"].sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    folds.to_csv(results_dir / "nested_grouped_cv_folds.csv", index=False)
    summary.to_csv(results_dir / "nested_grouped_cv_summary.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(results_dir / "nested_grouped_cv_tuning.csv", index=False)
    pd.DataFrame(participant_rows).to_csv(
        results_dir / "nested_grouped_cv_participants.csv", index=False
    )
    return folds, summary


def tune_full_development(
    dataset: Dataset,
    models: dict[str, tuple[Pipeline, dict[str, list[Any]]]],
    n_jobs: int,
    results_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    grouped_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    for model_name, (pipeline, grid) in models.items():
        print(f"Full-development grouped tuning: {model_name}")
        started = time.perf_counter()
        if grid:
            search = GridSearchCV(
                clone(pipeline),
                grid,
                scoring=macro_scorer(),
                cv=grouped_cv,
                n_jobs=n_jobs,
                refit=True,
                return_train_score=False,
                error_score="raise",
            )
            search.fit(dataset.x_train, dataset.y_train, groups=dataset.subject_train)
            fitted = search.best_estimator_
            params = search.best_params_
            development_score = float(search.best_score_)
        else:
            fitted = clone(pipeline).fit(dataset.x_train, dataset.y_train)
            params = {}
            development_score = float("nan")
        elapsed = time.perf_counter() - started
        row: dict[str, Any] = {
            "model": model_name,
            "development_grouped_cv_best_macro_f1": development_score,
            "best_params": json.dumps(params, sort_keys=True),
            "fit_tune_seconds": elapsed,
        }
        rows.append(row)
        fitted_models[model_name] = fitted
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "full_development_tuning.csv", index=False)
    return table, fitted_models


def evaluate_official_test(
    dataset: Dataset,
    fitted_models: dict[str, Pipeline],
    results_dir: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Later-stage descriptive evaluation after Pareto representatives are frozen."""
    if dataset.x_test is None or dataset.y_test is None or dataset.subject_test is None:
        raise RuntimeError("Official test data were not loaded")
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for model_name, fitted in fitted_models.items():
        prediction = fitted.predict(dataset.x_test)
        row: dict[str, Any] = {"model": model_name}
        row.update(metric_bundle(dataset.y_test, prediction))
        row.update(participant_f1(dataset.y_test, prediction, dataset.subject_test))
        rows.append(row)
        predictions[model_name] = prediction
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "official_test_results.csv", index=False)
    return table, predictions


def evaluate_fixed_cv(
    estimator: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    split_mode: str,
) -> pd.DataFrame:
    if split_mode == "participant_grouped":
        splitter: Iterable[tuple[np.ndarray, np.ndarray]] = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=OUTER_SEED
        ).split(x, y, groups)
    elif split_mode == "random_window":
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED).split(x, y)
    else:
        raise ValueError(split_mode)
    rows: list[dict[str, Any]] = []
    for fold, (train_idx, validation_idx) in enumerate(splitter, start=1):
        model = clone(estimator).fit(x[train_idx], y[train_idx])
        prediction = model.predict(x[validation_idx])
        row: dict[str, Any] = {"split": split_mode, "fold": fold}
        row.update(metric_bundle(y[validation_idx], prediction))
        rows.append(row)
    return pd.DataFrame(rows)


def split_diagnostic(
    dataset: Dataset,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed_linear = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LinearSVC(
                    C=DIAGNOSTIC_LINEAR_C,
                    class_weight="balanced",
                    dual=False,
                    max_iter=3000,
                    tol=1e-3,
                    random_state=OUTER_SEED,
                ),
            ),
        ]
    )
    grouped = evaluate_fixed_cv(
        fixed_linear,
        dataset.x_train,
        dataset.y_train,
        dataset.subject_train,
        "participant_grouped",
    )
    random = evaluate_fixed_cv(
        fixed_linear,
        dataset.x_train,
        dataset.y_train,
        dataset.subject_train,
        "random_window",
    )
    folds = pd.concat([grouped, random], ignore_index=True)
    summary = (
        folds.groupby("split")[["accuracy", "macro_f1", "daily_macro_f1", "transition_macro_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "split" if a == "split" else f"{a}_{b}"
        for a, b in summary.columns.to_flat_index()
    ]
    folds.to_csv(results_dir / "random_vs_grouped_folds.csv", index=False)
    summary.to_csv(results_dir / "random_vs_grouped_summary.csv", index=False)
    return folds, summary


def feature_domain(name: str) -> str:
    if name.startswith("f"):
        return "frequency"
    if name.startswith("t"):
        return "time"
    return "angle_or_other"


def feature_sensor(name: str) -> str:
    if "Gyro" in name:
        return "gyroscope"
    if "Acc" in name:
        return "accelerometer"
    return "mixed_or_other"


def feature_count_sweep(
    dataset: Dataset,
    n_jobs: int,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate MI-ranked feature counts with participant-nested C selection.

    MI rankings are computed once per inner training partition and reused across
    candidate feature counts and C values. This is equivalent to fitting the
    selector inside every training fold, but avoids recomputing the same MI
    scores for each SVM hyperparameter.
    """
    _ = n_jobs  # The cached manual loop is faster than repeated parallel MI fits.
    k_values = [20, 40, 60, 100, 200, 561]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    selections: list[set[int]] = []
    feature_counts = np.zeros(dataset.x_train.shape[1], dtype=int)
    for fold, (train_idx, validation_idx) in enumerate(
        outer.split(dataset.x_train, dataset.y_train, dataset.subject_train), start=1
    ):
        x_fit = dataset.x_train[train_idx]
        y_fit = dataset.y_train[train_idx]
        groups_fit = dataset.subject_train[train_idx]
        inner = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=INNER_SEED_BASE + fold,
        )
        inner_scores: dict[tuple[int, float], list[float]] = {
            (k, c): [] for k in k_values for c in FEATURE_SWEEP_C_GRID
        }
        for inner_fold, (inner_train_idx, inner_validation_idx) in enumerate(
            inner.split(x_fit, y_fit, groups_fit), start=1
        ):
            scores = mutual_info_classif(
                x_fit[inner_train_idx],
                y_fit[inner_train_idx],
                random_state=MI_SEED + 100 * fold + inner_fold,
            )
            ranking = np.argsort(scores)[::-1]
            for k in k_values:
                selected = (
                    np.arange(x_fit.shape[1], dtype=int)
                    if k == x_fit.shape[1]
                    else ranking[:k]
                )
                for c_value in FEATURE_SWEEP_C_GRID:
                    model = Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "classifier",
                                LinearSVC(
                                    C=c_value,
                                    class_weight="balanced",
                                    dual=False,
                                    max_iter=3000,
                                    tol=1e-3,
                                    random_state=OUTER_SEED,
                                ),
                            ),
                        ]
                    )
                    model.fit(x_fit[inner_train_idx][:, selected], y_fit[inner_train_idx])
                    prediction = model.predict(x_fit[inner_validation_idx][:, selected])
                    inner_scores[(k, c_value)].append(
                        metric_bundle(y_fit[inner_validation_idx], prediction)["macro_f1"]
                    )

        best_c_by_k: dict[int, float] = {}
        best_inner_by_k: dict[int, float] = {}
        for k in k_values:
            mean_by_c = {
                c_value: float(np.mean(inner_scores[(k, c_value)]))
                for c_value in FEATURE_SWEEP_C_GRID
            }
            best_c = max(
                FEATURE_SWEEP_C_GRID,
                key=lambda c_value: (mean_by_c[c_value], -c_value),
            )
            best_c_by_k[k] = best_c
            best_inner_by_k[k] = mean_by_c[best_c]

        outer_scores = mutual_info_classif(
            x_fit,
            y_fit,
            random_state=MI_SEED + fold,
        )
        outer_ranking = np.argsort(outer_scores)[::-1]
        top60 = set(map(int, outer_ranking[:60]))
        selections.append(top60)
        feature_counts[list(top60)] += 1
        for k in k_values:
            selected = (
                np.arange(x_fit.shape[1], dtype=int)
                if k == x_fit.shape[1]
                else outer_ranking[:k]
            )
            c_value = best_c_by_k[k]
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LinearSVC(
                            C=c_value,
                            class_weight="balanced",
                            dual=False,
                            max_iter=3000,
                            tol=1e-3,
                            random_state=OUTER_SEED,
                        ),
                    ),
                ]
            )
            model.fit(x_fit[:, selected], y_fit)
            prediction = model.predict(dataset.x_train[validation_idx][:, selected])
            row: dict[str, Any] = {
                "fold": fold,
                "feature_count": k,
                "linear_svm_C": c_value,
                "inner_best_macro_f1": best_inner_by_k[k],
            }
            row.update(metric_bundle(dataset.y_train[validation_idx], prediction))
            rows.append(row)
            participant_rows.extend(
                participant_metric_rows(
                    dataset.y_train[validation_idx],
                    prediction,
                    dataset.subject_train[validation_idx],
                    method="mutual_information",
                    outer_fold=fold,
                    feature_count=k,
                )
            )
    fold_table = pd.DataFrame(rows)
    summary = (
        fold_table.groupby("feature_count")[["macro_f1", "daily_macro_f1", "transition_macro_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "feature_count" if a == "feature_count" else f"{a}_{b}"
        for a, b in summary.columns.to_flat_index()
    ]
    jaccards: list[float] = []
    for i in range(len(selections)):
        for j in range(i + 1, len(selections)):
            jaccards.append(len(selections[i] & selections[j]) / len(selections[i] | selections[j]))
    stability_rows: list[dict[str, Any]] = []
    for index, count in enumerate(feature_counts):
        if count:
            name = dataset.feature_names[index]
            stability_rows.append(
                {
                    "feature_index_1_based": index + 1,
                    "feature_name": name,
                    "selected_in_folds": int(count),
                    "selection_frequency": float(count / 5),
                    "domain": feature_domain(name),
                    "sensor": feature_sensor(name),
                }
            )
    stability = pd.DataFrame(stability_rows).sort_values(
        ["selected_in_folds", "feature_index_1_based"], ascending=[False, True]
    )
    participant_table = pd.DataFrame(participant_rows)
    full = participant_table[participant_table["feature_count"] == 561][
        [
            "participant_id",
            "participant_macro_f1",
            "participant_daily_macro_f1",
            "participant_transition_macro_f1",
        ]
    ].rename(
        columns={
            "participant_macro_f1": "full_participant_macro_f1",
            "participant_daily_macro_f1": "full_participant_daily_macro_f1",
            "participant_transition_macro_f1": "full_participant_transition_macro_f1",
        }
    )
    participant_table = participant_table.merge(full, on="participant_id", how="left")
    participant_table["overall_loss_vs_full"] = (
        participant_table["full_participant_macro_f1"]
        - participant_table["participant_macro_f1"]
    )
    participant_table["daily_loss_vs_full"] = (
        participant_table["full_participant_daily_macro_f1"]
        - participant_table["participant_daily_macro_f1"]
    )
    participant_table["transition_loss_vs_full"] = (
        participant_table["full_participant_transition_macro_f1"]
        - participant_table["participant_transition_macro_f1"]
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    uncertainty_rows: list[dict[str, Any]] = []
    for k, group in participant_table.groupby("feature_count", sort=True):
        losses = group["overall_loss_vs_full"].dropna().to_numpy(dtype=float)
        sampled = rng.choice(
            losses,
            size=(BOOTSTRAP_REPLICATES, len(losses)),
            replace=True,
        ).mean(axis=1)
        uncertainty_rows.append(
            {
                "feature_count": int(k),
                "participants": int(len(losses)),
                "mean_overall_loss": float(np.mean(losses)),
                "sample_std_overall_loss": float(np.std(losses, ddof=1)),
                "bootstrap_mean_loss_lower_95": float(np.quantile(sampled, 0.025)),
                "bootstrap_mean_loss_upper_95": float(np.quantile(sampled, 0.975)),
                "bootstrap_mean_loss_upper_one_sided_95": float(np.quantile(sampled, 0.95)),
                "median_overall_loss": float(np.median(losses)),
                "overall_loss_q1": float(np.quantile(losses, 0.25)),
                "overall_loss_q3": float(np.quantile(losses, 0.75)),
                "overall_loss_p95": float(np.quantile(losses, 0.95)),
                "worst_participant_loss": float(np.max(losses)),
                "noninferior_at_0_03": bool(
                    np.quantile(sampled, 0.95) <= PRIMARY_NONINFERIORITY_MARGIN
                ),
                "noninferior_at_0_02_sensitivity": bool(
                    np.quantile(sampled, 0.95) <= SENSITIVITY_MARGIN
                ),
            }
        )
    uncertainty = pd.DataFrame(uncertainty_rows)
    fold_table.to_csv(results_dir / "feature_count_sweep_folds.csv", index=False)
    summary.to_csv(results_dir / "feature_count_sweep_summary.csv", index=False)
    stability.to_csv(results_dir / "top60_feature_stability.csv", index=False)
    participant_table.to_csv(
        results_dir / "feature_count_participant_scores.csv", index=False
    )
    uncertainty.to_csv(
        results_dir / "feature_count_participant_uncertainty.csv", index=False
    )
    (results_dir / "feature_stability_summary.json").write_text(
        json.dumps(
            {
                "mean_pairwise_jaccard_top60": float(np.mean(jaccards)),
                "std_pairwise_jaccard_top60": float(np.std(jaccards, ddof=1)),
                "features_selected_in_all_five_folds": int(np.sum(feature_counts == 5)),
                "features_selected_in_at_least_four_folds": int(np.sum(feature_counts >= 4)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return fold_table, summary, stability, participant_table, uncertainty


def classwise_table(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    results_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, y_pred in predictions.items():
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, zero_division=0
        )
        for idx, label in enumerate(LABELS):
            rows.append(
                {
                    "model": model_name,
                    "label": int(label),
                    "activity": ACTIVITY_NAMES[int(label)],
                    "precision": float(precision[idx]),
                    "recall": float(recall[idx]),
                    "f1": float(f1[idx]),
                    "support": int(support[idx]),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "official_test_classwise_metrics.csv", index=False)
    return table


def save_plots(
    cv_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    sweep_summary: pd.DataFrame,
    participant_scores: pd.DataFrame,
    figures_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    # Compact vector overview used in the method section of the paper.
    fig, ax = plt.subplots(figsize=(12.0, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    nodes = [
        (0.25, "HAPT windows\n561 inertial features", "#E9F5F7"),
        (2.65, "Outer split\nheld-out participants", "#EAF0F8"),
        (5.05, "Inner grouped search\nNSGA-II + linear SVM", "#FFF1E8"),
        (7.45, "Freeze 3 representatives\nsmallest, knee, transition", "#F3EEFA"),
        (9.85, "Outer evaluation\nF1 + participant loss", "#EAF6EE"),
    ]
    for x_pos, label, fill in nodes:
        box = FancyBboxPatch(
            (x_pos, 0.72),
            1.9,
            1.12,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.3,
            edgecolor="#334155",
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(
            x_pos + 0.95,
            1.28,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#172033",
            linespacing=1.25,
        )
    for x_pos in [2.18, 4.58, 6.98, 9.38]:
        ax.add_patch(
            FancyArrowPatch(
                (x_pos, 1.28),
                (x_pos + 0.42, 1.28),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.5,
                color="#516174",
            )
        )
    ax.text(
        6.0,
        0.28,
        "Scaling, tuning, and feature selection are fitted only on training participants; no official-test result is used in Stage 3.",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#475569",
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(figures_dir / "proposed_method_pipeline.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "proposed_method_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)

    display_names = {
        "majority": "Majority",
        "linear_svm": "Linear SVM",
        "pca_svm": "PCA + Linear SVM",
        "rbf_svm": "RBF SVM",
        "mutual_information_svm": "MI + Linear SVM",
    }
    plot_rows: list[dict[str, Any]] = []
    for _, row in cv_summary.iterrows():
        for metric, label in [("macro_f1", "Overall macro-F1"), ("transition_macro_f1", "Transition macro-F1")]:
            plot_rows.append(
                {
                    "model": display_names.get(row["model"], row["model"]),
                    "metric": label,
                    "mean": row[f"{metric}_mean"],
                    "std": row[f"{metric}_std"],
                }
            )
    plot_frame = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    models = list(dict.fromkeys(plot_frame["model"]))
    x = np.arange(len(models))
    width = 0.36
    colors = ["#16A6B6", "#F08A5D"]
    for j, metric in enumerate(plot_frame["metric"].unique()):
        sub = plot_frame[plot_frame["metric"] == metric].set_index("model").loc[models]
        ax.bar(x + (j - 0.5) * width, sub["mean"], width, yerr=sub["std"], capsize=4, label=metric, color=colors[j])
    ax.set_xticks(x, models, rotation=12, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score")
    ax.set_title("Participant-grouped baseline performance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_grouped_cv.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "baseline_grouped_cv.pdf", bbox_inches="tight")
    plt.close(fig)

    split_long: list[dict[str, Any]] = []
    for _, row in split_summary.iterrows():
        for metric, label in [("macro_f1", "Overall macro-F1"), ("transition_macro_f1", "Transition macro-F1")]:
            split_long.append(
                {
                    "split": "Grouped by participant" if row["split"] == "participant_grouped" else "Random windows",
                    "metric": label,
                    "mean": row[f"{metric}_mean"],
                    "std": row[f"{metric}_std"],
                }
            )
    split_frame = pd.DataFrame(split_long)
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    splits = list(dict.fromkeys(split_frame["split"]))
    x = np.arange(len(splits))
    for j, metric in enumerate(split_frame["metric"].unique()):
        sub = split_frame[split_frame["metric"] == metric].set_index("split").loc[splits]
        ax.bar(x + (j - 0.5) * width, sub["mean"], width, yerr=sub["std"], capsize=4, label=metric, color=colors[j])
    ax.set_xticks(x, splits)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score")
    ax.set_title("Validation split changes the estimated score")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "random_vs_grouped.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "random_vs_grouped.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.errorbar(
        sweep_summary["feature_count"],
        sweep_summary["macro_f1_mean"],
        yerr=sweep_summary["macro_f1_std"],
        marker="o",
        linewidth=2.4,
        capsize=4,
        label="Overall macro-F1",
        color="#16A6B6",
    )
    ax.errorbar(
        sweep_summary["feature_count"],
        sweep_summary["transition_macro_f1_mean"],
        yerr=sweep_summary["transition_macro_f1_std"],
        marker="s",
        linewidth=2.4,
        capsize=4,
        label="Transition macro-F1",
        color="#F08A5D",
    )
    ax.set_xscale("log")
    ax.set_xticks([20, 40, 60, 100, 200, 561], [20, 40, 60, 100, 200, 561])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of retained features")
    ax.set_ylabel("F1 score")
    ax.set_title("Mutual-information feature-count sweep")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_count_tradeoff.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "feature_count_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)

    at_60 = participant_scores[participant_scores["feature_count"] == 60].sort_values(
        "overall_loss_vs_full"
    )
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    y_positions = np.arange(len(at_60))
    colors = np.where(
        at_60["overall_loss_vs_full"] <= PRIMARY_NONINFERIORITY_MARGIN,
        "#16A6B6",
        "#F08A5D",
    )
    ax.scatter(at_60["overall_loss_vs_full"], y_positions, s=62, c=colors, zorder=3)
    ax.axvline(
        PRIMARY_NONINFERIORITY_MARGIN,
        color="#1F3557",
        linestyle="-",
        linewidth=2,
        label="Stage 2 H1 margin (0.03)",
    )
    ax.axvline(
        SENSITIVITY_MARGIN,
        color="#718096",
        linestyle="--",
        linewidth=2,
        label="Sensitivity margin (0.02)",
    )
    ax.axvline(0, color="#718096", linewidth=1)
    ax.set_yticks(y_positions, [f"P{int(v):02d}" for v in at_60["participant_id"]])
    ax.set_xlabel("Participant macro-F1 loss: 561 features minus 60 features")
    ax.set_ylabel("Held-out participant")
    ax.set_title("A mean can hide large participant-level losses")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "participant_loss_at_60.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "participant_loss_at_60.pdf", bbox_inches="tight")
    plt.close(fig)


def save_official_test_plot(
    y_test: np.ndarray,
    linear_prediction: np.ndarray,
    figures_dir: Path,
) -> None:
    """Final-stage-only plot; never called in the default Stage 3 run."""
    cm = confusion_matrix(y_test, linear_prediction, labels=LABELS, normalize="true")
    pd.DataFrame(
        cm,
        index=[ACTIVITY_NAMES[int(label)] for label in LABELS],
        columns=[ACTIVITY_NAMES[int(label)] for label in LABELS],
    ).to_csv(figures_dir / "linear_svm_test_confusion_matrix_normalized.csv")
    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    sns.heatmap(
        cm,
        cmap=sns.light_palette("#16A6B6", as_cmap=True),
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Recall within true class"},
        ax=ax,
    )
    ax.set_xticklabels([ACTIVITY_NAMES[int(label)].replace("_", " ") for label in LABELS], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([ACTIVITY_NAMES[int(label)].replace("_", " ") for label in LABELS], rotation=0, fontsize=9)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Descriptive evaluation on official test participants")
    fig.tight_layout()
    fig.savefig(figures_dir / "linear_svm_test_confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_environment(results_dir: Path, runtime_seconds: float) -> None:
    memory_gb = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        memory_gb = page_size * pages / (1024**3)
    except (ValueError, OSError, AttributeError):
        pass
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "not reported by runtime",
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_gb": None if memory_gb is None else round(memory_gb, 1),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "total_runtime_seconds": runtime_seconds,
        "gpu_used": False,
        "outer_seed": OUTER_SEED,
        "mutual_information_seed": MI_SEED,
    }
    (results_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.unlock_official_test and args.test_lock_token != OFFICIAL_TEST_LOCK_TOKEN:
        raise SystemExit(
            "Official-test evaluation is disabled for this run. Freeze the Pareto "
            "representatives first, then re-run with --unlock-official-test "
            f"--test-lock-token {OFFICIAL_TEST_LOCK_TOKEN}. Label any resulting HAPT "
            "test score descriptive because earlier debugging accessed that partition."
        )
    started = time.perf_counter()
    ensure_dataset(args.data_dir, args.download, include_test=args.unlock_official_test)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data_dir, include_test=args.unlock_official_test)
    models = make_models(args.quick, args.cache_dir)
    folds, cv_summary = nested_grouped_cv(dataset, models, args.n_jobs, args.results_dir)
    development_tuning, fitted_models = tune_full_development(
        dataset, models, args.n_jobs, args.results_dir
    )
    split_folds, split_summary = split_diagnostic(dataset, args.results_dir)
    sweep_folds, sweep_summary, stability, participant_scores, uncertainty = feature_count_sweep(
        dataset, args.n_jobs, args.results_dir
    )
    save_plots(
        cv_summary,
        split_summary,
        sweep_summary,
        participant_scores,
        args.figures_dir,
    )
    test_table: pd.DataFrame | None = None
    if args.unlock_official_test:
        test_table, test_predictions = evaluate_official_test(
            dataset, fitted_models, args.results_dir
        )
        assert dataset.y_test is not None
        classwise_table(dataset.y_test, test_predictions, args.results_dir)
        save_official_test_plot(
            dataset.y_test,
            test_predictions["linear_svm"],
            args.figures_dir,
        )
    runtime_seconds = time.perf_counter() - started
    write_environment(args.results_dir, runtime_seconds)
    summary = {
        "dataset": {
            "training_windows": int(dataset.x_train.shape[0]),
            "features": int(dataset.x_train.shape[1]),
            "training_participants": int(len(np.unique(dataset.subject_train))),
            "classes": 12,
            "archive_sha256": DATASET_SHA256,
        },
        "development_cv": cv_summary.to_dict(orient="records"),
        "full_development_tuning": development_tuning.to_dict(orient="records"),
        "random_vs_grouped": split_summary.to_dict(orient="records"),
        "feature_count_sweep": sweep_summary.to_dict(orient="records"),
        "participant_uncertainty": uncertainty.to_dict(orient="records"),
        "official_test_unlocked": bool(args.unlock_official_test),
        "runtime_seconds": runtime_seconds,
        "quick_mode": bool(args.quick),
    }
    if test_table is not None:
        summary["official_test"] = test_table.to_dict(orient="records")
    (args.results_dir / "stage3_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(f"Completed in {runtime_seconds / 60:.1f} minutes")
    print(cv_summary.to_string(index=False))
    print(uncertainty.to_string(index=False))
    if test_table is not None:
        print(test_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
