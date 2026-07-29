#!/usr/bin/env python3
"""Generate post-hoc action and safety diagnostics from Shadow Replay artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


VIOLATION_TYPES = (
    "joint_limit",
    "max_joint_step",
    "max_joint_velocity",
    "max_joint_acceleration",
)


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_violations(path: Path, episode_id: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record.get("episode_id", -1)) == episode_id:
                records.append(record)
    return records


def _rank_dimensions(
    raw_mae: np.ndarray, safe_mae: np.ndarray, count: int
) -> np.ndarray:
    severity = np.maximum(raw_mae, safe_mae)
    return np.argsort(severity)[::-1][: min(count, len(severity))]


def _save_action_comparison(
    path: Path,
    *,
    gt: np.ndarray,
    raw: np.ndarray,
    safe: np.ndarray,
    joint_names: list[str],
    dimensions: np.ndarray,
) -> None:
    plt = _pyplot()
    figure, axes = plt.subplots(
        len(dimensions),
        1,
        figsize=(13, max(7, 2.25 * len(dimensions))),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    steps = np.arange(len(gt))
    for axis, dimension in zip(axes, dimensions):
        axis.plot(steps, gt[:, dimension], color="black", linewidth=1.8, label="GT")
        axis.plot(
            steps,
            raw[:, dimension],
            color="tab:orange",
            linewidth=1.4,
            linestyle="--",
            label="raw joint target",
        )
        axis.plot(
            steps,
            safe[:, dimension],
            color="tab:blue",
            linewidth=1.4,
            linestyle=":",
            label="safe output",
        )
        raw_mae = float(np.mean(np.abs(raw[:, dimension] - gt[:, dimension])))
        safe_mae = float(np.mean(np.abs(safe[:, dimension] - gt[:, dimension])))
        axis.set_title(
            f"dim {dimension}: {joint_names[dimension]}  "
            f"(raw MAE={raw_mae:.3f}, safe MAE={safe_mae:.3f} rad)",
            loc="left",
            fontsize=10,
        )
        axis.set_ylabel("rad")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=3, loc="best")
    axes[-1].set_xlabel("replay step (chunk horizon 0)")
    figure.suptitle("GT vs raw prediction vs safety-filtered output", fontsize=15)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _violation_matrix(
    records: list[dict[str, Any]], step_count: int, *, first_horizon_only: bool
) -> np.ndarray:
    matrix = np.zeros((len(VIOLATION_TYPES), step_count), dtype=np.int8)
    type_to_row = {name: index for index, name in enumerate(VIOLATION_TYPES)}
    for record in records:
        horizon = int(record.get("horizon", -1))
        if first_horizon_only and horizon != 0:
            continue
        violation_type = str(record.get("violation_type", ""))
        step = int(record.get("step_id", -1))
        if violation_type in type_to_row and 0 <= step < step_count:
            matrix[type_to_row[violation_type], step] = 1
    return matrix


def _save_safety_breakdown(
    path: Path, *, records: list[dict[str, Any]], step_count: int
) -> None:
    plt = _pyplot()
    first = _violation_matrix(records, step_count, first_horizon_only=True)
    any_horizon = _violation_matrix(records, step_count, first_horizon_only=False)
    counts = Counter(str(record.get("violation_type", "")) for record in records)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1, 1, 1.2)},
    )
    for axis, matrix, title in (
        (axes[0], first, "Violations at executed candidate (horizon 0)"),
        (axes[1], any_horizon, "Violations anywhere in the 16-step chunk"),
    ):
        axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            vmin=0,
            vmax=1,
            cmap="Reds",
        )
        axis.set_yticks(np.arange(len(VIOLATION_TYPES)))
        axis.set_yticklabels(VIOLATION_TYPES)
        axis.set_xticks(np.arange(step_count))
        axis.set_title(title)
        axis.set_ylabel("check")
    axes[1].set_xlabel("replay step")
    values = [counts[name] for name in VIOLATION_TYPES]
    axes[2].bar(VIOLATION_TYPES, values, color="tab:red", alpha=0.8)
    for index, value in enumerate(values):
        axes[2].text(index, value, str(value), ha="center", va="bottom")
    axes[2].set_ylabel("violation records")
    axes[2].set_title(
        "Record counts over all horizons (one step/horizon can create multiple records)"
    )
    axes[2].tick_params(axis="x", rotation=15)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _diagnostics(
    *,
    episode_id: int,
    gt: np.ndarray,
    raw: np.ndarray,
    safe: np.ndarray,
    safety_valid: np.ndarray,
    latency_ms: np.ndarray,
    joint_names: list[str],
    violations: list[dict[str, Any]],
    top_count: int,
) -> dict[str, Any]:
    raw_error = np.abs(raw - gt)
    safe_error = np.abs(safe - gt)
    raw_per_dimension = np.mean(raw_error, axis=0)
    safe_per_dimension = np.mean(safe_error, axis=0)
    ranked = _rank_dimensions(raw_per_dimension, safe_per_dimension, top_count)
    violation_type_counts = Counter(
        str(record.get("violation_type", "")) for record in violations
    )
    violation_joint_counts: Counter[int] = Counter()
    for record in violations:
        violation_joint_counts.update(int(index) for index in record.get("indices") or [])
    return {
        "episode_id": episode_id,
        "step_count": int(len(gt)),
        "action_dimension": int(gt.shape[1]),
        "comparison_scope": "horizon 0 at each replay step",
        "raw_action_mae_rad": float(np.mean(raw_error)),
        "safe_action_mae_rad": float(np.mean(safe_error)),
        "safe_minus_raw_mae_rad": float(np.mean(safe_error) - np.mean(raw_error)),
        "safe_is_closer_cell_ratio": float(np.mean(safe_error < raw_error)),
        "first_action_valid_count": int(np.sum(safety_valid)),
        "first_action_invalid_count": int(np.sum(~safety_valid)),
        "first_action_valid_ratio": float(np.mean(safety_valid)),
        "latency_total_mean_ms": float(np.mean(latency_ms)),
        "latency_total_p95_ms": float(np.percentile(latency_ms, 95)),
        "top_error_dimensions": [
            {
                "dimension": int(index),
                "joint_name": joint_names[index],
                "raw_mae_rad": float(raw_per_dimension[index]),
                "safe_mae_rad": float(safe_per_dimension[index]),
            }
            for index in ranked
        ],
        "violation_record_count": len(violations),
        "violation_record_counts_by_type": {
            name: int(violation_type_counts[name]) for name in VIOLATION_TYPES
        },
        "top_violation_dimensions": [
            {
                "dimension": index,
                "joint_name": joint_names[index],
                "record_appearances": count,
            }
            for index, count in violation_joint_counts.most_common(top_count)
        ],
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    top_rows = "\n".join(
        f"| {item['dimension']} | {item['joint_name']} | "
        f"{item['raw_mae_rad']:.4f} | {item['safe_mae_rad']:.4f} |"
        for item in result["top_error_dimensions"]
    )
    violation_rows = "\n".join(
        f"| {name} | {count} |"
        for name, count in result["violation_record_counts_by_type"].items()
    )
    text = f"""# Episode {result['episode_id']:04d} action diagnostics

Comparison scope: the horizon-0 prediction at each of {result['step_count']} replay steps.

## Summary

- Raw prediction MAE: `{result['raw_action_mae_rad']:.4f} rad`
- Safety-filtered MAE: `{result['safe_action_mae_rad']:.4f} rad`
- Safe minus raw MAE: `{result['safe_minus_raw_mae_rad']:+.4f} rad`
- Cells where safety output is closer to GT: `{result['safe_is_closer_cell_ratio']:.1%}`
- First-action valid ratio: `{result['first_action_valid_ratio']:.1%}`
- Total latency: mean `{result['latency_total_mean_ms']:.1f} ms`, P95 `{result['latency_total_p95_ms']:.1f} ms`

## Highest-error dimensions

| Dimension | Joint | Raw MAE [rad] | Safe MAE [rad] |
|---:|---|---:|---:|
{top_rows}

## Safety violation records over all chunk horizons

| Type | Records |
|---|---:|
{violation_rows}

Record counts are diagnostic events, not failed-action counts. One step/horizon can
produce multiple records. The safety-filtered output follows the configured clip or
reject behavior; inspect `config_resolved.yaml` before interpreting it.
"""
    path.write_text(text, encoding="utf-8")


def analyze(output_dir: Path, episode_id: int, top_count: int) -> list[Path]:
    episode_stem = f"episode_{episode_id:04d}"
    npz_path = output_dir / "episode_results" / f"{episode_stem}.npz"
    summary_path = output_dir / "dataset_summary.json"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Episode artifact does not exist: {npz_path}")
    summary = _load_json(summary_path)
    with np.load(npz_path, allow_pickle=True) as data:
        gt = np.asarray(data["gt_action"], dtype=np.float64)
        raw = np.asarray(data["pred_joint_target"][:, 0], dtype=np.float64)
        safe = np.asarray(data["pred_action_safe"][:, 0], dtype=np.float64)
        safety_valid = np.asarray(data["safety_valid"], dtype=bool)
        latency_ms = np.asarray(data["latency_total_ms"], dtype=np.float64)
    if gt.shape != raw.shape or gt.shape != safe.shape:
        raise ValueError(f"Expected matching [T,D] arrays, got {gt.shape}/{raw.shape}/{safe.shape}")
    joint_names = list(summary.get("action_names") or [])
    if len(joint_names) != gt.shape[1]:
        joint_names = [f"dimension_{index}" for index in range(gt.shape[1])]
    violations = _load_violations(output_dir / "safety_violations.jsonl", episode_id)
    raw_per_dimension = np.mean(np.abs(raw - gt), axis=0)
    safe_per_dimension = np.mean(np.abs(safe - gt), axis=0)
    dimensions = _rank_dimensions(raw_per_dimension, safe_per_dimension, top_count)

    visualization_dir = output_dir / "visualizations"
    analysis_dir = output_dir / "analysis"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    action_plot = visualization_dir / f"{episode_stem}_action_diagnostics.png"
    safety_plot = visualization_dir / f"{episode_stem}_safety_breakdown.png"
    json_path = analysis_dir / f"{episode_stem}_diagnostics.json"
    markdown_path = analysis_dir / f"{episode_stem}_diagnostics.md"
    _save_action_comparison(
        action_plot,
        gt=gt,
        raw=raw,
        safe=safe,
        joint_names=joint_names,
        dimensions=dimensions,
    )
    _save_safety_breakdown(safety_plot, records=violations, step_count=len(gt))
    result = _diagnostics(
        episode_id=episode_id,
        gt=gt,
        raw=raw,
        safe=safe,
        safety_valid=safety_valid,
        latency_ms=latency_ms,
        joint_names=joint_names,
        violations=violations,
        top_count=top_count,
    )
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(markdown_path, result)
    return [action_plot, safety_plot, json_path, markdown_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Shadow Replay output directory")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--top-count", type=int, default=6)
    args = parser.parse_args()
    for generated_path in analyze(
        args.output_dir.resolve(), args.episode_id, max(1, args.top_count)
    ):
        print(generated_path)


if __name__ == "__main__":
    main()
