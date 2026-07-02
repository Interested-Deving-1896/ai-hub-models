# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.grootn15 import MODEL_ID, Model
from qai_hub_models.models.grootn15.app import (
    build_app,
    get_default_dataset_path,
)
from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.data.dataset import (
    LeRobotSingleDataset,
)
from qai_hub_models.models.grootn15.model import load_checkpoint
from qai_hub_models.utils.args import (
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.evaluate.helpers import EvalMode


def _plot_action_chunks(
    pred_concat: np.ndarray,
    gt_concat: np.ndarray,
    modality_keys: list[str],
    action_dims: list[int],
    output_dir: Path,
    eval_mode: EvalMode,
) -> None:
    """Plot predicted vs ground-truth action chunks side-by-side per DoF."""
    total_dof = sum(action_dims)
    time_steps = np.arange(pred_concat.shape[0])  # action_horizon

    # Build a flat label per DOF: e.g. "arm [0]", "arm [1]", "gripper [0]", ...
    dof_labels: list[str] = []
    dof_labels.extend(
        f"{key} [{d}]"
        for key, dim in zip(modality_keys, action_dims, strict=False)
        for d in range(dim)
    )

    n_cols = 2
    n_rows = (total_dof + n_cols - 1) // n_cols
    mode_label = eval_mode.value.upper()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), squeeze=False)
    fig.suptitle(
        f"Predicted vs Ground Truth Action Chunk  [{mode_label}]",
        fontsize=13,
        fontweight="bold",
    )

    for dof_idx in range(total_dof):
        row, col = divmod(dof_idx, n_cols)
        ax = axes[row][col]
        ax.plot(
            time_steps,
            gt_concat[:, dof_idx],
            label="Ground Truth",
            color="steelblue",
            linewidth=1.5,
        )
        ax.plot(
            time_steps,
            pred_concat[:, dof_idx],
            label="Predicted",
            color="darkorange",
            linewidth=1.5,
            linestyle="--",
        )
        ax.set_title(f"DoF {dof_idx}  —  {dof_labels[dof_idx]}", fontsize=9)
        ax.set_xlabel("Action Horizon (t)")
        ax.set_ylabel("Value")
        ax.legend(fontsize=7)

    # Hide unused subplots (when total_dof is odd)
    for dof_idx in range(total_dof, n_rows * n_cols):
        row, col = divmod(dof_idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plot_path = output_dir / f"{MODEL_ID}_demo_action_chunk_plot_{mode_label}.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Action chunk plot saved at {plot_path}")


# Entry point
def main() -> None:
    # Argument parsing
    parser = get_model_cli_parser(Model)
    parser = get_on_device_demo_parser(
        parser,
        add_output_dir=True,
        supported_precisions={Precision.float},
        available_target_runtimes=[TargetRuntime.QNN_CONTEXT_BINARY],
        default_device="Samsung Galaxy S25 (Family)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=get_default_dataset_path(),
        help="Path to LeRobot-format dataset directory.",
    )

    args = parser.parse_args()
    validate_on_device_demo_args(args, MODEL_ID)

    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    policy = load_checkpoint(
        checkpoint=args.checkpoint,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        device=args.host_device,
    )

    # Build GrootApp
    app = build_app(
        policy, args.eval_mode, args.device, args.hub_model_id, args.host_device
    )

    # Load dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=policy.modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )
    step_data = dataset.get_step_data(trajectory_id=0, base_index=0)

    # Run inference
    pred_actions = app.predict_action_chunk(step_data)

    # Evaluate predictions
    modality_keys = [
        key.split(".")[-1] for key in policy.modality_config["action"].modality_keys
    ]

    pred_actions_squeezed = {k: np.asarray(v)[0] for k, v in pred_actions.items()}
    pred_action_across_time: list[np.ndarray] = []
    for j in range(app.config.action_horizon):
        concat_pred = np.concatenate(
            [pred_actions_squeezed[f"action.{key}"][j] for key in modality_keys],
            axis=0,
        )
        pred_action_across_time.append(concat_pred)
    pred_concat = np.stack(pred_action_across_time)

    # Ground-truth actions for comparison.
    gt_action_across_time: list[np.ndarray] = []
    for step in range(app.config.action_horizon):
        data_point_step = dataset.get_step_data(trajectory_id=0, base_index=step)
        concat_gt = np.concatenate(
            [data_point_step[f"action.{key}"][0] for key in modality_keys], axis=0
        )
        gt_action_across_time.append(concat_gt)
    gt_concat = np.array(gt_action_across_time)  # (16, total_dof)

    action_dims = [
        pred_actions_squeezed[f"action.{key}"].shape[-1]  # (H, dof) -> dof
        for key in modality_keys
    ]

    rmse = np.sqrt(np.mean((gt_concat - pred_concat) ** 2))
    print(
        f"Action RMSE over {app.config.action_horizon}-step horizon [{args.eval_mode.value.upper()}]: {rmse:.6f}"
    )

    _plot_action_chunks(
        pred_concat,
        gt_concat,
        modality_keys,
        action_dims,
        output_dir,
        args.eval_mode,
    )


if __name__ == "__main__":
    main()
