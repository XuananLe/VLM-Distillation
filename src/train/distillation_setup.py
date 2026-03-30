import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

from transformers import TrainerCallback

from src.train.train_utils import rank0_print


@dataclass
class DistillationArguments:
    """Arguments for knowledge distillation."""

    student_model_id: str = field(
        metadata={"help": "Student model ID or path."}
    )

    teacher_model_ids: str = field(
        metadata={"help": "Teacher model IDs as a Python list literal or comma-separated string."}
    )

    distillation_loss: str = field(
        default="uld_loss",
        metadata={
            "help": "KD loss to use. Supported by src/components/loss.py, e.g. uld_loss, forward_kl, reverse_kl, jensen_shannon_divergence."
        },
    )

    temperature: float = field(
        default=2.0,
        metadata={"help": "Legacy shorthand temperature. Used for both student and teacher if separate temperatures are not set."}
    )

    student_temperature: float | None = field(
        default=None,
        metadata={"help": "Student softmax temperature for KD. Defaults to --temperature when omitted."}
    )

    teacher_temperature: float | None = field(
        default=None,
        metadata={"help": "Teacher softmax temperature for KD. Defaults to --temperature when omitted."}
    )

    skip_student_eos: bool = field(
        default=True,
        metadata={"help": "Drop the last supervised student token from KD, matching the paper's optional EOS skip."}
    )

    skip_teacher_eos: bool = field(
        default=True,
        metadata={"help": "Drop the last supervised teacher token from KD, matching the paper's optional EOS skip."}
    )

    alpha: float = field(
        default=1.0,
        metadata={"help": "Weight on KD in `ce_loss + alpha * kd_loss`."},
    )

    post_save_eval_root: str = field(
        default="/output/vlmeval",
        metadata={"help": "Root directory for post-save benchmark eval artifacts."},
    )


class VlmEvalOnSaveCallback(TrainerCallback):
    DEFAULT_DATASET_NAMES = (
        "ChartQA_TEST",
        "DocVQA_VAL",
        "TextVQA_VAL",
    )
    def __init__(
        self,
        *,
        root_dir: Path,
        student_model_id: str,
        eval_base_dir: str | Path,
        dataset_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.student_model_id = student_model_id
        self.eval_base_dir = Path(eval_base_dir)
        self.dataset_names = tuple(dataset_names or self.DEFAULT_DATASET_NAMES)
        self.model_class_name = self._infer_model_class_name(student_model_id)

    @staticmethod
    def _infer_model_class_name(student_model_id: str) -> str | None:
        model_id = student_model_id.lower()
        if "smolvlm2" in model_id:
            return "SmolVLM2"
        if "smolvlm" in model_id:
            return "SmolVLM"
        if "qwen3-vl" in model_id:
            return "Qwen3VLChat"
        if "qwen2-vl" in model_id or "qwen2.5-vl" in model_id:
            return "Qwen2VLChat"
        if "gemma-3" in model_id:
            return "Gemma3"
        return None

    @override
    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control

        if self.model_class_name is None:
            rank0_print(
                f"Skipping post-save eval: unsupported student model "
                f"for vlmeval auto-mapping: {self.student_model_id}"
            )
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.is_dir():
            rank0_print(f"Skipping post-save eval: checkpoint not found at {checkpoint_dir}")
            return control

        run_name = Path(args.output_dir).name
        checkpoint_eval_root = self.eval_base_dir / run_name / f"checkpoint-{state.global_step}"
        os.makedirs(checkpoint_eval_root, exist_ok=True)
        submit_log_path = checkpoint_eval_root / "submit.log"

        pending_eval_jobs = []
        for dataset_name in self.dataset_names:
            eval_root = checkpoint_eval_root / dataset_name.lower()
            done_path = eval_root / "done"
            failed_path = eval_root / "failed"
            os.makedirs(eval_root, exist_ok=True)

            if done_path.exists():
                rank0_print(f"Skipping post-save {dataset_name} eval: already completed for {checkpoint_dir}")
                continue

            failed_path.unlink(missing_ok=True)
            done_path.unlink(missing_ok=True)
            pending_eval_jobs.append(
                {
                    "dataset_name": dataset_name,
                    "eval_root": eval_root,
                    "done_path": done_path,
                    "failed_path": failed_path,
                }
            )

        if not pending_eval_jobs:
            return control

        wandb_run_id = None
        wandb_project = None
        wandb_entity = None
        try:
            import wandb

            current_run = getattr(wandb, "run", None)
            if current_run is not None:
                wandb_run_id = getattr(current_run, "id", None)
                wandb_project = getattr(current_run, "project", None)
                wandb_entity = getattr(current_run, "entity", None)
        except Exception:
            pass

        dataset_summary = ", ".join(job["dataset_name"] for job in pending_eval_jobs)
        rank0_print(
            f"Queueing async post-save eval for {checkpoint_dir} "
            f"with vlmeval class {self.model_class_name} on [{dataset_summary}]"
        )

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        pythonpath_entries = [str(self.root_dir), str(self.root_dir / "src")]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_entries.extend(path for path in existing_pythonpath.split(os.pathsep) if path)
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_entries))

        model_name = f"{checkpoint_dir.parent.name}_{checkpoint_dir.name}".replace("-", "_")

        with open(submit_log_path, "w", encoding="utf-8") as submit_log_file:
            for eval_job in pending_eval_jobs:
                config_path = eval_job["eval_root"] / "config.json"
                log_path = eval_job["eval_root"] / "run.log"
                config = {
                    "model": {
                        model_name: {
                            "class": self.model_class_name,
                            "model_path": str(checkpoint_dir),
                        }
                    },
                    "data": {
                        eval_job["dataset_name"]: {
                            "class": "ImageVQADataset",
                            "dataset": eval_job["dataset_name"],
                        }
                    },
                }
                with open(config_path, "w", encoding="utf-8") as config_file:
                    json.dump(config, config_file, indent=2, ensure_ascii=True)
                    config_file.write("\n")

                eval_command = [
                    env.get("PYTHON", "python"),
                    str(self.root_dir / "src" / "eval" / "run.py"),
                    "--config",
                    str(config_path),
                    "--work-dir",
                    str(eval_job["eval_root"]),
                ]
                logger_command = [
                    env.get("PYTHON", "python"),
                    str(self.root_dir / "src" / "train" / "post_save_eval_logger.py"),
                    "--eval-root",
                    str(eval_job["eval_root"]),
                    "--dataset-name",
                    eval_job["dataset_name"],
                    "--checkpoint-step",
                    str(int(state.global_step)),
                ]
                if wandb_run_id and wandb_project:
                    logger_command.extend(["--wandb-run-id", wandb_run_id, "--wandb-project", wandb_project])
                    if wandb_entity:
                        logger_command.extend(["--wandb-entity", wandb_entity])

                shell_command = (
                    "set -uo pipefail\n"
                    f"cd {shlex.quote(str(self.root_dir))}\n"
                    f"{shlex.join(eval_command)} >> {shlex.quote(str(log_path))} 2>&1\n"
                    "status=$?\n"
                    "if [ \"$status\" -ne 0 ]; then\n"
                    f"  echo \"$status\" > {shlex.quote(str(eval_job['failed_path']))}\n"
                    "  exit \"$status\"\n"
                    "fi\n"
                    f"{shlex.join(logger_command)} >> {shlex.quote(str(log_path))} 2>&1\n"
                    "logger_status=$?\n"
                    "if [ \"$logger_status\" -ne 0 ]; then\n"
                    f"  echo \"logger:$logger_status\" > {shlex.quote(str(eval_job['failed_path']))}\n"
                    "  exit \"$logger_status\"\n"
                    "fi\n"
                    f"echo ok > {shlex.quote(str(eval_job['done_path']))}\n"
                )

                proc = subprocess.Popen(
                    ["/bin/bash", "-lc", shell_command],
                    cwd=str(self.root_dir),
                    env=env,
                    stdout=submit_log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
                submit_log_file.write(
                    json.dumps(
                        {
                            "dataset_name": eval_job["dataset_name"],
                            "pid": proc.pid,
                            "eval_root": str(eval_job["eval_root"]),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                submit_log_file.flush()
                rank0_print(
                    f"Queued async post-save eval for {checkpoint_dir} [{eval_job['dataset_name']}] "
                    f"locally. Launcher pid: {proc.pid}. Submission log: {submit_log_path}"
                )

        return control


def infer_post_save_eval_datasets(data_path: str) -> tuple[str, ...]:
    normalized_data_path = str(data_path).lower()
    if "textvqa" in normalized_data_path:
        return ("TextVQA_VAL",)
    if "chartqa" in normalized_data_path:
        return ("ChartQA_TEST",)
    if "docvqa" in normalized_data_path:
        return ("ChartQA_TEST",)
    return VlmEvalOnSaveCallback.DEFAULT_DATASET_NAMES


def validate_distillation_args(distillation_args) -> None:
    if distillation_args.alpha < 0.0:
        raise ValueError("--alpha must be >= 0.")
    if not distillation_args.post_save_eval_root.strip():
        raise ValueError("--post_save_eval_root must be non-empty.")
    if distillation_args.student_temperature is not None and distillation_args.student_temperature <= 0:
        raise ValueError("--student_temperature must be > 0.")
    if distillation_args.teacher_temperature is not None and distillation_args.teacher_temperature <= 0:
        raise ValueError("--teacher_temperature must be > 0.")
    if distillation_args.temperature <= 0:
        raise ValueError("--temperature must be > 0.")


def log_distillation_setup(
    *,
    teacher_ids,
    data_args,
    training_args,
    distillation_args,
    gradient_checkpointing_kwargs,
) -> None:
    post_save_eval_datasets = infer_post_save_eval_datasets(data_args.data_path)
    rank0_print("=" * 80)
    rank0_print("Logits Distillation Training")
    rank0_print("=" * 80)
    rank0_print(f"Student Model: {distillation_args.student_model_id}")
    rank0_print(f"Teacher Model(s): {teacher_ids}")
    rank0_print(
        "Teacher Weighting: orientation vote"
        if len(teacher_ids) > 1
        else "Teacher Weighting: uniform mean"
    )
    rank0_print("Objective: CE + alpha * KD")
    rank0_print(f"KD Function: {distillation_args.distillation_loss}")
    rank0_print(f"Alpha: {distillation_args.alpha}")
    resolved_student_temperature = (
        distillation_args.temperature
        if distillation_args.student_temperature is None
        else distillation_args.student_temperature
    )
    resolved_teacher_temperature = (
        distillation_args.temperature
        if distillation_args.teacher_temperature is None
        else distillation_args.teacher_temperature
    )
    rank0_print(f"Student Temperature: {resolved_student_temperature}")
    rank0_print(f"Teacher Temperature: {resolved_teacher_temperature}")
    rank0_print(f"Skip Student EOS: {distillation_args.skip_student_eos}")
    rank0_print(f"Skip Teacher EOS: {distillation_args.skip_teacher_eos}")
    rank0_print(f"Post-save Eval Root: {distillation_args.post_save_eval_root}")
    rank0_print(f"Post-save Eval Dataset(s): {post_save_eval_datasets}")
    if training_args.gradient_checkpointing:
        rank0_print(f"Gradient Checkpointing Kwargs: {gradient_checkpointing_kwargs}")
    rank0_print("=" * 80)
