import argparse
import json
from pathlib import Path

import pandas as pd


RESULT_PATTERNS = (
    "acc.csv",
    "score.csv",
    "score.json",
    "acc.json",
    "acc_all.csv",
    "merged_score.json",
    "*_score.csv",
)

DATASET_TABLE_COLUMNS = {
    "ChartQA_TEST": ("Human", "Augmented", "Overall"),
    "DocVQA_VAL": ("Overall",),
    "TextVQA_VAL": ("Overall",),
}


def coerce_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def find_result_file(eval_root: Path) -> Path | None:
    for pattern in RESULT_PATTERNS:
        matches = sorted(eval_root.rglob(pattern))
        if matches:
            return matches[0]
    fallback = sorted(
        path for path in eval_root.rglob("*")
        if path.is_file() and path.name not in {"config.json", "run.log", "done", "failed", "submit.log", "summary.json"}
    )
    return fallback[0] if fallback else None


def coerce_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda column: column.map(coerce_value)) if not df.empty else df


def extract_scalar_metrics(df: pd.DataFrame):
    scalar_metrics = {}
    if df.empty:
        return scalar_metrics

    if len(df) == 1:
        row = df.iloc[0].to_dict()
        return {
            str(key): value
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    if len(df.columns) == 2:
        label_col, value_col = list(df.columns)
        values = df[value_col]
        if values.map(lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)).all():
            labels = df[label_col].astype(str)
            if labels.is_unique:
                return dict(zip(labels.tolist(), values.tolist(), strict=False))

    return scalar_metrics


def parse_csv(result_file: Path):
    df = coerce_frame(pd.read_csv(result_file))
    return df, extract_scalar_metrics(df)


def parse_json(result_file: Path):
    with result_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        df = coerce_frame(pd.json_normalize(payload, sep="."))
        return df, extract_scalar_metrics(df)

    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            df = coerce_frame(pd.json_normalize(payload, sep="."))
            return df, extract_scalar_metrics(df)
        df = coerce_frame(pd.DataFrame({"value": payload}))
        return df, extract_scalar_metrics(df)

    df = coerce_frame(pd.DataFrame({"value": [payload]}))
    return df, extract_scalar_metrics(df)


def parse_result_file(result_file: Path):
    if result_file.suffix.lower() == ".csv":
        return parse_csv(result_file)
    if result_file.suffix.lower() == ".json":
        return parse_json(result_file)
    return pd.DataFrame(), {}


def build_logged_table(dataset_name, checkpoint_step, table_df, scalar_metrics):
    if not table_df.empty:
        logged_table = table_df.reset_index(drop=True).copy()
    elif scalar_metrics:
        logged_table = pd.DataFrame([scalar_metrics])
    else:
        return pd.DataFrame()

    if dataset_name in DATASET_TABLE_COLUMNS:
        lower_to_column = {str(column).lower(): column for column in logged_table.columns}
        selected_metric_columns = [
            lower_to_column[column_name.lower()]
            for column_name in DATASET_TABLE_COLUMNS[dataset_name]
            if column_name.lower() in lower_to_column
        ]
        if not selected_metric_columns:
            selected_metric_columns = [
                column_name
                for column_name in logged_table.columns
                if pd.api.types.is_numeric_dtype(logged_table[column_name])
            ]
        logged_table = logged_table.loc[:, selected_metric_columns]

    logged_table.insert(0, "dataset_name", dataset_name)
    logged_table.insert(0, "checkpoint_step", checkpoint_step)
    return logged_table


def log_to_wandb(*, run_id, project, entity, dataset_name, checkpoint_step, eval_root, table_df, scalar_metrics):
    import wandb

    logged_table = build_logged_table(
        dataset_name=dataset_name,
        checkpoint_step=checkpoint_step,
        table_df=table_df,
        scalar_metrics=scalar_metrics,
    )

    run = wandb.init(
        project=project,
        entity=entity or None,
        id=run_id,
        dir=str(eval_root),
        settings=wandb.Settings(
            mode="shared",
            x_primary=False,
            x_label=f"post_eval_{dataset_name.lower()}_ckpt_{checkpoint_step}",
            x_update_finish_state=False,
        ),
    )

    payload = {}
    for metric_name, metric_value in scalar_metrics.items():
        payload[f"post_save_eval/{dataset_name}/{metric_name}"] = metric_value

    if not logged_table.empty:
        payload[f"post_save_eval/{dataset_name}/table"] = wandb.Table(dataframe=logged_table)

    if payload:
        run.log(payload, step=checkpoint_step)
    run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    summary_path = eval_root / "summary.json"
    result_file = find_result_file(eval_root)
    table_df, scalar_metrics = (pd.DataFrame(), {})
    if result_file is not None:
        table_df, scalar_metrics = parse_result_file(result_file)
    logged_table = build_logged_table(
        dataset_name=args.dataset_name,
        checkpoint_step=args.checkpoint_step,
        table_df=table_df,
        scalar_metrics=scalar_metrics,
    )
    rows = [
        {key: coerce_value(value) for key, value in row.items()}
        for row in logged_table.to_dict(orient="records")
    ]

    summary = {
        "dataset_name": args.dataset_name,
        "checkpoint_step": args.checkpoint_step,
        "eval_root": str(eval_root),
        "result_file": str(result_file) if result_file is not None else None,
        "scalar_metrics": scalar_metrics,
        "rows": rows,
        "logged_to_wandb": False,
    }

    if args.wandb_run_id and args.wandb_project:
        log_to_wandb(
            run_id=args.wandb_run_id,
            project=args.wandb_project,
            entity=args.wandb_entity,
            dataset_name=args.dataset_name,
            checkpoint_step=args.checkpoint_step,
            eval_root=eval_root,
            table_df=table_df,
            scalar_metrics=scalar_metrics,
        )
        summary["logged_to_wandb"] = True

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
        f.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
