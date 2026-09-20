# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "openai",
#     "pandas",
#     "pydantic",
# ]
# ///

"""Run a deliberately gated Azure GPT-5.6 Sol CORAL evaluation."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math
    import os
    from pathlib import Path
    import sys

    import marimo as mo
    import pandas as pd
    import pydantic

    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

    def read_checkpoint_records(checkpoint_path: Path) -> list[dict[str, object]]:
        """Read complete checkpoint records while ignoring a truncated JSONL tail."""
        if not checkpoint_path.exists():
            return []
        records: list[dict[str, object]] = []
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    from coral.azure_evaluation import (
        Usage,
        estimate_cost,
        load_azure_settings,
        run_evaluation,
        write_legacy_csv,
        write_observed_summary,
    )

    return (
        Path,
        Usage,
        estimate_cost,
        json,
        load_azure_settings,
        math,
        mo,
        os,
        pd,
        pydantic,
        read_checkpoint_records,
        run_evaluation,
        write_legacy_csv,
        write_observed_summary,
    )


@app.cell
def _(mo, pydantic):
    mo.md(
        f"""
        # Azure GPT-5.6 Sol evaluation

        This notebook sends clinical text only after an explicit run action. It starts
        with an **eight-input smoke test** and keeps the projected and observed spend
        visible. GPT-5.6 Sol pricing used here is $4.00 / million input tokens and
        $20.00 / million output tokens.

        The repository-wide legacy macro is flawed because it includes synthetic
        Cartesian-product rows. Use the observed-only summary below for comparisons.
        This notebook was generated with marimo and uses pydantic {pydantic.VERSION}.
        """
    )
    return


@app.cell
def _(Path):
    data_path = Path("data/coral_inference.csv")
    output_path = Path("output")
    checkpoint_path = output_path / "gpt56_sol_azure.jsonl"
    legacy_path = output_path / "gpt56_sol_azure_legacy.csv"
    instance_score_path = output_path / "gpt56_sol_azure_relation_instance_scores.csv"
    aggregate_score_path = output_path / "gpt56_sol_azure_relation_aggregate_scores.csv"
    reformatted_score_path = output_path / "gpt56_sol_azure_relation_reformatted_scores.csv"
    observed_summary_path = output_path / "gpt56_sol_azure_relation_observed_summary.csv"
    return (
        aggregate_score_path,
        checkpoint_path,
        data_path,
        instance_score_path,
        legacy_path,
        observed_summary_path,
        output_path,
        reformatted_score_path,
    )


@app.cell
def _(data_path, mo, pd):
    required_columns = {"doc_idx", "section_name", "section_text", "task"}
    if data_path.exists():
        loaded_rows = pd.read_csv(data_path)
        input_rows = loaded_rows.rename(columns={"inference_subtype": "task"})
        missing_columns = required_columns.difference(input_rows.columns)
        if missing_columns:
            dataset_message = mo.callout(
                f"`{data_path}` is missing: {', '.join(sorted(missing_columns))}.",
                kind="warn",
            )
            input_rows = pd.DataFrame(columns=sorted(required_columns))
        else:
            dataset_message = mo.md(
                f"Loaded **{len(input_rows)}** inference rows from `{data_path}`. "
                "The full-run control uses at most the first 515 rows."
            )
    else:
        input_rows = pd.DataFrame(columns=sorted(required_columns))
        dataset_message = mo.callout(
            f"`{data_path}` is not present. Create the inference CSV before running Azure.",
            kind="warn",
        )
    dataset_message
    return (input_rows,)


@app.cell
def _(load_azure_settings, mo, os):
    azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")
    if azure_key.strip() and azure_endpoint.strip():
        azure_environment = {
            "AZURE_OPENAI_API_KEY": azure_key,
            "AZURE_OPENAI_ENDPOINT": azure_endpoint,
        }
        if azure_deployment.strip():
            azure_environment["AZURE_OPENAI_DEPLOYMENT"] = azure_deployment
        if azure_api_version.strip():
            azure_environment["AZURE_OPENAI_API_VERSION"] = azure_api_version
        settings = load_azure_settings(
            azure_environment
        )
        endpoint_host = settings.endpoint.split("//", maxsplit=1)[-1].split("/", maxsplit=1)[0]
        configuration_message = mo.md(
            f"Azure configuration is available for `{endpoint_host}` using deployment "
            f"`{settings.deployment}` (API version `{settings.api_version}`). "
            "The API key is redacted and is never written to an artifact."
        )
    else:
        settings = None
        configuration_message = mo.callout(
            "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT before a run. "
            "No Azure client has been constructed.",
            kind="warn",
        )
    configuration_message
    return (settings,)


@app.cell
def _(mo):
    input_limit = mo.ui.number(
        start=1, stop=515, step=1, value=8, label="Preview / projection input limit"
    )
    output_token_ceiling = mo.ui.number(
        start=64, stop=4096, step=64, value=512, label="Maximum output tokens per request"
    )
    reasoning_effort = mo.ui.dropdown(
        options=["low", "medium", "high"], value="low", label="Reasoning effort"
    )
    spend_cap = mo.ui.number(start=0, step=0.25, value=10.0, label="Spend cap (USD)")
    smoke_button = mo.ui.run_button(
        label="Run 8-input smoke evaluation", kind="warn"
    )
    full_confirmation = mo.ui.text(
        placeholder="Type RUN 515 exactly", label="Full-run confirmation"
    )
    full_run_button = mo.ui.run_button(
        label="Run full 515-input evaluation", kind="danger"
    )
    mo.vstack(
        [
            input_limit,
            output_token_ceiling,
            reasoning_effort,
            spend_cap,
            smoke_button,
            full_confirmation,
            full_run_button,
        ]
    )
    return (
        full_confirmation,
        full_run_button,
        input_limit,
        output_token_ceiling,
        reasoning_effort,
        smoke_button,
        spend_cap,
    )


@app.cell
def _(Usage, estimate_cost, input_limit, input_rows, math, mo, output_token_ceiling, pd):
    requested_limit = int(input_limit.value or 8)
    preview_rows = input_rows.head(min(max(requested_limit, 1), 515))
    estimated_input_tokens = sum(
        max(1, math.ceil(len(str(section_text)) / 4)) + 256
        for section_text in preview_rows.get("section_text", pd.Series(dtype=str))
    )
    projected_cost = estimate_cost(
        Usage(estimated_input_tokens, len(preview_rows) * int(output_token_ceiling.value or 512))
    )
    projection_display = mo.vstack(
        [
            mo.md(
                f"### Data preview and upper-bound projection\n"
                f"Previewing **{len(preview_rows)}** rows (maximum 515). At the current token "
                f"ceiling, the conservative upper-bound request cost is **${projected_cost:.4f}**."
            ),
            preview_rows.head(8),
        ]
    )
    projection_display
    return (preview_rows,)


@app.cell
def _(
    checkpoint_path,
    full_confirmation,
    full_run_button,
    input_rows,
    mo,
    os,
    output_token_ceiling,
    pd,
    reasoning_effort,
    run_evaluation,
    settings,
    smoke_button,
    spend_cap,
):
    is_script_mode = mo.app_meta().mode == "script"
    script_requested = is_script_mode and os.getenv("CORAL_AZURE_RUN", "") == "1"
    full_requested = (
        not is_script_mode
        and bool(full_run_button.value)
        and full_confirmation.value == "RUN 515"
    )
    smoke_requested = (not is_script_mode and bool(smoke_button.value)) or script_requested
    run_requested = full_requested or smoke_requested
    if full_requested:
        run_rows = input_rows.head(515)
        run_label = "full 515-input evaluation"
    elif smoke_requested:
        run_rows = input_rows.head(8)
        run_label = "8-input smoke evaluation"
    else:
        run_rows = input_rows.head(0)
        run_label = "no evaluation"

    if run_requested and settings is None:
        run_results = pd.DataFrame()
        run_message = mo.callout(
            "Azure configuration is required before a request can be sent.", kind="warn"
        )
    elif run_requested and run_rows.empty:
        run_results = pd.DataFrame()
        run_message = mo.callout("There are no valid inference rows to run.", kind="warn")
    elif run_requested:
        # The Azure client exists only inside this explicit run branch. In script
        # mode that branch requires CORAL_AZURE_RUN=1.
        from openai import AzureOpenAI, OpenAI

        if settings.endpoint.lower().endswith(".services.ai.azure.com/openai/v1"):
            # Azure AI Foundry provides this OpenAI-compatible v1 base URL.
            client = OpenAI(base_url=settings.endpoint, api_key=settings.api_key)
        else:
            client = AzureOpenAI(
                api_key=settings.api_key,
                azure_endpoint=settings.endpoint,
                api_version=settings.api_version,
            )
        run_results = run_evaluation(
            run_rows,
            client,
            settings,
            checkpoint_path,
            float(spend_cap.value or 0),
            int(output_token_ceiling.value or 512),
            str(reasoning_effort.value),
        )
        run_message = mo.md(f"Completed **{run_label}**. Checkpoint: `{checkpoint_path}`.")
    else:
        run_results = pd.DataFrame()
        run_message = mo.md(
            "No Azure request has been made. Use the smoke button, or type `RUN 515` "
            "exactly and then use the full-run button. Script mode needs `CORAL_AZURE_RUN=1`."
        )
    run_message
    return (run_results,)


@app.cell
def _(mo, pd, run_results):
    if run_results.empty:
        results_display = mo.md("### Results\nNo evaluation results are available yet.")
    else:
        status_counts = run_results["validation_status"].value_counts().rename_axis("status").reset_index(name="count")
        observed_cost = run_results.get("cost", pd.Series(dtype=float)).fillna(0).sum()
        observed_tokens = (
            run_results.get("input_tokens", pd.Series(dtype=float)).fillna(0).sum()
            + run_results.get("output_tokens", pd.Series(dtype=float)).fillna(0).sum()
        )
        results_display = mo.vstack(
            [
                mo.md(
                    f"### Results\nObserved Azure usage: **{int(observed_tokens)} tokens**; "
                    f"observed cost: **${observed_cost:.4f}**."
                ),
                status_counts,
                run_results[run_results["validation_status"].isin(["api_failed", "validation_failed"])],
            ]
        )
    results_display
    return


@app.cell
def _(
    aggregate_score_path,
    checkpoint_path,
    instance_score_path,
    legacy_path,
    mo,
    observed_summary_path,
    output_path,
    Path,
    pd,
    reformatted_score_path,
    read_checkpoint_records,
    run_results,
    settings,
    write_legacy_csv,
    write_observed_summary,
):
    if run_results.empty or settings is None:
        legacy_export_message = mo.md("### Legacy export\nRun an evaluation before exporting accepted responses.")
    else:
        checkpoint_records = read_checkpoint_records(checkpoint_path)
        legacy_frame = write_legacy_csv(checkpoint_records, legacy_path, settings.deployment)
        if legacy_frame.empty:
            legacy_export_message = mo.callout(
                "No validated responses are available for legacy scoring yet.", kind="warn"
            )
        else:
            from coral.benchmarking.evaluate_model import evaluate, get_annots, get_outputs

            scored_data = get_annots("coral_inference.csv", str(Path("data")))
            scored_outputs = get_outputs(legacy_path.name, str(output_path))
            evaluate(
                scored_data,
                scored_outputs,
                instance_score_path.name,
                aggregate_score_path.name,
                reformatted_score_path.name,
                str(output_path),
            )
            instance_scores = pd.read_csv(instance_score_path)
            exported_keys = legacy_frame.loc[:, ["doc_idx", "section_name", "task"]]
            write_observed_summary(instance_scores, exported_keys, observed_summary_path)
            legacy_export_message = mo.md(
                f"### Legacy export\nWrote `{legacy_path}` and the gpt56_sol_azure scoring "
                "artifacts. The repository-wide legacy macro remains flawed; use the "
                "observed-only summary below."
            )
    legacy_export_message
    return


@app.cell
def _(mo, observed_summary_path, pd):
    if observed_summary_path.exists():
        observed_summary = pd.read_csv(observed_summary_path)
        score_summary_display = mo.vstack(
            [
                mo.md(
                    "### Observed-only score summary\nThis excludes the legacy scorer's "
                    "synthetic Cartesian-product rows."
                ),
                observed_summary,
            ]
        )
    else:
        score_summary_display = mo.md(
            "### Observed-only score summary\nNo observed-only summary has been produced yet."
        )
    score_summary_display
    return


if __name__ == "__main__":
    app.run()
