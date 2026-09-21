# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "absl-py",
#     "evaluate",
#     "marimo",
#     "nltk",
#     "numpy",
#     "openai",
#     "pandas",
#     "rouge-score",
#     "scikit-learn",
#     "torch",
#     "transformers",
# ]
# ///

"""Deliberately gated Azure GPT-5.6 Sol replication of the CORAL paper protocol."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from pathlib import Path
    import sys
    from time import perf_counter

    import marimo as mo
    import pandas as pd

    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

    from coral.paper_replication import (
        PAPER_SOURCE_COMMIT,
        build_request_grid,
        checkpoint_spend,
        completed_keys,
        create_azure_client,
        load_azure_settings,
        load_source,
        project_grid_cost,
        read_checkpoint,
        run_replication,
        score_completed_records,
        smoke_is_complete,
        smoke_is_scored,
        write_artifacts,
        write_smoke_marker,
    )

    return (
        PAPER_SOURCE_COMMIT,
        Path,
        build_request_grid,
        checkpoint_spend,
        completed_keys,
        create_azure_client,
        load_azure_settings,
        load_source,
        mo,
        os,
        pd,
        perf_counter,
        project_grid_cost,
        read_checkpoint,
        run_replication,
        score_completed_records,
        smoke_is_complete,
        smoke_is_scored,
        write_artifacts,
        write_smoke_marker,
    )


@app.cell
def _(PAPER_SOURCE_COMMIT, mo):
    mo.md(
        f"""
        # GPT-5.6 Sol · CORAL paper replication

        Frozen protocol: OncLLMExtraction `{PAPER_SOURCE_COMMIT[:7]}`. The
        paper's original prompts cover **40 documents × 2 sections × 14 tasks
        = 1,120 requests**, using named-tuple responses and paper-style defaults.
        Scoring averages actual completed outputs by relation, then takes the
        unweighted mean across relations. An incomplete run is a partial result.
        Published GPT-4 values are BLEU-4 **0.73**, ROUGE-1 **0.72**, and EM F1
        **0.51** (Sushil et al., NEJM AI, 2024; doi:10.1056/AIdbp2300110).
        `difference_vs_paper` is this run's score minus that published value.

        This run uses the Azure **Responses API**, with **low reasoning effort**
        and a fixed **4,096** output-token ceiling. The historical GPT-4 run used
        Chat Completions without reasoning controls. An allowlisted parser
        preserves valid named tuples and paper defaults without executing text.

        First run the eight-request smoke test and inspect its checkpoint and
        top-line artifact. Full execution requires all eight smoke keys for the
        current deployment, verified smoke scores, and the exact confirmation
        `RUN 1120`. The smoke proof is tied to the current protocol and source
        data, and is saved only after scoring and artifact writing succeed. It resumes
        those eight completions, leaving 1,112 requests. The spend cap includes
        prior checkpoint spend. Script mode reads the optional
        `CORAL_PAPER_REPLICATION_SPEND_CAP` environment variable and defaults to
        $100. Pricing: $4/M input and $20/M output tokens.
        """
    )
    return


@app.cell
def _(Path):
    data_path = Path("data/coral_inference.csv")
    output_path = Path("output")
    checkpoint_path = output_path / "gpt56_sol_paper_replication.jsonl"
    topline_path = output_path / "gpt56_sol_paper_replication_topline.csv"
    smoke_marker_path = output_path / "gpt56_sol_paper_replication_smoke.json"
    return checkpoint_path, data_path, output_path, smoke_marker_path, topline_path


@app.cell
def _(build_request_grid, data_path, load_source, mo, project_grid_cost):
    mo.stop(
        not data_path.exists(),
        mo.md(f"Source missing: `{data_path}`. Add the inference CSV before a run."),
    )
    source = load_source(data_path)
    grid = build_request_grid(source)
    projected_full_cost = project_grid_cost(grid)
    projected_smoke_cost = project_grid_cost(grid.head(8))
    setup_message = (
        f"Validated {len(grid):,} requests from 80 document sections. "
        f"Projected smoke cost: ${projected_smoke_cost:.2f}; "
        f"projected full cost: ${projected_full_cost:.2f}. "
        "Projection estimates input tokens and assumes the full output ceiling."
    )
    if mo.app_meta().mode == "script":
        print(setup_message)
    mo.md(setup_message)
    return grid, source


@app.cell
def _(load_azure_settings, mo, os):
    try:
        settings = load_azure_settings(os.environ)
        configuration_message = (
            "Azure configuration validated; credentials and endpoint are redacted. "
            f"Deployment: `{settings.deployment}`. "
            "AZURE_OPENAI_DEPLOYMENT defaults to gpt-5.6-sol."
        )
    except ValueError as configuration_error:
        settings = None
        configuration_message = (
            f"Azure configuration unavailable: {configuration_error}. "
            "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT; "
            "optionally set AZURE_OPENAI_DEPLOYMENT and AZURE_OPENAI_API_VERSION."
        )
    mo.md(configuration_message)
    return (settings,)


@app.cell
def _(mo):
    spend_cap = mo.ui.number(start=0, step=1, value=100, label="Spend cap (USD)")
    smoke_button = mo.ui.run_button(label="Run eight-request smoke test", kind="warn")
    full_confirmation = mo.ui.text(
        placeholder="RUN 1120", label="Full-run confirmation"
    )
    full_button = mo.ui.run_button(label="Run full 1,120-request grid", kind="danger")
    mo.vstack([spend_cap, smoke_button, full_confirmation, full_button])
    return full_button, full_confirmation, smoke_button, spend_cap


@app.cell
def _(
    checkpoint_path,
    full_button,
    full_confirmation,
    grid,
    mo,
    os,
    read_checkpoint,
    settings,
    smoke_button,
    smoke_is_complete,
    smoke_is_scored,
    smoke_marker_path,
    source,
    spend_cap,
):
    import math

    is_script_mode = mo.app_meta().mode == "script"
    requested_action = (
        os.getenv("CORAL_PAPER_REPLICATION_RUN", "")
        if is_script_mode
        else "full" if full_button.value else "smoke" if smoke_button.value else ""
    )
    confirmation = (
        os.getenv("CORAL_PAPER_REPLICATION_CONFIRM", "")
        if is_script_mode else full_confirmation.value
    )
    # Reading the cap in this cell makes every control change recheck the buttons.
    raw_spend_cap = (
        os.getenv("CORAL_PAPER_REPLICATION_SPEND_CAP", str(spend_cap.value))
        if is_script_mode
        else spend_cap.value
    )
    try:
        effective_spend_cap = float(raw_spend_cap)
    except (TypeError, ValueError):
        effective_spend_cap = 0.0
        spend_cap_error = True
    else:
        spend_cap_error = not math.isfinite(effective_spend_cap) or effective_spend_cap < 0
    checkpoint_records = read_checkpoint(checkpoint_path)
    if requested_action not in ("smoke", "full"):
        authorized_action = ""
        selection_message = (
            "Inert: no run requested. Use a run button, or set "
            "CORAL_PAPER_REPLICATION_RUN=smoke. Full script execution requires "
            "CORAL_PAPER_REPLICATION_RUN=full and "
            "CORAL_PAPER_REPLICATION_CONFIRM='RUN 1120' after the smoke test. "
            "Set CORAL_PAPER_REPLICATION_SPEND_CAP to override the $100 default."
        )
    elif spend_cap_error:
        authorized_action = ""
        selection_message = (
            "Rejected: script spend cap (CORAL_PAPER_REPLICATION_SPEND_CAP) "
            "must be a finite nonnegative number."
        )
    elif requested_action == "full" and confirmation != "RUN 1120":
        authorized_action = ""
        selection_message = "Rejected: full execution requires exact confirmation RUN 1120."
    elif settings is None:
        authorized_action = ""
        selection_message = "Rejected: valid Azure configuration is required before a run."
    elif requested_action == "full" and not smoke_is_complete(
        grid, checkpoint_records, settings.deployment
    ):
        authorized_action = ""
        selection_message = (
            "Rejected: all eight isolated smoke keys must be completed for the "
            "current deployment before full execution."
        )
    elif requested_action == "full" and not smoke_is_scored(
        grid, source, checkpoint_records, settings.deployment, smoke_marker_path
    ):
        authorized_action = ""
        selection_message = (
            "Rejected: a verified scored smoke artifact is required for this "
            "deployment, protocol, and source. Run the smoke action again to "
            "score its completed responses and save the smoke proof."
        )
    else:
        authorized_action = requested_action
        selection_message = f"Authorized {requested_action} action."
    run_rows = (
        grid if authorized_action == "full"
        else grid.head(8) if authorized_action == "smoke"
        else grid.head(0)
    )
    if not authorized_action and is_script_mode:
        print(selection_message)
    mo.md(selection_message)
    return authorized_action, effective_spend_cap, run_rows


@app.cell
def _(
    authorized_action,
    checkpoint_path,
    create_azure_client,
    effective_spend_cap,
    pd,
    perf_counter,
    run_replication,
    run_rows,
    settings,
    smoke_marker_path,
):
    if authorized_action:
        if authorized_action == "smoke":
            # A failed new smoke action must not leave a previous success proof.
            smoke_marker_path.unlink(missing_ok=True)
        print(
            f"Starting {authorized_action} run: {len(run_rows):,} requests; "
            f"deployment {settings.deployment}; spend cap ${effective_spend_cap:.2f}."
        )
        run_started = perf_counter()
        client = create_azure_client(settings)
        run_results = run_replication(
            run_rows, client, settings, checkpoint_path, effective_spend_cap,
            progress=print,
        )
        run_elapsed = perf_counter() - run_started
        status_counts = run_results["status"].value_counts().to_dict()
        print(
            f"Completed {authorized_action} run: {status_counts}; "
            f"elapsed {run_elapsed:.1f}s; checkpoint: {checkpoint_path}."
        )
    else:
        run_results = pd.DataFrame()
        run_elapsed = 0.0
    return run_elapsed, run_results


@app.cell
def _(
    authorized_action,
    checkpoint_path,
    grid,
    mo,
    output_path,
    read_checkpoint,
    run_results,
    score_completed_records,
    settings,
    smoke_is_complete,
    smoke_marker_path,
    source,
    write_artifacts,
    write_smoke_marker,
):
    # Depend on run_results so the checkpoint is read after the runner finishes.
    refreshed_records = read_checkpoint(checkpoint_path)
    scoring_records = [
        record for record in refreshed_records
        if record.get("status") == "completed"
        and settings is not None and record.get("model") == settings.deployment
    ]
    scores = None
    artifact_paths = None
    scoring_message = ""
    if authorized_action and not run_results.empty and scoring_records:
        try:
            scores = score_completed_records(source, scoring_records, model=settings.deployment)
            artifact_paths = write_artifacts(scores, output_path)
            if authorized_action == "smoke" and smoke_is_complete(
                grid, refreshed_records, settings.deployment
            ):
                write_smoke_marker(
                    grid, source, refreshed_records, settings.deployment,
                    smoke_marker_path, artifact_paths.topline,
                )
        except Exception:
            # Resource, parser, and I/O failures can carry source text: show only
            # actionable safe guidance, and never turn failure into smoke proof.
            scores = None
            artifact_paths = None
            if authorized_action == "smoke":
                smoke_marker_path.unlink(missing_ok=True)
            scoring_message = (
                "Scoring or artifact writing failed. Check local metric resources "
                "and output access, then rerun the smoke action. Full execution "
                "requires a successful scored smoke artifact."
            )
            print(scoring_message)
    mo.md(scoring_message)
    return artifact_paths, refreshed_records, scores


@app.cell
def _(
    artifact_paths,
    checkpoint_spend,
    completed_keys,
    grid,
    mo,
    refreshed_records,
    run_elapsed,
    run_results,
    settings,
):
    completed_identities = completed_keys(refreshed_records)
    current_completed = sum(
        (str(row.doc_idx), str(row.section_name), str(row.task), settings.deployment)
        in completed_identities
        for row in grid.itertuples(index=False)
    ) if settings is not None else 0
    summary_statuses = (
        run_results["status"].value_counts().to_dict() if not run_results.empty else {}
    )
    summary_message = (
        f"Completed keys: {current_completed:,}/1,120 for the configured deployment. "
        f"Checkpoint spend: ${checkpoint_spend(refreshed_records):.4f}. "
        f"Latest run: {run_elapsed:.1f}s. Statuses: {summary_statuses or 'no run'}. "
        + (
            f"Saved top-line scores: `{artifact_paths.topline}`." if artifact_paths
            else "Saved scores, if present, are from the last authorized run."
        )
    )
    mo.md(summary_message)
    return


@app.cell
def _(mo, pd, scores, topline_path):
    topline_columns = ["metric", "gpt56_sol", "paper_gpt4", "difference_vs_paper"]
    if scores is not None:
        topline = scores.topline.loc[:, topline_columns]
    elif topline_path.exists():
        # Loading saved scores is inert: no metric resources or clients are loaded.
        topline = pd.read_csv(topline_path, usecols=topline_columns).loc[:, topline_columns]
    else:
        topline = pd.DataFrame(columns=topline_columns)
    topline_display = (
        mo.ui.table(topline, selection=None, pagination=False)
        if not topline.empty else mo.md("No scored results yet. Run the smoke test to begin.")
    )
    topline_display
    return


if __name__ == "__main__":
    app.run()
