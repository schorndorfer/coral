# Azure GPT-5.6 Sol Evaluation Notebook Design

## Goal

Add a clean, documented marimo notebook that evaluates the CORAL inference
dataset with an Azure-hosted GPT-5.6 Sol deployment. The notebook starts with
an eight-input smoke test and requires an explicit user action before a full
515-input run.

## Scope

The root checkout contains the inference CSV, the legacy scorer, and output
artifacts, but it does not contain the structured benchmark runner used for
the earlier local Qwen run. The notebook therefore owns the Azure request,
structured-response validation, resumable checkpointing, cost reporting, and
legacy-output conversion. It does not modify the existing legacy evaluator or
claim that its repository-wide macro is free from the known Cartesian-product
issue.

## Configuration and credentials

The notebook reads these environment variables at execution time:

- `AZURE_OPENAI_API_KEY`: Azure OpenAI credential; never written to a file or
  displayed.
- `AZURE_OPENAI_ENDPOINT`: Azure resource endpoint.
- `AZURE_OPENAI_DEPLOYMENT`: optional deployment name, defaulting to
  `gpt-5.6-sol`.
- `AZURE_OPENAI_API_VERSION`: optional Azure API version, with a documented
  default in the notebook.

The notebook fails before making a request when required credentials are
missing. It reports the selected endpoint host and deployment, but redacts the
credential.

## Notebook flow

1. **Introduction and safety.** Explain the benchmark, pricing assumptions,
   output locations, and the distinction between the legacy scorer and the
   actual-prompt-only summary.
2. **Configuration.** Display the current data path, Azure deployment,
   input limit, maximum output tokens, reasoning effort, and a spend cap.
3. **Dataset preview and cost estimate.** Load `data/coral_inference.csv`,
   show the selected rows, estimate a pre-run upper-bound cost, and require
   the limit to be positive and no greater than 515.
4. **Smoke-test control.** An explicit button runs the first eight selected
   inputs. Script mode uses the same default limit but does not contact Azure
   unless an opt-in environment variable is set.
5. **Resumable execution.** For each selected input, send the task-specific
   prompt to Azure, request JSON output, validate it, append a terminal
   checkpoint record to JSONL, and skip terminal records when resumed. A
   validation failure gets one corrective retry; transport/API failures are
   recorded and stop only that input.
6. **Results.** Show status counts, request latency, Azure token usage and
   actual cost when usage is available, plus a table of validation failures.
7. **Scoring export.** Convert accepted results to the repository’s legacy
   response CSV, run `coral.benchmarking.evaluate_model`, and produce the
   actual-prompt-only score summary. The notebook labels the repository-wide
   macro as flawed and uses the observed summary for comparisons.
8. **Full-run control.** A separate confirmation widget is the only route to
   set the limit to all 515 inputs. It retains the same spend cap and resume
   behavior as the smoke test.

## Azure request contract

The notebook uses the Azure OpenAI Python client and the Responses API with
the user-selected Azure deployment. It requests structured JSON compatible
with the current CORAL task schema, uses deterministic decoding where the API
supports it, and records the raw response plus parsed result separately.

Requests are one input at a time. This makes the cost ceiling visible, allows
safe resumption, and avoids losing progress if a request fails. The notebook
checks the projected spend before every request and stops before exceeding the
configured cap.

## Artifacts

All generated files live under `output/` and include the deployment name in
their filename:

- `gpt56_sol_azure.jsonl`: canonical checkpoint records, including request
  metadata, terminal validation status, errors, usage, and cost.
- `gpt56_sol_azure_legacy.csv`: accepted responses converted at the legacy
  scorer boundary.
- `gpt56_sol_azure_relation_instance_scores.csv`, aggregate scores, and
  reformatted scores: unchanged scorer output.
- `gpt56_sol_azure_relation_observed_summary.csv`: actual-prompt-only summary.

No artifact contains an API key.

## Error handling

- Missing Azure configuration: block execution with actionable setup text.
- Invalid deployment or authentication error: preserve the error in the
  notebook display and do not write secrets to artifacts.
- Invalid or nonconforming JSON: make one corrective retry, then checkpoint a
  terminal validation failure.
- Spend cap reached: checkpoint completed work and stop cleanly.
- Existing terminal checkpoint: skip it on resume.

## Tests and verification

Extract pure helpers from the notebook into a small module so they can be
tested without Azure credentials. Tests cover configuration validation,
request-cost accounting, spend-cap enforcement, checkpoint resume behavior,
and conversion of a validated response to a legacy CSV row. Azure calls are
dependency-injected and tested with a fake client.

Before handoff, run the focused unit tests, `uvx marimo check` on the
notebook, and script-mode execution with Azure calls disabled. The notebook
itself never runs a paid request without the explicit smoke-test or full-run
action.
