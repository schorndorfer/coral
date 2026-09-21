# GPT-5.6 Sol Paper-Protocol Replication Design

## Objective

Add a separate, resumable Azure GPT-5.6 Sol evaluation that reproduces the
published CORAL evaluation protocol closely enough for a meaningful comparison
with the paper's GPT-4 results. Preserve the existing 515-input,
evidence-grounded JSON evaluation as a distinct experiment.

The replication source is the local `OncLLMExtraction` repository at commit
`ddf1792`, specifically its advanced-inference prompts, named-tuple output
contract, parsing behavior, relation metrics, and aggregation logic.

## Protocol Boundary

The paper-compatible path will reproduce:

- the 14 advanced-inference tasks;
- the original system preamble and task-specific prompt text;
- all 40 documents and both `hpi` and `a&p` sections;
- the complete 1,120-request grid: 40 documents × 2 sections × 14 tasks;
- named-tuple response syntax;
- task-specific `unknown` defaults for empty or malformed responses;
- lowercased relation strings;
- smoothed BLEU-4, stemmed ROUGE-1, and exact-match precision, recall, and F1;
- per-task and per-relation aggregation over actual completed model outputs;
- an unweighted top-line mean over the resulting per-relation aggregates.

The following differences are unavoidable and will be disclosed in the
notebook:

- GPT-5.6 Sol is accessed through the Azure Responses API rather than the
  historical Chat Completions API.
- The deployed model supports reasoning controls that did not exist for the
  paper's GPT-4 run. The replication will use low reasoning effort and record
  that setting.
- Parsing will use an allowlisted AST interpreter rather than executing model
  output with `eval()`. Valid named tuples and paper-style fallbacks will retain
  equivalent behavior.

## Architecture

Create an isolated `coral.paper_replication` module. It owns the immutable
paper protocol and does not depend on the sibling repository at runtime. The
module will include:

1. The exact advanced-inference system preamble and 14 task prompts, with the
   source repository and commit recorded beside them.
2. A grid builder that extracts one canonical text for every document-section
   pair from `coral_inference.csv`, checks for conflicting duplicate text, and
   cross-joins those 80 sections with all 14 tasks.
3. A safe named-tuple parser that accepts only the paper's known constructors
   and literal values. Empty, non-answer, or malformed output maps to the
   task's default tuple, matching the paper's behavior.
4. A checkpointed Azure runner using the existing Azure environment variables,
   cost accounting, and endpoint/client selection. It sends the paper prompt
   unchanged and never adds a corrective prompt.
5. A scorer that iterates actual completed outputs, supplies default reference
   annotations when a document-section-task has no annotation row, and uses the
   paper's metric and aggregation definitions without fabricating missing model
   outputs.

Create a separate marimo notebook,
`notebooks/evaluate_gpt56_sol_paper_replication.py`, rather than adding a mode
switch to the existing notebook. This keeps checkpoints, prompts, semantics,
and displayed results unambiguous.

## Data Flow

1. Load `data/coral_inference.csv` and normalize `inference_subtype` to `task`.
2. Validate that it contains 40 document IDs, both required sections, and 14
   task definitions.
3. Collapse repeated task rows into 80 unique document-section texts, rejecting
   conflicting text for the same pair.
4. Cross-join the 80 sections with the ordered 14-task protocol to obtain 1,120
   unique requests.
5. Build each request as `section_text + original_task_prompt`, with the paper's
   advanced-inference system preamble supplied separately.
6. Send requests serially to the configured Azure GPT-5.6 Sol deployment and
   append each terminal result to the paper-replication JSONL checkpoint.
7. Parse completed response text into allowlisted named tuples, applying the
   paper default for empty or malformed text.
8. Join each completed output to its annotation row or task default, compute
   relation scores, and write instance and per-relation aggregate CSVs.
9. Compute the three top-line means and render only the comparison with the
   published GPT-4 values.

## Azure Execution and Cost Controls

The notebook will continue to use:

- `AZURE_OPENAI_API_KEY`;
- `AZURE_OPENAI_ENDPOINT`;
- `AZURE_OPENAI_DEPLOYMENT`, defaulting to `gpt-5.6-sol`.

The run will use a 4,096-token output ceiling and low reasoning effort. The
notebook will expose an eight-input smoke run and a full-run control guarded by
the exact confirmation text `RUN 1120`. It will show a conservative cost
projection and enforce a user-visible hard spend cap before every request.

Rate-limit and transient API retries may resend the identical request. No retry
may alter the paper prompt or add correction text. API failures are recorded
but excluded from scoring, consistent with the paper scorer iterating only
actual output rows. They remain resumable so a later run can retry them.

## Checkpoint and Artifacts

All new artifacts will use a `gpt56_sol_paper_replication` prefix:

- `output/gpt56_sol_paper_replication.jsonl`: canonical request checkpoint;
- `output/gpt56_sol_paper_replication_outputs.csv`: raw and parsed completed
  outputs in scorer-ready form;
- `output/gpt56_sol_paper_replication_instance_scores.csv`: document-level
  relation metrics;
- `output/gpt56_sol_paper_replication_relation_scores.csv`: paper-compatible
  per-relation aggregates;
- `output/gpt56_sol_paper_replication_topline.csv`: the three top-line metrics
  and differences from published GPT-4.

Existing `gpt56_sol_azure` artifacts will not be read, overwritten, or used to
resume this run because their prompts and response contracts are incompatible.

## Notebook Presentation

The notebook will document the protocol source and unavoidable API/model
differences. During execution it may show concise progress, status counts,
elapsed time, and spend. It will not render individual model outputs or
instance-level score tables.

The final visible result will be one table with:

- metric name;
- GPT-5.6 Sol paper-protocol score;
- published paper GPT-4 score;
- absolute difference.

Detailed outputs remain available only in the artifact files.

## Error Handling

- Missing Azure configuration prevents client construction and paid calls.
- Conflicting text for a document-section pair stops grid construction with an
  actionable error.
- An unexpected document, section, task, or grid cardinality stops the full run
  before any paid call.
- Spend-cap exhaustion is checkpointed and stops subsequent calls.
- Empty or malformed model text receives the paper's task-specific default and
  remains scoreable.
- API failures contain sanitized error information, are excluded from scores,
  and remain eligible for resume.
- A partial final JSONL line is ignored when resuming.

## Testing

Tests will make no network calls and will cover:

- the exact ordered set of 14 task prompts and their frozen provenance;
- construction of 1,120 unique request keys from 80 document sections;
- rejection of missing, duplicate-conflicting, or incomplete grid input;
- safe parsing of every allowed named-tuple type;
- paper-compatible defaults for empty, non-answer, and malformed outputs;
- rejection of arbitrary Python expressions without code execution;
- scoring parity with the replication repository on fixed valid and default
  fixtures;
- absence of synthetic model-output rows;
- checkpoint resume and retry eligibility for API failures;
- hard spend-cap enforcement;
- concise notebook and script output;
- marimo structural validation.

Before enabling a full run, an eight-input smoke test must complete and produce
parseable checkpoint rows and a top-line score artifact.

## Acceptance Criteria

- The request grid contains exactly 1,120 unique keys.
- Every request uses the corresponding paper task prompt without corrective or
  evidence-schema additions.
- Existing evidence-grounded checkpoints and artifacts remain unchanged.
- Valid named-tuple fixtures produce the same relation values and metric scores
  as the paper replication implementation.
- Missing model outputs are never replaced with synthetic perfect predictions.
- The notebook requires `RUN 1120` for full execution and respects the spend
  cap.
- The final display contains only the top-line paper comparison table, with
  concise run metadata outside the table.
- All automated tests and `marimo check` pass.
