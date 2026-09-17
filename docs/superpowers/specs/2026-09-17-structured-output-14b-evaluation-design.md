# Structured-Output 14B Evaluation Design

## Objective

Add a local structured-generation path for CORAL and use it to evaluate
Qwen2.5-14B-Instruct on the same 515 document-section-task inputs used by the
existing 3B baseline. The new path must produce typed, auditable JSON with
verbatim evidence while retaining compatibility with the existing tuple-based
scorer.

Fixing the evaluator's synthetic `unknown == unknown` scoring flaw is explicitly
out of scope for this change. Results may still be converted for comparison,
but the evaluator itself will not be modified.

## Model and Runtime

- Model: Qwen2.5-14B-Instruct, loaded from local weights.
- Backend: Hugging Face Transformers with PyTorch MPS.
- Precision: FP16.
- Decoding: deterministic greedy decoding (`do_sample=False`).
- Initial batch size: 2. A smoke test may increase it only after confirming
  acceptable memory use and latency on this Mac's 64 GB unified memory.
- Output limits: 512 new tokens for symptoms, genomics, biomarkers, histology,
  stage, TNM, and grade; 1,024 new tokens for radiology, procedures,
  metastasis, and medication tasks.
- Data handling: clinical text and model inference remain local.

## Canonical Output Format

Structured JSONL becomes the canonical inference artifact. Each line represents
one input and contains:

- the document, section, and task key;
- model path or identifier and revision when available;
- schema version and decoding configuration;
- raw model responses for the initial attempt and optional retry;
- validation status, structured validation errors, and retry count;
- generation timing; and
- the validated structured response, when validation succeeds.

Every task response uses a common envelope:

```json
{
  "task": "symptoms",
  "records": [
    {
      "symptom": "abdominal pain",
      "datetimes": ["three weeks ago"],
      "evidence_quotes": ["abdominal pain began three weeks ago"]
    }
  ]
}
```

The 14 tasks each have a task-specific record schema. Missing scalar values use
JSON `null`; missing collections use empty arrays. The literal string
`"unknown"` is not part of the canonical representation, although the legacy
adapter may emit it where required by the existing scorer.

Each record requires at least one non-empty `evidence_quote`. Evidence must be a
verbatim substring of the source note section after line-ending normalization.
Model-written explanations or rationales are not accepted as evidence.

## Components

### Task schemas

A dedicated structured-output module using Pydantic v2 defines:

- the common response envelope;
- one typed record schema for each extraction task;
- the mapping from task name to response schema;
- enum constraints such as medication continuity and future-medication
  consideration; and
- schema-version metadata.

The schemas are independent of model loading and legacy tuple classes.

### Prompt construction

The structured prompt builder reuses the clinical requirements in the existing
task prompts while replacing named-tuple instructions with the selected task's
JSON schema. It instructs the model to:

- return exactly one JSON object;
- include only facts supported by the supplied note section;
- use `null` and empty arrays consistently; and
- attach at least one verbatim supporting quote to every record.

### Generation and validation

For each input, the runner:

1. Selects the schema from the task name.
2. Builds the system and user messages.
3. Generates a deterministic response.
4. Extracts the JSON object without evaluating executable text.
5. Validates the envelope and task-specific record types.
6. Verifies every evidence quote against the source section.
7. Saves a successful result or performs one corrective retry.

The retry prompt includes the original raw response and concise validation
errors. It asks for a complete corrected response, not a patch. The same schema,
source text, and deterministic decoding settings are used.

If the retry also fails, the runner records both raw responses and all errors.
It does not fabricate an empty result or substitute `unknown` values.

### Checkpointing and resume

The runner appends one completed JSONL record at a time. A run key consists of
document ID, section name, task, model identifier, and schema version. At
startup, valid existing keys are loaded and skipped, allowing interruption and
safe resumption without duplicate inference.

Malformed or truncated checkpoint lines are reported and ignored rather than
treated as completed inputs.

### Legacy adapter

A separate adapter converts validated task records into the existing CORAL
named tuples. Evidence and run metadata are omitted from this representation.
The adapter maps `null` and empty collections to the legacy `unknown` convention
only at this boundary.

The adapter produces a scorer-compatible CSV without changing the current
evaluator. A terminal validation failure is emitted explicitly as the legacy
default tuple with a `conversion_status=validation_failed` column; it is never
silently omitted and remains traceable to the failed canonical JSONL record.
The existing scorer may ignore this additional status column. The canonical
JSONL remains the source of truth.

## Evaluation Procedure

1. Download Qwen2.5-14B-Instruct into the ignored local `models/` directory.
2. Run a small multi-task smoke set to verify loading, memory, JSON generation,
   schema validation, evidence checking, and legacy conversion.
3. Select a safe batch size, beginning at 2.
4. Run the same 515 inputs used by the Qwen2.5-3B baseline.
5. Report validation success rate, retry recovery rate, unrecoverable failures,
   throughput, and peak MPS memory.
6. Convert successful results for legacy scoring and report both the existing
   repository summary and the actual-prompt-only summary, clearly labeling the
   former as affected by the known evaluator flaw.

## Testing

Automated tests cover:

- valid and invalid examples for all 14 task schemas;
- task-to-schema dispatch;
- null, empty-list, enum, and required-field behavior;
- extraction from plain and fenced JSON responses;
- rejection of non-verbatim or empty evidence;
- successful first-pass validation;
- retry recovery and retry exhaustion;
- JSONL checkpoint writing and resume-key handling;
- malformed checkpoint recovery;
- conversion of every task schema to its legacy named tuple; and
- a golden scorer-compatible CSV fixture.

Model-dependent tests are separate smoke tests and are not required for the
ordinary unit-test suite. Before the full run, the smoke test must demonstrate
that the 14B model loads on MPS and completes representative simple and complex
tasks without exceeding available memory.

## Success Criteria

- All 14 tasks have explicit, tested schemas.
- Every accepted record has verified verbatim evidence.
- Invalid responses receive at most one deterministic corrective retry.
- Failed responses remain auditable and are never silently converted to
  successful empty answers.
- An interrupted run resumes without duplicating completed inputs.
- All 515 inputs receive a terminal success or failure record.
- Legacy scorer input is reproducibly derived from canonical JSONL.
- Existing repository tests and the new structured-output tests pass.
