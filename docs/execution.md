# Running token experiments

Execution settings belong to each experiment in `config/experiment_templates.yaml`.
Sampling and existing commands continue to work with the cached vocabulary.

| Setting | Purpose |
| --- | --- |
| `cpu_workers` | Separate Python processes for CPU annotators such as Markov. |
| `concurrency` | Maximum simultaneous requests for an annotator such as Ollama. |
| `batch_size` | Independent vocabulary tokens per work item or LLM request. |
| `retries` | Additional attempts after a failed batch; default 2. |
| `resume` | Reuse completed annotations from checkpoints; default true. |
| `checkpoint_key` | Manual annotation namespace. Change it to force fresh labels. |

Use `cpu_workers: 10` for the ten-core Markov configuration. Keep `concurrency: 1`
for CPU execution. For Ollama, keep `cpu_workers: 1` and tune `concurrency` and
`batch_size`. The queue is bounded by the worker/request count, so a full
vocabulary does not create hundreds of thousands of pending jobs.

Mistral defaults to four concurrent requests with one independent token per
request. Larger batches are available by changing `batch_size`, but should be
evaluated before using their labels for training: an initial 32-token test found
that eight-token batches changed nine labels compared with individual requests.

## Ollama parallel requests

The Python request limit and Ollama's server limit are separate. To allow four
simultaneous requests when using the macOS Ollama app:

```bash
launchctl setenv OLLAMA_NUM_PARALLEL 4
```

Quit and reopen Ollama to apply the setting. If starting the server from a
terminal instead, quit the app first and run:

```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
```

Ollama allocates additional context memory for parallel requests. Increasing
concurrency is useful only while throughput improves within available memory.
See the [Ollama configuration documentation](https://docs.ollama.com/faq).

Each LLM batch contains unrelated tokens. Structured output uses fixed item IDs
so labels attach to the original spellings, including hyphens and repetitions.
Every ID, label, and confidence is validated before a batch is accepted.

## Checkpoints and recovery

All token experiments share the file-backed SQLite database
`data/dataset/annotations.sqlite3`, excluded from version control alongside the
published dataset. For a custom dataset location, the database is placed in that
dataset's directory. Changing the CSV output location does not change checkpoint
storage. The main process commits completed batches while other workers
continue running. A bounded retry failure stops new submissions, drains running
work, and records the failure. Rerun the command to retry unfinished tokens.
Failed responses are never substituted with fabricated labels.

Each experiment controls its checkpoint namespace with `checkpoint_key` in
`config/experiment_templates.yaml`. The program does not infer invalidation from
model settings, installed model digests, prompts, source code, batch size, or
transition matrices. Keep the key unchanged to reuse labels; change it manually
when you want a fresh annotation set. Changing the sample size or worker count
can reuse matching annotations. Frequencies always come from the current
vocabulary.

Set `resume: false` for a fresh timing measurement. It still writes every batch to
the database under a new run identity, preserving existing checkpoints. Run
identities are included in timing summaries so the stored annotations can be
retrieved. Subsequent resumable runs reuse the configured `checkpoint_key`; fresh
benchmark runs remain separate. No run uses an in-memory SQLite database.

Final CSVs contain the selected tokens in a stable order. Timing summaries report
newly processed and cached tokens separately, so reuse does not inflate inference
throughput. A single run also writes its configuration and metrics beside the CSV.

## Benchmarking

Run the benchmark on the local cached vocabulary:

```bash
uv run python scripts/benchmark_tokens.py
```

Use `--model markov` or `--model ollama_mistral_7b` to select one annotator.
The script checks one versus ten CPU workers on the full vocabulary and several
batch/concurrency combinations on the same 32-token sample for Mistral.
It excludes model warm-up, stores fresh annotations in the dataset database, and saves timings,
labels, failures, and agreement under `data/outputs/benchmarks/`.
Set `OLLAMA_HOST` to benchmark a separate server.

Agreement with the serial run measures consistency, not linguistic accuracy.
LLM labels remain provisional training labels; throughput does not establish
that a batched prediction is correct.

On this Apple M5, a 32-token sample with Mistral 7B produced the following warm
inference measurements. The server allowed four parallel requests in every run.

| Tokens per request | Concurrent requests | Tokens/second | Agreement with individual serial labels |
| --- | --- | --- | --- |
| 1 | 1 | 0.75 | 100% |
| 1 | 2 | 1.25 | 100% |
| 1 | 4 | 1.60 | 100% |
| 8 | 2 | 2.18 | 71.9% |
| 8 | 4 | 2.93 | 65.6% |

All rows passed structural validation. These small-sample results motivated
the four-request, one-token default; they do not establish accuracy or predict
full-corpus throughput. The four-slot model occupied approximately 6.6 GB
according to `ollama ps`. Markov annotated all 933,460 tokens in 3.59 seconds
with one worker and 3.43 seconds with ten, with identical labels. Timings include
the runner's earlier in-memory result storage but exclude CSV export and model
warm-up. These historical measurements predate the switch to disk-only SQLite;
rerun the benchmark to measure disk-backed execution.
