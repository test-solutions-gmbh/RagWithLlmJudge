# LLM Judge Benchmarking

*Anupam Krishnamurthy*

A live-demo harness that accompanies the LLM Judge Benchmarking talk. The demo features a UI where the following steps can be followed:

1. A RAG model's performance is evaluated against a set of ground-truth questions by a human evaluator.
2. The human evaluator's verdicts are then used to create a benchmark.
3. A LLM judge is used to evaluate the performance of the RAG model against the benchmark.
4. The performance of multiple LLM judge runs are compared to the benchmark.

## Setup

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) API key (used for the RAG model, the embeddings, and the judge)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Get an OpenRouter API key

1. Sign up at [openrouter.ai](https://openrouter.ai/).
2. Go to [openrouter.ai/keys](https://openrouter.ai/keys) and click **Create Key**.
3. Copy the key — it starts with `sk-or-v1-...`.

The default models in `.env.example` (`openai/gpt-oss-120b:free`) are on OpenRouter's free tier, so you can run the demo without adding credits. If you want to swap in a paid model (e.g. `openai/gpt-4o`), top up your account at [openrouter.ai/credits](https://openrouter.ai/credits).

### Configure

```bash
cp .env.example .env
```

... then edit `.env` and fill in `OPENROUTER_API_KEY` with the key you just created. The other variables (`RAG_MODEL`, `JUDGE_MODEL`, `EMBEDDING_MODEL`) come pre-filled with sensible defaults — change them only if you want to try a different model.

## Commands

### Generate RAG responses

Runs the RAG model over every question in [data/ground_truths.json](data/ground_truths.json) and writes a fresh `results.json` under a timestamped folder in `data/results/`.

```bash
python -m src.cli generate-rag-responses
# → data/results/<timestamp>/results.json
```

### Run the LLM judge

Reads the most recent `results.json` (or a specific one via `--path`) and fills in the `llm_judge` field for every entry, in place.

```bash
python -m src.cli evaluate-responses-with-llm-judge
# or
python -m src.cli evaluate-responses-with-llm-judge --path data/results/<timestamp>/results.json
```

### Launch the live UI

A lean Streamlit app renders one question at a time, collects the audience's verdict via two buttons, and reveals the LLM judge's verdict alongside — with an agree/disagree flag. A Summary view aggregates the whole run.

Each deployment is bound to **one** results file — pass it via CLI or env var:

```bash
# Bind to a specific run (explicit, preferred for the live demo)
streamlit run src/app.py -- --results-path data/results/<timestamp>/results.json

# Omitting this defaults to the most recent run in data/results/
streamlit run src/app.py
```

Human verdicts and comments are persisted back into the bound results file as soon as they're recorded, so you can re-run `evaluate-responses-with-llm-judge` or reload the UI without losing them.

## Human evaluation benchmark

Once you've finished evaluating a `results.json` (every entry has a human verdict), you can freeze it as a **benchmark** — an immutable snapshot of questions, RAG responses, and human verdicts. You can then run any judge model against that frozen snapshot, accumulate multiple judge runs, and compare each judge's alignment with the humans.

Benchmarks live in their own tree:

```text
data/
  benchmarks/
    <benchmark-name>/
      benchmark.json
      judge_runs/
        <judge-model>__<timestamp>.json
```

The benchmark itself is written once and never mutated; each judge run is a separate, append-only file.

### Promote a results file to a benchmark

```bash
python -m src.cli promote-benchmark \
  --path data/results/<timestamp>/results.json \
  --name baseline
```

This validates that every entry has a non-null human verdict, strips the `llm_judge` block, and writes `data/benchmarks/baseline/benchmark.json`. Pass `--force` to overwrite an existing benchmark with the same name.

### Run a judge against a benchmark

```bash
python -m src.cli judge-benchmark --name baseline
# or with a specific model
python -m src.cli judge-benchmark --name baseline --model openai/gpt-oss-20b:free
```

The judge model defaults to `$JUDGE_MODEL` (or `openai/gpt-oss-120b:free`). Each invocation writes a new file under `data/benchmarks/baseline/judge_runs/` and prints an alignment summary against the benchmark's human verdicts.

### List benchmarks

```bash
python -m src.cli list-benchmarks
```

## Links
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/pdf/2306.05685)
- [BeyondQuality Community](https://beyondquality.org/)
- [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)
- [Free course on AI Evals](https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/free_courses/ai_evals_for_everyone) 
