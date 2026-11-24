# Copy First, Check the Field Later: A Judge-Free Benchmark for Tool-Result Mis-Grounding in LLM Agents

Does an LLM agent notice when a tool returns something that looks right but
does not answer the question? This repository contains `uptake-probe`, a
judge-free benchmark for **tool-result mis-grounding**, together with the
dataset and the model outputs of the paper below.

Every tool result carries a **nonce value** (a random code such as `BL74-QU42`
that no model can know in advance). If the code shows up in the agent's answer,
it came from the result, so reliance is scored by exact match. No LLM judge is
involved.

Paper: Takuya Omi, *Copy First, Check the Field Later: A Judge-Free Diagnosis of
Tool-Result Mis-Grounding in LLM Agents*, Findings of EACL 2027.

## Installation

Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tokimoa/agent-tool-faithfulness.git
cd agent-tool-faithfulness
uv sync                 # scoring and tests: no GPU, no third-party runtime dependencies
uv run pytest
```

Running local models needs Apple Silicon: `uv sync --extra mlx`.

## Quickstart

Score a set of answers without loading any model. The repository ships the
answers logged for the paper, so this prints the profile of one of the paper's
models:

```bash
uv run python debug/score_example.py benchmark/dataset.jsonl \
    --answers results/raw/baseline_raw.jsonl --model Phi-4-mini
```

```
condition                  n  rate                  outcomes
  attr_mismatch          125  mis-grounding 0.59    confab_decoy=39 confab_novel=35 vague=51
  good                   125  utilization   0.94    missed=8 used=117
  irrelevant             125  mis-grounding 0.88    confab_decoy=110 vague=15
  partial                125  mis-grounding 0.00    abstain=18 vague=107
  stale                  125  mis-grounding 0.67    abstain=25 confab_decoy=84 vague=16
  ...
```

The model uses a correct result 94% of the time, but when the result concerns a
different entity (`irrelevant`) it still reports a code in 88% of episodes.

## Evaluating your own model or agent

### A local MLX model, with the built-in ReAct harness

```bash
uv run --extra mlx python debug/score_example.py benchmark/dataset.jsonl \
    mlx-community/Phi-4-mini-instruct-4bit
```

Use `--mock` in place of the model id for a dry run that needs no model.

### Any agent, by scoring its answers offline

Each line of `benchmark/dataset.jsonl` is self-contained: it holds the
`question` and the tool `results` (each with a `snippet` and a `full_text`).
Feed these to your agent in whatever way it consumes tool results, write one
line per instance,

```json
{"instance_id": "s20000_k5_n1_center", "answer": "EU53-VM98"}
```

and score the file:

```bash
uv run python debug/score_example.py benchmark/dataset.jsonl --answers my_answers.jsonl
```

### Any model, inside the harness

The harness accepts any object with a `name`, a `generate(prompt) -> str`
method and a `count_tokens(text) -> int` method:

```python
from uptake_probe.scaffolds import run
from uptake_probe.schema import DeliveryMode, ResultQuality
from uptake_probe.synth import make_instance

class MyBackend:
    name = "my-model"

    def generate(self, prompt: str) -> str:
        return call_my_model(prompt)       # must reply with "Action: SEARCH[...]", "FETCH[...]" or "ANSWER[...]"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

instance = make_instance(seed=1, k=5, necessary_quality=ResultQuality.STALE,
                         style="record", domain="payments")
record = run(instance, MyBackend(), DeliveryMode.AGENTIC, max_steps=14)
print(record.answer)
```

`DeliveryMode.PASSIVE` places all results in the prompt instead of running the
search-and-fetch loop.

## How answers are scored

| outcome | meaning |
|---|---|
| `used` / `missed` | on a correct result: the planted value is, or is not, in the answer |
| `confab_decoy` | on a mismatched result: the answer asserts the planted wrong value |
| `confab_novel` | the answer asserts some other complete code |
| `abstain` | the answer declines or flags the problem (this takes priority, even if a code is also quoted) |
| `vague` | no complete code and no abstention |

**Utilization** is the share of `used`; **mis-grounding** is the share of
`confab_decoy` plus `confab_novel`. Abstentions are detected by a regular
expression that was validated against hand labels (`src/uptake_probe/detect.py`).

## The dataset

`benchmark/dataset.jsonl` holds 5,097 instances; `benchmark/DATACARD.md`
lists its modules and cells. Each instance asks for one attribute of one
entity, and the result that should answer it is in one of these conditions:

| condition | what the result contains | expected behaviour |
|---|---|---|
| `good` | the requested value | report it |
| `irrelevant`, `irrelevant_near`, `irrelevant_far` | a record of a different entity | abstain |
| `attr_mismatch` | a different field of the right entity | abstain |
| `stale` | a value marked as superseded | abstain |
| `partial` | a truncated value | abstain |
| `error`, `empty`, `redacted` | an explicit failure | abstain |

The remaining conditions are intervention variants of these (cues added to the
question or the record, and tool-side fixes such as returning only the requested
field).

Generation is deterministic in a seed, so you can draw a fresh set that no model
has seen:

```python
from uptake_probe.schema import ResultQuality
from uptake_probe.synth import make_instance

instance = make_instance(seed=987654, k=5, necessary_quality=ResultQuality.ATTR_MISMATCH,
                         style="record", domain="logistics")
print(instance.question)
for result in instance.results:
    print(result.full_text)
```

`style="record"` gives multi-field business records in the domains `logistics`,
`payments`, `devops` and `support`; `style="template"` gives single-sentence
results in the `access` domain.

## Reproducing the paper

All per-instance model outputs are in `results/raw/`, and every reported rate
can be recomputed from them without running a model. [REPRODUCING.md](REPRODUCING.md)
lists the script, result file and raw log behind each table.

## Repository layout

| path | contents |
|---|---|
| `src/uptake_probe/` | generator, scorer and agent harness |
| `benchmark/` | the frozen dataset and its data card |
| `debug/` | experiment runners and analysis scripts |
| `results/` | aggregated results (`*.json`) and per-instance outputs (`raw/`) |
| `tests/` | unit tests |

## Citation

```bibtex
@inproceedings{omi2027copyfirst,
  title     = {Copy First, Check the Field Later: A Judge-Free Diagnosis of Tool-Result Mis-Grounding in {LLM} Agents},
  author    = {Omi, Takuya},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```

## License

Code: MIT (`LICENSE`). Dataset and released outputs: CC BY 4.0
(`benchmark/DATACARD.md`). An AI coding assistant was used in parts of the
implementation and analysis scripting; the paper's Acknowledgments give the full
statement.
