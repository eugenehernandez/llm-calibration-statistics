# Do Language Models Know When They Are Wrong?
### Calibration of LLM confidence on introductory statistics questions

When a language model says it is 90% sure, is it right 90% of the time? This project measures that gap on 65 multiple-choice questions in probability and statistics, the material I teach in introductory courses. It compares two ways of getting a confidence number from a model.

* **Verbalized confidence.** The model states a probability, from 0 to 100, that its answer is correct.
* **Sampling consistency.** The model answers the same question several more times at a higher temperature, and confidence is the share of those answers that agree with its main answer.

Calibration matters for safety and oversight. If people are going to rely on models in areas where they cannot check every answer themselves, the model's stated confidence needs to mean something. A model that is confidently wrong is harder to supervise than one that signals its own uncertainty.

## Research questions

1. Is the model's stated confidence calibrated, or is it systematically overconfident?
2. Does sampling consistency track correctness better than the confidence the model states?
3. Does the gap between confidence and accuracy grow as questions get harder?

## Method

**Questions.** 65 four-option multiple-choice questions written for this project (20 easy, 22 medium, 23 hard) across 15 topics, including conditional probability, distributions, expected value, hypothesis testing, and regression. Many wrong options are the mistakes students actually make, such as using the independence formula for mutually exclusive events or ignoring the base rate in a Bayes problem. Answer positions are shuffled with a fixed seed, so the correct letter is spread evenly across A to D. Every numeric answer key is verified by `scripts/verify_answers.py`, which recomputes each value exactly or by simulation and checks that no wrong option is equally close. The questions were drafted with help from an AI assistant and checked by me.

**Models.** Two models were tested through the Together AI API, Llama 3.3 70B Instruct Turbo and Qwen 3.5 9B. Qwen 3.5 reasons at length before answering by default, so it was run with that reasoning turned off, which matches how Llama answers. Smaller models were also tried, but none of the ones available on Together's serverless API could complete an answer in the required format.

**Elicitation.** For each question the model answers once at temperature 0, which gives its main answer and stated confidence, and 5 more times at temperature 0.7. The prompt asks for brief reasoning followed by an answer letter and a confidence from 0 to 100. Responses are limited to 1,024 tokens. The few that reached that limit before giving an answer (about 2% for each model, all on hard questions) were collected again with a limit of 2,048 tokens.

**Metrics.**

| Metric | What it measures |
|---|---|
| Accuracy | Share of main answers that are correct |
| Overconfidence | Mean confidence minus accuracy |
| Brier score | Mean squared error of the confidence (lower is better) |
| Expected calibration error (ECE) | Average gap between confidence and accuracy across 10 bins, weighted by bin size |
| AUROC | Probability that a correct answer got higher confidence than a wrong one (0.5 means no signal) |

All metrics come with 95% bootstrap intervals, resampling questions.

## Results

**Main answers** (one temperature 0 answer per question, 95% bootstrap intervals in brackets)

| Model | Method | Accuracy | Overconfidence | Brier | ECE | AUROC |
|---|---|---|---|---|---|---|
| Llama 3.3 70B | Verbalized | 0.985 [0.954, 1.000] | 0.013 | 0.016 | 0.013 | 0.47 |
| Llama 3.3 70B | Consistency | 0.985 [0.954, 1.000] | -0.043 | 0.026 | 0.043 | 0.98 |
| Qwen 3.5 9B | Verbalized | 1.000 | -0.002 | 0.000 | 0.002 | undefined |
| Qwen 3.5 9B | Consistency | 1.000 | 0.000 | 0.000 | 0.000 | undefined |

AUROC is undefined for Qwen because it made no mistakes. Full intervals for every metric are in `results/summary_metrics.csv`.

**All responses by difficulty** (main answer plus 5 samples per question, Llama 3.3 70B)

| Difficulty | Responses | Accuracy | Mean stated confidence | Gap |
|---|---|---|---|---|
| Easy | 120 | 0.992 | 1.000 | 0.008 |
| Medium | 132 | 0.962 | 0.984 | 0.022 |
| Hard | 137 | 0.898 | 0.982 | 0.084 |

Qwen 3.5 9B answered all 390 of its responses correctly.

![Reliability diagram for Llama 3.3 70B](results/figures/reliability_Llama-3.3-70B-Instruct-Turbo.png)

### Findings

**Both models are at or near the ceiling on this question set.** Llama 3.3 70B answered 64 of 65 questions correctly and Qwen 3.5 9B answered all 65. Both stated a confidence of 95% or higher on every main answer. With so few errors the summary calibration metrics look nearly perfect, but they say little, because there is almost nothing to calibrate against.

**Overconfidence appears on the hardest questions.** Across all six responses per question, Llama's accuracy fell from 99% on easy questions to 90% on hard ones, while its average stated confidence stayed between 98% and 100%. The gap between confidence and accuracy grew from under 1 point on easy questions to about 8 points on hard ones. Of Llama's 20 wrong responses, 16 came with a stated confidence of 90% or higher.

**Sampling consistency caught the error that stated confidence missed.** Llama's one wrong main answer was a four-door version of the Monty Hall problem (question h01). It answered 3/4 with 100% confidence, but its six responses gave three different answers, and only 2 of the 5 samples agreed with its main answer. AUROC tells the same story (0.47 for stated confidence and 0.98 for consistency), but with a single wrong main answer this is an example rather than a statistical result.

**In short,** stated confidence was nearly constant and carried little information about correctness, sampling consistency was more informative, and overconfidence grew with difficulty. Testing these patterns properly requires a question set on which models make more mistakes.

## Limitations

* **Ceiling effect.** Both models answered nearly every question correctly, so the calibration metrics rest on very few errors. This is the main limitation of the study.
* **Small sample.** 65 questions give wide intervals, so small differences between methods may not be meaningful.
* **Multiple choice.** With four options, a blind guess is right 25% of the time, and the options themselves can hint at the answer.
* **Familiar puzzles.** Some hard questions, such as the Monty Hall and boy born on Tuesday problems, are well known and likely appear in training data. Accuracy on them may reflect memory rather than reasoning.
* **Coarse consistency scores.** With 5 samples, consistency can only take six values (0, 0.2, 0.4, 0.6, 0.8, 1).
* **One prompt.** Stated confidence is known to be sensitive to prompt wording, and only one wording is tested here.
* **Subjective difficulty.** Difficulty labels were assigned by me before any model was run.
* **Reasoning models.** Newer hybrid reasoning models sometimes spend their whole response on internal reasoning and never give a final answer, even when asked not to. Qwen 3.5 was run with reasoning turned off, and two smaller models could not be used for this reason. Evaluations like this one need to check for this failure explicitly.

## Running it

Requires Python 3.9 or later and a [Together AI](https://www.together.ai/) API key.

```
pip install -r requirements.txt
cp .env.example .env            # then paste your key into .env
python -m pytest                # unit tests for the metrics and answer parsing
python scripts/verify_answers.py
python run_experiment.py --dry-run      # free test with a simulated model
python run_experiment.py --limit 5      # quick real test on 5 questions
python run_experiment.py                # full run, 390 API calls
jupyter notebook analysis.ipynb
```

The full run with the default model (`meta-llama/Llama-3.3-70B-Instruct-Turbo`) uses roughly 250,000 tokens. To compare another model, pass `--model` with any name from `python run_experiment.py --list-models`. Each model's responses go to their own file in `results/raw/`, and the notebook compares every model it finds there. Runs can be interrupted and resumed, because already saved responses are skipped. Some newer models, such as Qwen 3.5, reason at length before answering by default and can run out of room before giving a final answer. For those, add `--no-thinking` so every model answers the same way.

## Repository layout

```
data/questions.jsonl        the question set (correct answer listed first, shuffled at run time)
calibration/prompts.py      prompt construction and answer parsing
calibration/models.py       Together AI client and a simulated model for dry runs
calibration/metrics.py      Brier score, ECE, AUROC, reliability tables, bootstrap intervals
run_experiment.py           collects model responses
analysis.ipynb              metrics, reliability diagrams, and error analysis
scripts/verify_answers.py   independent check of every numeric answer key
tests/                      unit tests
results/raw/                saved model responses, one JSON Lines file per model
```

## Possible extensions

* Replace multiple choice with free-response questions graded against the answer key, which removes the hints that options give and should lower accuracy enough to measure calibration.
* Add graduate-level questions, or a weaker model, so that errors are common enough to study.
* Use token log probabilities as a third confidence measure.
* Test several prompt wordings to measure how sensitive stated confidence is.
