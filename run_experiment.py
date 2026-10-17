"""Asks a model every question and saves each response to a JSON Lines file.

For each question the model answers once at temperature 0 (its "main" answer)
and several more times at a higher temperature. The analysis notebook uses
these to compare two confidence measures:
  verbalized   the confidence number the model states with its main answer
  consistency  the share of sampled answers that agree with the main answer

Results are appended one line per response, so an interrupted run picks up
where it left off. Delete the output file to start over.

Examples
  python run_experiment.py --dry-run                      # free test with a fake model
  python run_experiment.py --list-models                  # chat models on your account
  python run_experiment.py --limit 5                      # quick real test, 5 questions
  python run_experiment.py                                # full run with the default model
  python run_experiment.py --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
"""
import argparse
import json
import pathlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from calibration.models import SimulatedModel, TogetherModel
from calibration.prompts import build_messages, parse_response

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


def load_questions(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def slug(model_name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.split("/")[-1])


def load_done(path):
    """(question id, sample index) pairs already saved, so they are skipped."""
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["question_id"], r["sample"]))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--samples", type=int, default=5,
                    help="extra answers per question at --temperature (default 5)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    ap.add_argument("--questions", default=str(ROOT / "data" / "questions.jsonl"))
    ap.add_argument("--out", help="output file (default results/raw/<model>.jsonl)")
    ap.add_argument("--seed", type=int, default=0, help="seed for answer-choice order")
    ap.add_argument("--dry-run", action="store_true",
                    help="use a simulated model, no API calls, output to results/dry_run/")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        for name in TogetherModel(args.model).list_chat_models():
            print(name)
        return

    if args.dry_run:
        model = SimulatedModel()
        out = pathlib.Path(args.out or ROOT / "results" / "dry_run" / "simulated.jsonl")
    else:
        model = TogetherModel(args.model)
        out = pathlib.Path(args.out or ROOT / "results" / "raw" / f"{slug(args.model)}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.questions)[: args.limit]
    done = load_done(out)

    # Sample 0 is the main answer at temperature 0. Samples 1..K use --temperature.
    jobs = []
    for q in questions:
        for s in range(args.samples + 1):
            if (q["id"], s) not in done:
                jobs.append((q, s))
    total = len(questions) * (args.samples + 1)
    print(f"Model: {model.model}\nOutput: {out}\n"
          f"{total - len(jobs)} of {total} responses already saved, {len(jobs)} to go.")
    if not jobs:
        return

    lock = threading.Lock()
    finished = 0

    def run(job):
        q, s = job
        messages, correct_letter = build_messages(q, seed=args.seed)
        temperature = 0.0 if s == 0 else args.temperature
        if isinstance(model, SimulatedModel):
            reply = model.complete(messages, temperature, seed=s,
                                   question=q, correct_letter=correct_letter)
        else:
            reply = model.complete(messages, temperature, seed=s)
        letter, confidence = parse_response(reply["text"])
        return {
            "question_id": q["id"], "topic": q["topic"], "difficulty": q["difficulty"],
            "sample": s, "temperature": temperature, "model": model.model,
            "correct_letter": correct_letter, "answer": letter, "confidence": confidence,
            "correct": letter == correct_letter, "parsed": letter is not None,
            "response": reply["text"],
            "prompt_tokens": reply["prompt_tokens"],
            "completion_tokens": reply["completion_tokens"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(out, "a") as f:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            q, s = futures[future]
            try:
                record = future.result()
            except Exception as e:                  # keep going, rerun to retry
                errors += 1
                print(f"  failed {q['id']} sample {s}: {type(e).__name__}: {e}")
                continue
            with lock:
                f.write(json.dumps(record) + "\n")
                f.flush()
                finished += 1
                if finished % 25 == 0 or finished == len(jobs):
                    print(f"  {finished}/{len(jobs)} saved")

    print(f"Done. {finished} saved, {errors} failed.")
    if errors:
        print("Run the same command again to retry the failed requests.")


if __name__ == "__main__":
    main()
