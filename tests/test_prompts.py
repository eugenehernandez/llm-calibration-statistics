import json
import pathlib
from collections import Counter

from calibration.prompts import build_messages, parse_response, shuffled_choices

QUESTIONS = [json.loads(line) for line in
             open(pathlib.Path(__file__).resolve().parents[1] / "data" / "questions.jsonl")]


def test_parse_plain():
    assert parse_response("blah\nAnswer: B\nConfidence: 85") == ("B", 0.85)


def test_parse_markdown_and_percent():
    assert parse_response("**Answer:** (c)\n**Confidence:** 70%") == ("C", 0.70)


def test_parse_takes_last_answer():
    text = "Answer: A looks tempting, but no.\nAnswer: D\nConfidence: 60"
    assert parse_response(text) == ("D", 0.60)


def test_parse_probability_style_confidence():
    assert parse_response("Answer: A\nConfidence: 0.9") == ("A", 0.9)


def test_parse_clamps_and_handles_missing():
    assert parse_response("Answer: A\nConfidence: 150") == ("A", 1.0)
    assert parse_response("I am not sure.") == (None, None)
    assert parse_response(None) == (None, None)


def test_parse_ignores_answer_inside_words():
    # "the answer is a probability" must not be read as option A
    assert parse_response("The answer is a probability.\nAnswer: B\nConfidence: 50")[0] == "B"


def test_shuffle_is_deterministic_and_correct():
    for q in QUESTIONS:
        choices, letter = shuffled_choices(q)
        assert choices == shuffled_choices(q)[0]
        assert choices["ABCD".index(letter)] == q["choices"][0]
        assert sorted(choices) == sorted(q["choices"])


def test_correct_letters_are_spread_out():
    counts = Counter(shuffled_choices(q)[1] for q in QUESTIONS)
    assert set(counts) == set("ABCD")
    assert max(counts.values()) <= len(QUESTIONS) * 0.4


def test_prompt_contains_options():
    messages, letter = build_messages(QUESTIONS[0])
    assert messages[0]["role"] == "system"
    assert "A) " in messages[1]["content"] and "Confidence:" in messages[1]["content"]
