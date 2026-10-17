"""Builds the exam prompt and reads the model's answer and confidence."""
import random
import re

LETTERS = "ABCD"

SYSTEM_PROMPT = "You are a careful student taking an introductory statistics and probability exam."

INSTRUCTIONS = (
    "Work through the problem briefly. Then end your response with exactly these two lines.\n"
    "Answer: <one letter from A, B, C, D>\n"
    "Confidence: <a number from 0 to 100>\n"
    "Confidence is your probability, as a percentage, that your answer is correct. "
    "Use the full range. 25 means a blind guess among four options, and 100 means certain."
)


def shuffled_choices(question, seed=0):
    """Returns the choices in a fixed random order and the letter of the correct one.

    In the data file the correct choice is always listed first. The order is
    shuffled with a seed tied to the question id, so every model and every
    sample sees the same order, and the correct letter is spread across A to D.
    """
    rng = random.Random(f"{seed}-{question['id']}")
    order = list(range(len(question["choices"])))
    rng.shuffle(order)
    choices = [question["choices"][i] for i in order]
    correct_letter = LETTERS[order.index(0)]
    return choices, correct_letter


def build_messages(question, seed=0):
    choices, correct_letter = shuffled_choices(question, seed)
    options = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
    user = f"{question['question']}\n\n{options}\n\n{INSTRUCTIONS}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    return messages, correct_letter


# "Answer: B", "**Answer:** (B)", "Final answer - B". The last match wins,
# because models sometimes restate the options before their final answer.
_ANSWER = re.compile(r"answer[\s*:_\-]*\(?\s*([ABCD])\b", re.IGNORECASE)
_CONFIDENCE = re.compile(r"confidence[\s*:_\-]*(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)


def parse_response(text):
    """Returns (letter or None, confidence from 0 to 1 or None)."""
    text = text or ""
    answers = _ANSWER.findall(text)
    confidences = _CONFIDENCE.findall(text)
    letter = answers[-1].upper() if answers else None
    confidence = None
    if confidences:
        value = float(confidences[-1])
        if value <= 1 and "." in confidences[-1]:
            value *= 100            # the model wrote a probability such as 0.85
        confidence = min(max(value, 0.0), 100.0) / 100
    return letter, confidence
