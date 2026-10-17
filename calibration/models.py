"""Model clients. TogetherModel calls the Together AI API. SimulatedModel is a
stand-in used for dry runs, so the whole pipeline can be tested for free."""
import os
import random

from .prompts import LETTERS


class TogetherModel:
    def __init__(self, model, max_tokens=1024, thinking=True):
        from dotenv import load_dotenv
        from together import Together

        load_dotenv()
        api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            raise SystemExit("No API key found. Put TOGETHER_API_KEY=... in a .env file "
                             "(see .env.example).")
        # The SDK retries rate limits and server errors with exponential backoff
        self.client = Together(api_key=api_key, max_retries=8, timeout=120)
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking

    def complete(self, messages, temperature, seed=None):
        kwargs = dict(model=self.model, messages=messages,
                      temperature=temperature, max_tokens=self.max_tokens)
        if seed is not None:
            kwargs["seed"] = seed
        if not self.thinking:
            # Hybrid models such as Qwen 3.5 "think" before answering by default.
            # This asks them to answer directly, as the Llama models do.
            kwargs["reasoning"] = {"enabled": False}
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)
        reasoning = (getattr(message, "reasoning", None)
                     or getattr(message, "reasoning_content", None))
        return {
            "text": message.content or "",
            "reasoning": reasoning,
            "finish_reason": getattr(choice, "finish_reason", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }

    def list_chat_models(self):
        return sorted(m.id for m in self.client.models.list()
                      if getattr(m, "type", "chat") == "chat")


class SimulatedModel:
    """Fake model for testing the pipeline. It is right more often on easy
    questions and is deliberately overconfident. Its output means nothing."""

    ACCURACY = {"easy": 0.95, "medium": 0.75, "hard": 0.45}

    def __init__(self, model="simulated", seed=0):
        self.model = model
        self.seed = seed

    def complete(self, messages, temperature, seed=None, question=None, correct_letter=None):
        rng = random.Random(f"{self.seed}-{question['id']}-{seed}-{temperature}")
        skill = self.ACCURACY[question["difficulty"]]
        if temperature == 0:
            rng_answer = random.Random(f"{self.seed}-{question['id']}-greedy")
        else:
            rng_answer = rng
        right = rng_answer.random() < skill
        letter = correct_letter if right else rng_answer.choice(
            [c for c in LETTERS if c != correct_letter])
        confidence = round(min(100, 100 * skill + 20 + rng.gauss(0, 8)))
        text = f"Simulated reasoning.\nAnswer: {letter}\nConfidence: {confidence}"
        return {"text": text, "reasoning": None, "finish_reason": "stop",
                "prompt_tokens": 0, "completion_tokens": 0}
