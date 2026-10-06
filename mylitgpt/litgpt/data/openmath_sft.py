"""OpenMath SFT using the same ChatML prompt as the repository benchmarks."""
from dataclasses import dataclass

from litgpt.data.json_data import JSON
from litgpt.prompts import PromptStyle


class OpenMathPrompt(PromptStyle):
    def apply(self, prompt: str, **kwargs) -> str:
        user = (
            "Solve the following math problem step by step.\n"
            "The last line of your response should be in the format: \\boxed{ANSWER}\n"
            f"Problem: {prompt}"
        )
        return (
            "<|im_start|>system\nYou are a careful mathematical problem solver.<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        )


@dataclass
class OpenMathSFT(JSON):
    def connect(self, tokenizer=None, batch_size=1, max_seq_length=None):
        super().connect(tokenizer, batch_size, max_seq_length)
        # Learn the assistant-turn terminator, rather than the base-model document EOS.
        self.tokenizer.eos_id = self.tokenizer.token_to_id("<|im_end|>")
