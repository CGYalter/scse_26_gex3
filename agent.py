"""RAG agent.

Loads the Qwen3-0.6B model and answers a user's question by first retrieving the
most relevant university IT policy excerpts with the RAG tool, then asking the
model to answer strictly from that retrieved context. The model is instructed to
say "not specified" rather than invent an answer, which is what prevents it from
hallucinating a phone number for the IT Service Desk.

The model path is configurable through the ``QWEN_MODEL_PATH`` environment
variable and defaults to a locally downloaded ``models/qwen3-0.6b`` directory.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_tool import rag_tool

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "models/qwen3-0.6b")

SYSTEM_INSTRUCTION = (
    "You are a university IT support assistant. "
    "Answer using ONLY the university IT policy information provided below. "
    "Never invent any fact. If the requested information is not present in the "
    "provided policy information, say that it is not specified and do not guess."
)


class ITSupportAgent:
    def __init__(self, model_path=MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def answer(self, question):
        context = rag_tool(question)

        prompt = (
            SYSTEM_INSTRUCTION
            + "\n\nRelevant policy information:\n"
            + context
            + "\n\nQuestion: "
            + question
            + "\nAnswer:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # The base model tends to keep echoing the "Answer:" turn; keep only the
        # first line, which is the actual answer.
        first_line = response.splitlines()
        return first_line[0].strip() if first_line else ""


def main():
    agent = ITSupportAgent()

    question = (
        "What exact phone number should I call for the "
        "university IT Service Desk?"
    )

    print("QUESTION:", question)
    print()
    print("ANSWER:", agent.answer(question))


if __name__ == "__main__":
    main()
