import torch
from tools.generation_utils import load_generation_model

MODEL_NAME = "aayush14/PeptideGPT_soluble"
_peptide_gpt = None


class PeptideGPT_soluble:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def _get_peptide_gpt() -> PeptideGPT_soluble:
    global _peptide_gpt
    if _peptide_gpt is None:
        model, tokenizer = load_generation_model(MODEL_NAME)
        _peptide_gpt = PeptideGPT_soluble(model=model, tokenizer=tokenizer)
    return _peptide_gpt


def generate_soluble_peptide_sequence(prompt: str) -> str:
    """Generate soluble peptide sequences based on the given prompt."""
    return _get_peptide_gpt().generate(prompt)
