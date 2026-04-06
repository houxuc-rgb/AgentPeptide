from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "aayush14/PeptideGPT_non_hemolytic"
_peptide_gpt = None


class PeptideGPT_hemolytic:
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


def _get_peptide_gpt() -> PeptideGPT_hemolytic:
    global _peptide_gpt
    if _peptide_gpt is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        _peptide_gpt = PeptideGPT_hemolytic(model=model, tokenizer=tokenizer)
    return _peptide_gpt


def generate_non_hemolytic_peptide_sequence(prompt: str) -> str:
    """Generate non-hemolytic peptide sequences based on the given prompt."""
    return _get_peptide_gpt().generate(prompt)
