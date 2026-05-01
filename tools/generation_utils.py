import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_generation_model(model_name: str):
    """Load a causal LM with a dtype that matches the current runtime."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {}

    # Half precision is only safe to assume on CUDA. CPU users need full precision.
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = model.to(_get_device())
    model.eval()
    return model, tokenizer
