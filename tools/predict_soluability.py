import json
import os
from tools.prediction_utils import load_predictor, run_prediction, parse_sequences_input

_CHECKPOINTS_DIR = os.environ.get(
    "AGENTPMCP_CHECKPOINTS_DIR",
    "/home/houxuc/Documents/AgentP/checkpoints"
)
_RUN_NAME = "sol-0114_1359"
_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = load_predictor(_RUN_NAME, _CHECKPOINTS_DIR)
    return _predictor


def peptidebert_soluability(sequences: str) -> str:
    """
    Predict peptide solubility properties using a locally trained
    PeptideBERT-solubility checkpoint.

    Accepts:
      1. JSON string:  {"sequences": "ACDEFGHIK"} or {"sequences": ["ACDEFGHIK", ...]}
      2. Path to JSON file: "response/sequence.json"
      3. Plain sequence text: "ACDEFGHIK"

    Returns JSON string of predictions.
    """
    try:
        seq_list, payload = parse_sequences_input(sequences)
    except ValueError as e:
        return f"Error: {e}"

    _model, _tokenizer, _config, _device = _get_predictor()
    batch_size = int(payload.get("batch_size", _config.get("batch_size", 32)))
    max_length = int(payload.get("max_length", _config.get("max_length", 256)))
    space_tokens = bool(payload.get("space_tokens", _config.get("space_tokens", False)))

    preds = run_prediction(_model, _tokenizer, _config, _device, seq_list,
                           batch_size=batch_size, max_length=max_length,
                           space_tokens=space_tokens)
    return json.dumps(preds, indent=2)
