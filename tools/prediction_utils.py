"""Shared utilities for PeptideBERT prediction tools."""
import os
import json
import re
import torch
import yaml
from transformers import AutoTokenizer, BertTokenizer
from configs.network import create_model


def _split_seqs(text: str):
    return [s.strip() for s in text.replace(",", "\n").splitlines() if s.strip()]


def _sanitize_seq(seq: str):
    return re.sub(r"[UZOB]", "X", seq.strip().upper())


def _maybe_space(seq: str, space_tokens: bool):
    s = seq.replace(" ", "")
    return " ".join(list(s)) if space_tokens else s


def _resolve_hf_snapshot(model_name: str):
    """Return a local HF snapshot path when it exists, otherwise the original id."""
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    repo_dir = os.path.join(cache_root, f"models--{model_name.replace('/', '--')}")
    refs_main = os.path.join(repo_dir, "refs", "main")
    snapshots_dir = os.path.join(repo_dir, "snapshots")

    if os.path.isfile(refs_main):
        with open(refs_main, "r", encoding="utf-8") as f:
            revision = f.read().strip()
        if revision:
            snapshot_dir = os.path.join(snapshots_dir, revision)
            if os.path.isdir(snapshot_dir):
                return snapshot_dir

    if os.path.isdir(snapshots_dir):
        revisions = sorted(os.listdir(snapshots_dir), reverse=True)
        for revision in revisions:
            snapshot_dir = os.path.join(snapshots_dir, revision)
            if os.path.isdir(snapshot_dir):
                return snapshot_dir

    return model_name


def load_predictor(run_name: str, checkpoints_dir: str):
    """Load a PeptideBERT checkpoint and return (model, tokenizer, config)."""
    config_path = os.path.join(checkpoints_dir, run_name, "config.yaml")
    ckpt_path = os.path.join(checkpoints_dir, run_name, "model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["device"] = device

    tokenizer_name = config.get("tokenizer_name") or "Rostlab/prot_bert"
    tokenizer_source = _resolve_hf_snapshot(tokenizer_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
    except ValueError as exc:
        # ProtBERT uses a BERT wordpiece tokenizer; newer transformers releases can
        # incorrectly route AutoTokenizer through a fast-tokenizer conversion path.
        if "backend tokenizer" not in str(exc):
            raise
        tokenizer = BertTokenizer.from_pretrained(tokenizer_source, do_lower_case=False)

    config.setdefault("base_model_name", "Rostlab/prot_bert_bfd")
    config["base_model_name"] = _resolve_hf_snapshot(config["base_model_name"])

    model = create_model(config)

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        state_dict = (
            ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt
        )
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, config, device


@torch.inference_mode()
def run_prediction(model, tokenizer, config, device, seqs, batch_size=32, max_length=256, space_tokens=False):
    """Run batched inference and return a list of prediction dicts."""
    results = []

    for i in range(0, len(seqs), batch_size):
        batch_raw = seqs[i: i + batch_size]
        batch = [_maybe_space(_sanitize_seq(s), space_tokens) for s in batch_raw]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        out = model(input_ids, attention_mask)

        logits = out
        if isinstance(out, (tuple, list)):
            logits = out[0]
        elif isinstance(out, dict) and "logits" in out:
            logits = out["logits"]

        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)

        if logits.size(-1) == 1:
            probs = torch.sigmoid(logits).squeeze(-1).cpu().tolist()
            for s, p in zip(batch_raw, probs):
                results.append({"sequence": s, "p_positive": float(p)})
        else:
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            for s, p in zip(batch_raw, probs):
                pred_idx = int(torch.tensor(p).argmax().item())
                results.append({
                    "sequence": s,
                    "pred_index": pred_idx,
                    "scores": [float(x) for x in p]
                })

    return results


def parse_sequences_input(sequences: str):
    """
    Parse flexible sequence input:
      - JSON string: {"sequences": "..."} or {"sequences": [...]}
      - File path to JSON file
      - Plain sequence text (comma or newline separated)

    Returns (seq_list, payload) or raises ValueError.
    """
    sequences = (sequences or "").strip()
    if not sequences:
        raise ValueError("No input provided.")

    try:
        if os.path.isfile(sequences):
            with open(sequences, "r", encoding="utf-8") as f:
                try:
                    payload = json.load(f)
                except json.JSONDecodeError:
                    payload = {"sequences": f.read().strip()}
        else:
            payload = json.loads(sequences)
    except json.JSONDecodeError:
        payload = {"sequences": sequences}
    except OSError as e:
        raise ValueError(f"Failed to read file: {e}")

    if not isinstance(payload, dict):
        raise ValueError("Input must be JSON object, file path, or plain sequence.")

    seqs = payload.get("sequences") or payload.get("sequence")
    if seqs is None:
        raise ValueError("Missing 'sequences' field.")

    if isinstance(seqs, str) and os.path.isfile(seqs):
        try:
            with open(seqs, "r", encoding="utf-8") as f:
                nested = json.load(f)
            seqs = nested.get("sequences") or nested.get("sequence")
        except Exception:
            with open(seqs, "r", encoding="utf-8") as f:
                seqs = f.read().strip()

    if isinstance(seqs, str):
        seq_list = _split_seqs(seqs)
    elif isinstance(seqs, list):
        seq_list = [str(s).strip() for s in seqs if str(s).strip()]
    else:
        raise ValueError("'sequences' must be string or list.")

    if not seq_list:
        raise ValueError("No valid sequences found.")

    return seq_list, payload
