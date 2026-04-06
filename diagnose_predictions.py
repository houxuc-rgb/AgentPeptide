#!/usr/bin/env python3
"""
AgentP PeptideBERT Diagnostic Script
=====================================
Run this from the AgentP project directory to diagnose why all predictions
return identical values regardless of input sequence.

Usage:
    cd /home/houxuc/Documents/AgentP_MCP
    python diagnose_predictions.py
"""

import os
import sys
import yaml
import json

sys.path.insert(0, os.path.dirname(__file__))

CHECKPOINTS_DIR = os.environ.get(
    "AGENTPMCP_CHECKPOINTS_DIR",
    "/home/houxuc/Documents/AgentP/checkpoints"
)

MODELS = {
    "solubility": "sol-0114_1359",
    "hemolysis": "hemo-0328_1451",
    "non_fouling": "nf-0210_0959",
}

TEST_SEQS = ["GSAEKRG", "MLLFFWW", "KKKKKKK", "AAAAAA"]


def main():
    print("=" * 60)
    print("  PeptideBERT Diagnostic")
    print("=" * 60)

    # ── Step 1: Check config files ──
    print("\n── Step 1: Checking config.yaml files ──\n")
    for name, run_name in MODELS.items():
        config_path = os.path.join(CHECKPOINTS_DIR, run_name, "config.yaml")
        if not os.path.isfile(config_path):
            print(f"  [{name}] Config NOT FOUND: {config_path}")
            continue

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        space_tokens = config.get("space_tokens", "NOT SET")
        tokenizer_name = config.get("tokenizer_name", "NOT SET")
        base_model = config.get("base_model_name", "NOT SET")
        max_length = config.get("max_length", "NOT SET")
        batch_size = config.get("batch_size", "NOT SET")

        print(f"  [{name}] {run_name}")
        print(f"    space_tokens:    {space_tokens}")
        print(f"    tokenizer_name:  {tokenizer_name}")
        print(f"    base_model_name: {base_model}")
        print(f"    max_length:      {max_length}")
        print(f"    batch_size:      {batch_size}")
        print()

    # ── Step 2: Test tokenization ──
    print("── Step 2: Testing tokenization ──\n")

    try:
        from transformers import AutoTokenizer, BertTokenizer
    except ImportError:
        print("  ERROR: transformers not installed")
        return

    # Load one tokenizer (they should all use the same one)
    first_config_path = os.path.join(CHECKPOINTS_DIR, list(MODELS.values())[0], "config.yaml")
    with open(first_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tokenizer_name = config.get("tokenizer_name") or "Rostlab/prot_bert"
    space_tokens = config.get("space_tokens", False)
    print(f"  Tokenizer: {tokenizer_name}")
    print(f"  space_tokens in config: {space_tokens}")
    print()

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

    test_seq = "GSAEKRG"

    # Test WITHOUT spaces
    print(f"  Without spaces: \"{test_seq}\"")
    tokens_no_space = tokenizer(test_seq, return_tensors="pt")
    ids_no_space = tokens_no_space["input_ids"][0].tolist()
    decoded_no_space = [tokenizer.decode([t]).strip() for t in ids_no_space]
    print(f"    Token IDs: {ids_no_space}")
    print(f"    Decoded:   {decoded_no_space}")
    print(f"    Num tokens: {len(ids_no_space)}")
    print()

    # Test WITH spaces
    spaced = " ".join(list(test_seq))
    print(f"  With spaces: \"{spaced}\"")
    tokens_spaced = tokenizer(spaced, return_tensors="pt")
    ids_spaced = tokens_spaced["input_ids"][0].tolist()
    decoded_spaced = [tokenizer.decode([t]).strip() for t in ids_spaced]
    print(f"    Token IDs: {ids_spaced}")
    print(f"    Decoded:   {decoded_spaced}")
    print(f"    Num tokens: {len(ids_spaced)}")
    print()

    # Check if UNK token is present
    unk_id = tokenizer.unk_token_id
    print(f"  UNK token ID: {unk_id}")
    has_unk_no_space = unk_id in ids_no_space
    has_unk_spaced = unk_id in ids_spaced
    print(f"  UNK in no-space tokens:  {has_unk_no_space}")
    print(f"  UNK in spaced tokens:    {has_unk_spaced}")
    print()

    if has_unk_no_space and not has_unk_spaced:
        print("  *** DIAGNOSIS: Without spaces, the sequence is tokenized as UNK ***")
        print("  *** ProtBERT needs space-separated amino acids: 'G S A E K R G' ***")
        print(f"  *** But space_tokens is set to: {space_tokens} ***")
        if not space_tokens:
            print()
            print("  *** FIX: Set space_tokens: true in each config.yaml ***")
            print(f"  ***   {CHECKPOINTS_DIR}/sol-0114_1359/config.yaml")
            print(f"  ***   {CHECKPOINTS_DIR}/hemo-0209_1531/config.yaml")
            print(f"  ***   {CHECKPOINTS_DIR}/nf-0210_0959/config.yaml")
    elif has_unk_no_space and has_unk_spaced:
        print("  *** Both tokenizations contain UNK — tokenizer may be misconfigured ***")
    else:
        print("  Tokenization looks OK — no UNK tokens.")
        print("  The identical outputs may be caused by something else.")

    # ── Step 3: Test model outputs with both tokenizations ──
    print("\n── Step 3: Testing model outputs ──\n")

    try:
        import torch
        from tools.prediction_utils import load_predictor, run_prediction
    except ImportError as e:
        print(f"  Cannot import model tools: {e}")
        print("  Skipping model output test.")
        return

    # Pick one model to test
    test_model_name = "solubility"
    test_run = MODELS[test_model_name]
    print(f"  Loading {test_model_name} model ({test_run})...")

    model, tokenizer_loaded, config_loaded, device = load_predictor(test_run, CHECKPOINTS_DIR)

    # Test with space_tokens=False (current behavior)
    print(f"\n  Testing with space_tokens=False:")
    results_no_space = run_prediction(
        model, tokenizer_loaded, config_loaded, device,
        TEST_SEQS, batch_size=32, max_length=256, space_tokens=False
    )
    for r in results_no_space:
        p = r.get("p_positive", r.get("pred_index", "?"))
        print(f"    {r['sequence']:12s} → p={p}")

    all_same = len(set(
        r.get("p_positive", r.get("pred_index")) for r in results_no_space
    )) == 1
    if all_same:
        print("    *** All identical — confirms tokenization problem ***")

    # Test with space_tokens=True
    print(f"\n  Testing with space_tokens=True:")
    results_spaced = run_prediction(
        model, tokenizer_loaded, config_loaded, device,
        TEST_SEQS, batch_size=32, max_length=256, space_tokens=True
    )
    for r in results_spaced:
        p = r.get("p_positive", r.get("pred_index", "?"))
        print(f"    {r['sequence']:12s} → p={p}")

    all_same_spaced = len(set(
        r.get("p_positive", r.get("pred_index")) for r in results_spaced
    )) == 1
    if all_same_spaced:
        print("    *** Still identical with spaces — issue may be in the model itself ***")
    else:
        print("    *** Different values! space_tokens=True FIXES the issue ***")
        print()
        print("  ╔══════════════════════════════════════════════════════════╗")
        print("  ║  FIX: Set space_tokens: true in each checkpoint's      ║")
        print("  ║  config.yaml, or pass space_tokens=True in code.       ║")
        print("  ╚══════════════════════════════════════════════════════════╝")

    print("\n" + "=" * 60)
    print("  Diagnostic complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
