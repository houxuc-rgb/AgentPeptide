"""
AgentP MCP Server
=================
Exposes peptide engineering tools via the Model Context Protocol (MCP).

Tools
-----
- generate_non_hemolytic_peptide_sequence
- generate_soluble_peptide_sequence
- generate_non_fouling_peptide_sequence
- peptidebert_hemolysis
- peptidebert_soluability
- peptidebert_non_fouling
- llm_mutate_sequence  (primary — LLM-powered, property-aware)
- esm2_mutate_sequence (fallback — ESM2 delta log-likelihood)
- result_eval

Prompts
-------
- peptide_generation      — guide for choosing the right generator
- property_prediction     — guide for running all three predictors
- iterative_refinement    — strict multi-step refinement protocol

Usage
-----
    python server.py                    # runs with stdio transport (default)
    python server.py --transport sse    # runs as SSE server on port 8000

Environment variables (all optional, defaults shown)
-----
    AGENTPMCP_CHECKPOINTS_DIR   /home/houxuc/Documents/AgentP/checkpoints
    OPENAI_API_KEY              (required for CLI agent mode and LLM mutation)
    AGENTPMCP_LLM_MODEL         gpt-4o
"""

import sys
import os

# Allow imports relative to this file's directory
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Lazy imports — models are heavy; they load when the server starts
# ---------------------------------------------------------------------------
from tools.generate_non_hemolysis import generate_non_hemolytic_peptide_sequence
from tools.generate_soluability import generate_soluble_peptide_sequence
from tools.generate_non_fouling import generate_non_fouling_peptide_sequence
from tools.predict_hemolysis import peptidebert_hemolysis
from tools.predict_soluability import peptidebert_soluability
from tools.predict_non_fouling import peptidebert_non_fouling
from tools.mutation import esm2_mutate
from tools.structured_result import result_eval

from prompts.peptides_generation import PROMPT as PROMPT_GENERATION
from prompts.property_prediction import PROMPT as PROMPT_PREDICTION
from prompts.refinement import PROMPT as PROMPT_REFINEMENT


import json as _json
import random as _random

# ---------------------------------------------------------------------------
# CLI helpers (minimal — the LLM agent handles all logic)
# ---------------------------------------------------------------------------

# Standard amino acid alphabet for random seeds
_AA_RESIDUES = list("ACDEFGHIKLMNPQRSTVWY")


def _random_seed(length: int = 2) -> str:
    """Generate a short random amino acid seed for PeptideGPT."""
    return "".join(_random.choices(_AA_RESIDUES, k=length))


def _extract_sequence(raw_output: str, seed: str = "") -> str:
    """
    Extract a clean amino-acid sequence from PeptideGPT output.

    PeptideGPT is a causal LM that continues from a seed. The output
    includes the seed followed by the generated continuation. We strip
    non-AA characters and return the full sequence.
    """
    import re
    cleaned = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', raw_output.upper())
    return cleaned if len(cleaned) >= 2 else cleaned


def _run_cli() -> None:
    """
    LLM-agent-driven CLI mode.

    An LLM acts as the agent, deciding which tools to call based on
    the iterative refinement prompt. The CLI handles settings and input
    parsing, then hands control to the agent for the peptide engineering loop.

    Supports two modes:
      - Design from scratch: "design a soluble non-hemolytic peptide"
      - Refine existing:     "refine KRKLLKKILKKI" or just "KRKLLKKILKKI"

    The agent has access to all the same tools as MCP clients:
      - generate_non_hemolytic_peptide, generate_soluble_peptide, generate_non_fouling_peptide
      - predict_hemolysis, predict_solubility, predict_non_fouling
      - llm_mutate_sequence (primary), esm2_mutate_sequence (fallback)
      - evaluate_results
    """
    import re
    import json

    # ── Check for API key ──
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is required for agent mode.")
        print("  export OPENAI_API_KEY=sk-...")
        return

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed.")
        print("  pip install openai")
        return

    print()
    print("Describe the peptide properties you want, or provide a sequence to refine.")
    print("The LLM agent will decide which tools to call and how to iterate.")
    print()
    print("Commands:")
    print("  set model ..")
    print("  refine KRKLLKKILKKI          — refine a raw peptide sequence")
    print("  refine KRKLLKKILKKI soluble   — refine targeting specific properties")
    print("  Then any natural language request, e.g.:")
    print("    design a soluble non-hemolytic peptide")
    print()
    # ── Define tools in OpenAI function-calling format ──
    agent_tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_non_hemolytic_peptide",
                "description": "Generate a non-hemolytic peptide sequence using PeptideGPT. Pass a short 1-3 amino acid seed (e.g. 'KR', 'GS'). Do NOT pass natural language.",
                "parameters": {
                    "type": "object",
                    "properties": {"seed": {"type": "string", "description": "1-3 amino acid seed, e.g. 'KR'"}},
                    "required": ["seed"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_soluble_peptide",
                "description": "Generate a soluble peptide sequence using PeptideGPT. Pass a short 1-3 amino acid seed (e.g. 'KE', 'DS'). Do NOT pass natural language.",
                "parameters": {
                    "type": "object",
                    "properties": {"seed": {"type": "string", "description": "1-3 amino acid seed, e.g. 'KE'"}},
                    "required": ["seed"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_non_fouling_peptide",
                "description": "Generate a non-fouling peptide sequence using PeptideGPT. Pass a short 1-3 amino acid seed (e.g. 'SG', 'EK'). Do NOT pass natural language.",
                "parameters": {
                    "type": "object",
                    "properties": {"seed": {"type": "string", "description": "1-3 amino acid seed, e.g. 'SG'"}},
                    "required": ["seed"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "predict_hemolysis",
                "description": "Predict hemolysis probability for peptide sequence(s). Returns p_positive where p<0.5 means non-hemolytic (good). Pass sequences as JSON with space_tokens: true, e.g. '{\"sequences\": \"GSAEKRG\", \"space_tokens\": true}'",
                "parameters": {
                    "type": "object",
                    "properties": {"sequences": {"type": "string", "description": "JSON string with sequences and space_tokens: true"}},
                    "required": ["sequences"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "predict_solubility",
                "description": "Predict solubility for peptide sequence(s). Returns p_positive where p>=0.5 means soluble (good). Pass sequences as JSON with space_tokens: true.",
                "parameters": {
                    "type": "object",
                    "properties": {"sequences": {"type": "string", "description": "JSON string with sequences and space_tokens: true"}},
                    "required": ["sequences"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "predict_non_fouling",
                "description": "Predict non-fouling properties for peptide sequence(s). Returns p_positive where p>=0.5 means non-fouling (good). Pass sequences as JSON with space_tokens: true.",
                "parameters": {
                    "type": "object",
                    "properties": {"sequences": {"type": "string", "description": "JSON string with sequences and space_tokens: true"}},
                    "required": ["sequences"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "llm_mutate_sequence",
                "description": (
                    "Use an LLM to propose a single targeted amino-acid substitution. "
                    "Unlike ESM2, this considers the property scores and knows WHY a "
                    "residue should change. This is the PRIMARY mutation tool — use it "
                    "first. Pass the sequence (no spaces), current property scores, and "
                    "which property to improve."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sequence": {
                            "type": "string",
                            "description": "Amino acid sequence without spaces, e.g. 'GSAEKRG'",
                        },
                        "hemolysis_score": {
                            "type": "number",
                            "description": "Current hemolysis p_positive (lower is better, want < 0.5)",
                        },
                        "solubility_score": {
                            "type": "number",
                            "description": "Current solubility p_positive (higher is better, want >= 0.5)",
                        },
                        "non_fouling_score": {
                            "type": "number",
                            "description": "Current non-fouling p_positive (higher is better, want >= 0.5)",
                        },
                        "target_property": {
                            "type": "string",
                            "description": "Which property to prioritise improving: 'hemolysis', 'solubility', or 'non_fouling'",
                            "enum": ["hemolysis", "solubility", "non_fouling"],
                        },
                    },
                    "required": ["sequence", "hemolysis_score", "solubility_score", "non_fouling_score", "target_property"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "esm2_mutate_sequence",
                "description": (
                    "Fallback mutation tool using ESM2 delta log-likelihood scoring. "
                    "Use ONLY if llm_mutate_sequence fails or returns an invalid sequence. "
                    "This is a blind mutator — it does not consider property scores."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"sequence": {"type": "string", "description": "Amino acid sequence without spaces, e.g. 'GSAEKRG'"}},
                    "required": ["sequence"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "evaluate_results",
                "description": "Combine predictions from the three property tools into a unified summary per sequence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sol_res": {"type": "string", "description": "JSON from predict_solubility"},
                        "hem_res": {"type": "string", "description": "JSON from predict_hemolysis"},
                        "nf_res": {"type": "string", "description": "JSON from predict_non_fouling"},
                    },
                    "required": ["sol_res", "hem_res", "nf_res"],
                },
            },
        },
    ]

    # ── LLM-powered mutation function ──
    def _llm_mutate(args: dict) -> str:
        """
        Use the LLM to propose a single targeted amino-acid substitution
        based on the current property scores and the target property.
        """
        import json as _j
        seq = args["sequence"].replace(" ", "")
        hem = args["hemolysis_score"]
        sol = args["solubility_score"]
        nf  = args["non_fouling_score"]
        target = args["target_property"]

        mutation_prompt = f"""You are an expert peptide chemist. Given a peptide sequence and its
current property scores, propose exactly ONE single amino-acid substitution
to improve the target property while preserving the others as much as possible.

Sequence: {seq}
Current scores:
  - Hemolysis p_positive: {hem:.4f}  (want < 0.5; lower = less hemolytic)
  - Solubility p_positive: {sol:.4f}  (want >= 0.5; higher = more soluble)
  - Non-fouling p_positive: {nf:.4f}  (want >= 0.5; higher = more non-fouling)

Target property to improve: {target}

Rules:
- Change EXACTLY one residue (substitution only, no insertions or deletions).
- The output sequence must be the same length as the input ({len(seq)} residues).
- Use only standard amino acids: A C D E F G H I K L M N P Q R S T V W Y.
- Think about which residue is most likely hurting the target property
  and what substitution would help:
  · To REDUCE hemolysis: replace hydrophobic / cationic-amphipathic residues
    (L, I, V, F, W, K in hydrophobic face) with polar/neutral ones (S, T, G, N, Q).
  · To INCREASE solubility: replace hydrophobic residues (L, I, V, F, W, A)
    with charged/polar ones (K, R, E, D, S, T, N, Q).
  · To INCREASE non-fouling: replace sticky/hydrophobic residues with
    hydrophilic/zwitterionic ones (S, G, E, K, D).

Respond with ONLY a JSON object, nothing else:
{{"mutated_sequence": "<new sequence>", "position": <0-indexed>, "original_aa": "<old>", "new_aa": "<new>", "rationale": "<brief reason>"}}"""

        try:
            resp = client.chat.completions.create(
                model=model_name,
                max_tokens=300,
                temperature=0.3,
                messages=[{"role": "user", "content": mutation_prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = _j.loads(raw)
            mutated = result.get("mutated_sequence", "").replace(" ", "").upper()

            # Validate
            import re as _re
            if not _re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]+', mutated):
                return f"Error: LLM returned invalid sequence '{mutated}'. Use esm2_mutate_sequence as fallback."
            if len(mutated) != len(seq):
                return f"Error: LLM changed sequence length ({len(seq)} → {len(mutated)}). Use esm2_mutate_sequence as fallback."

            rationale = result.get("rationale", "no rationale given")
            pos = result.get("position", "?")
            old_aa = result.get("original_aa", "?")
            new_aa = result.get("new_aa", "?")
            return f"{mutated} (pos {pos}: {old_aa}→{new_aa}, reason: {rationale})"

        except Exception as e:
            return f"Error: LLM mutation failed ({e}). Use esm2_mutate_sequence as fallback."

    # ── Map tool names to local functions ──
    tool_dispatch = {
        "generate_non_hemolytic_peptide": lambda args: generate_non_hemolytic_peptide(args.get("seed", "")),
        "generate_soluble_peptide":       lambda args: generate_soluble_peptide(args.get("seed", "")),
        "generate_non_fouling_peptide":   lambda args: generate_non_fouling_peptide(args.get("seed", "")),
        "predict_hemolysis":    lambda args: peptidebert_hemolysis(args["sequences"]),
        "predict_solubility":   lambda args: peptidebert_soluability(args["sequences"]),
        "predict_non_fouling":  lambda args: peptidebert_non_fouling(args["sequences"]),
        "llm_mutate_sequence":  _llm_mutate,
        "esm2_mutate_sequence": lambda args: esm2_mutate(args["sequence"].replace(" ", "")),
        "evaluate_results":     lambda args: result_eval(args["sol_res"], args["hem_res"], args["nf_res"]),
    }

    # ── System prompt: strict iterative refinement protocol ──
    system_prompt = f"""You are AgentP, an autonomous peptide engineering agent. You MUST
act by calling tools — never just describe what you would do. Every
response from you MUST contain at least one tool call until you have a
final accepted candidate. Text-only replies that say "I will now …"
without a tool call are FORBIDDEN.

## TOOL USAGE RULES
- PeptideGPT generators expect a short amino acid seed (1-3 uppercase
  letters like "KR", "GS"), NOT natural language.
  · Non-hemolytic: hydrophilic seeds — K, R, E, D, S, T, G
  · Soluble: charged / polar seeds — K, E, D, S
  · Non-fouling: hydrophilic seeds — S, G, E, K
- For property predictions, ALWAYS pass sequences as JSON with
  space_tokens: true, e.g.
  '{{"sequences": "GSAEKRG", "space_tokens": true}}'.
  Without space_tokens ProtBERT tokenizes incorrectly.

## SCORING CRITERIA
- Hemolysis:   p_positive < 0.5 → non-hemolytic (GOOD)
- Solubility:  p_positive ≥ 0.5 → soluble        (GOOD)
- Non-fouling: p_positive ≥ 0.5 → non-fouling     (GOOD)
A candidate PASSES when ALL three criteria are met simultaneously.

## ITERATIVE REFINEMENT PROTOCOL (follow exactly)

### Phase 0 — Refine existing sequence (REFINE-EXISTING mode)
If the user provides a starting peptide sequence to refine:
1. SKIP all generators. Use the provided sequence as-is.
2. Immediately predict ALL THREE properties on the provided sequence.
3. Call evaluate_results to get a unified baseline summary.
4. Record the baseline scores. If it already passes all three → report
   success and stop. Otherwise note which properties fail.
5. Proceed directly to Phase 2 (mutation refinement) using this sequence
   as the starting best candidate.
6. If mutation stalls (2 consecutive rounds with no improvement on the
   worst property), THEN and ONLY THEN use a generator targeting the
   worst property to get a fresh candidate, and continue Phase 2 with it.
   But always compare back against the user's original sequence scores.

### Phase 1 — Seed generation (up to 3 candidates)
1. Call a generator (choose based on the user's priority, or default to
   generate_non_hemolytic_peptide).
2. Immediately predict ALL THREE properties on the generated sequence.
3. Call evaluate_results to get a unified summary.
4. Record the scores. If the candidate passes all three → ACCEPT, go to
   Phase 3. Otherwise note which properties failed.
5. Repeat steps 1-4 up to 3 times with different seeds, picking the
   generator that targets the worst-failing property each time.

### Phase 2 — Mutation refinement (up to 5 rounds per candidate)
Take the best candidate from Phase 1 (or Phase 0).
1. Call llm_mutate_sequence on the current best sequence. You MUST pass
   the current hemolysis_score, solubility_score, non_fouling_score, and
   set target_property to the worst-failing property.
   - If llm_mutate_sequence returns an error, call esm2_mutate_sequence
     as a fallback (it only needs the sequence).
2. Extract the mutated sequence from the output (strip the rationale
   text — use only the amino-acid string before the parenthesis).
3. Immediately predict ALL THREE properties on the mutated sequence.
4. Call evaluate_results.
5. Compare to the previous best:
   - If strictly better (more properties pass, or same passes with better
     scores on the failing property) → adopt as new best.
   - If worse or no improvement → discard, keep previous best.
6. If the candidate now passes all three → ACCEPT, go to Phase 3.
7. If 2 consecutive mutations show no improvement on the worst property,
   STOP mutating. Instead go back to Phase 1 and generate a fresh
   candidate using the generator that targets the worst-failing property.
8. Repeat up to 5 mutation rounds per candidate, and up to 3 full
   generate-then-mutate cycles total.

### Phase 3 — Final report
Print a summary with:
- Best sequence found
- All three property scores
- Pass/fail status for each
- How many generation + mutation steps were used

## CRITICAL BEHAVIORAL RULES
- NEVER stop to narrate without calling a tool. If you want to explain
  your reasoning, include the explanation AND a tool call in the same
  response.
- After every mutation call (llm_mutate_sequence or esm2_mutate_sequence),
  you MUST immediately call the three predict_* tools. Never skip
  prediction after mutation.
- After every round of three predictions, you MUST call evaluate_results.
- Keep a running "best candidate" with its scores. Always compare new
  candidates against it.
- If you are unsure what to do next, call the generator targeting the
  worst-failing property.
- For mutation tools, pass the sequence WITHOUT spaces (e.g. "GSKWQ",
  not "G S K W Q"). Spaces are for PeptideBERT only.
- ALWAYS use llm_mutate_sequence first. Only fall back to
  esm2_mutate_sequence if the LLM mutation returns an error.
- When llm_mutate_sequence returns output like "GSEWQ (pos 2: K→E,
  reason: ...)", extract ONLY the sequence part (GSEWQ) for prediction.
- If a tool call returns an error, DO NOT give up. Try again with
  corrected input, or switch strategy (e.g. regenerate instead of
  mutate). Errors are expected — recover and continue.
- NEVER end with a "could not complete" message if you still have
  generation or mutation attempts remaining.

{PROMPT_REFINEMENT}

{PROMPT_PREDICTION}

{PROMPT_GENERATION}
"""

    client = OpenAI(api_key=api_key)
    model_name = os.environ.get("AGENTPMCP_LLM_MODEL", "gpt-4o")

    print(f"  LLM model: {model_name}")

    while True:
        prompt_tag = "[agentp]> "
        try:
            user_input = input(prompt_tag).strip()
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print("\nExiting.")
            return

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            return

        # ── Setting commands (handled locally, not by the agent) ──
        set_match = re.match(r'^set\s+(\w+)\s+(.+)$', user_input, re.IGNORECASE)
        if set_match:
            key, val = set_match.group(1).lower(), set_match.group(2).strip()
            if key == "model":
                model_name = val.strip()
                print(f"  LLM model → {model_name}")
            else:
                print("  In agent mode, use 'set model'.")
                print("  Other settings can be communicated to the agent in natural language.")
            continue

        # ── Refine command: "refine SEQUENCE [goals]" ──
        refine_match = re.match(
            r'^refine\s+([A-Z]{3,})\s*(.*)?$', user_input, re.IGNORECASE
        )
        if refine_match:
            raw_seq = refine_match.group(1).upper()
            goals = refine_match.group(2).strip() if refine_match.group(2) else ""
            # Validate: only standard amino acids
            if not re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]+', raw_seq):
                print("  Invalid sequence. Use only standard amino acid letters (ACDEFGHIKLMNPQRSTVWY).")
                continue
            print(f"  Starting refinement of: {raw_seq} ({len(raw_seq)} aa)")
            if goals:
                print(f"  Goals: {goals}")
            goal_text = goals if goals else "all three properties (soluble, non-hemolytic, non-fouling)"
            user_message = (
                f"I have an existing peptide sequence that I want you to refine and improve.\n"
                f"\n"
                f"Starting sequence: {raw_seq}\n"
                f"Optimisation goals: {goal_text}\n"
                f"\n"
                f"Follow the REFINE-EXISTING protocol: predict its current properties first, "
                f"then iteratively mutate to improve the failing ones. Do NOT generate a new "
                f"sequence from scratch — start from this exact sequence."
            )
        # ── Auto-detect raw sequence (all uppercase AA letters, 3+ chars) ──
        elif re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]{3,}', user_input.strip().upper()) and user_input.strip().isalpha():
            raw_seq = user_input.strip().upper()
            print(f"  Detected raw peptide sequence: {raw_seq} ({len(raw_seq)} aa)")
            print(f"  Starting refinement (use 'refine {raw_seq} <goals>' for specific targets)")
            user_message = (
                f"I have an existing peptide sequence that I want you to refine and improve.\n"
                f"\n"
                f"Starting sequence: {raw_seq}\n"
                f"Optimisation goals: all three properties (soluble, non-hemolytic, non-fouling)\n"
                f"\n"
                f"Follow the REFINE-EXISTING protocol: predict its current properties first, "
                f"then iteratively mutate to improve the failing ones. Do NOT generate a new "
                f"sequence from scratch — start from this exact sequence."
            )
        else:
            user_message = user_input

        print(f"\n{'─'*60}")
        print("  Agent starting iterative refinement...")
        print(f"{'─'*60}\n")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        max_agent_turns = 50  # more headroom for iterative cycles
        stall_count = 0       # track consecutive text-only responses
        max_stalls = 3        # give up after this many nudges

        for turn in range(max_agent_turns):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    max_tokens=4096,
                    tools=agent_tools,
                    messages=messages,
                )
            except Exception as e:
                print(f"\n  API Error: {e}")
                break

            choice = response.choices[0]
            message = choice.message

            if message.content:
                print(f"\n  Agent: {message.content}")

            # ── Handle text-only responses (no tool calls) ──
            if choice.finish_reason == "stop" or not message.tool_calls:
                # Check if the agent is narrating a plan without acting.
                # Look for intent signals: "I will", "I'll", "let me",
                # "next step", etc. — these mean it stalled.
                content_lower = (message.content or "").lower()
                planning_signals = [
                    "i will", "i'll", "let me", "let's", "next step",
                    "now i", "going to", "attempt", "try to", "apply",
                    "let us",
                ]
                is_stalled = any(s in content_lower for s in planning_signals)

                if is_stalled and stall_count < max_stalls:
                    stall_count += 1
                    print(f"\n  [system] Agent stalled (narrated without acting). "
                          f"Nudging... ({stall_count}/{max_stalls})")
                    messages.append(message)
                    messages.append({
                        "role": "user",
                        "content": (
                            "You just described what you plan to do but did not "
                            "call any tools. This is not allowed. You MUST call "
                            "the tool NOW. Do not explain — just make the tool "
                            "call in your next response."
                        ),
                    })
                    continue
                else:
                    # Genuine final answer or max stalls reached
                    break

            # ── Agent made tool calls — reset stall counter ──
            stall_count = 0
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_input = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                print(f"\n  ┌─ Tool: {tool_name}")
                print(f"  │  Input: {json.dumps(tool_input)[:200]}")

                if tool_name in tool_dispatch:
                    try:
                        result = tool_dispatch[tool_name](tool_input)
                        result_str = str(result)
                        print(f"  │  Output: {result_str[:300]}")
                        print("  └─ Done")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        })
                    except Exception as e:
                        error_msg = f"Error: {e}"
                        print(f"  │  {error_msg}")
                        print("  └─ Failed")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_msg,
                        })
                else:
                    error_msg = f"Unknown tool: {tool_name}"
                    print(f"  └─ {error_msg}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg,
                    })
        else:
            print(f"\n  Agent reached max turns ({max_agent_turns}). Stopping.")

        print(f"\n{'─'*60}\n")


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "AgentP",
    instructions=(
        "You are AgentP, an AI peptide engineering assistant. "
        "Use the available tools to generate, predict, mutate, and evaluate peptides. "
        "Always prefer tool calls over speculation."
    ),
)

# ---------------------------------------------------------------------------
# Generation tools
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_non_hemolytic_peptide(seed: str = "") -> str:
    """
    Generate non-hemolytic peptide sequences using PeptideGPT (aayush14/PeptideGPT_non_hemolytic).

    Args:
        seed: Optional 1-3 amino acid seed to prime generation (e.g. "KR").
              If empty, a hydrophilic-biased seed is used. Do NOT pass natural language.

    Returns:
        Generated peptide sequence text.
    """
    if not seed or not seed.isalpha() or not seed.isupper():
        seed = _random_seed(2, "non-hemolytic")
    raw = generate_non_hemolytic_peptide_sequence(seed)
    return _extract_sequence(raw, seed)


@mcp.tool()
def generate_soluble_peptide(seed: str = "") -> str:
    """
    Generate soluble peptide sequences using PeptideGPT (aayush14/PeptideGPT_soluble).

    Args:
        seed: Optional 1-3 amino acid seed to prime generation (e.g. "KE").
              If empty, a polar-biased seed is used. Do NOT pass natural language.

    Returns:
        Generated peptide sequence text.
    """
    if not seed or not seed.isalpha() or not seed.isupper():
        seed = _random_seed(2, "soluble")
    raw = generate_soluble_peptide_sequence(seed)
    return _extract_sequence(raw, seed)


@mcp.tool()
def generate_non_fouling_peptide(seed: str = "") -> str:
    """
    Generate non-fouling (anti-biofouling) peptide sequences using PeptideGPT
    (aayush14/PeptideGPT_non_fouling).

    Args:
        seed: Optional 1-3 amino acid seed to prime generation (e.g. "SG").
              If empty, a hydrophilic-biased seed is used. Do NOT pass natural language.

    Returns:
        Generated peptide sequence text.
    """
    if not seed or not seed.isalpha() or not seed.isupper():
        seed = _random_seed(2, "non-fouling")
    raw = generate_non_fouling_peptide_sequence(seed)
    return _extract_sequence(raw, seed)


# ---------------------------------------------------------------------------
# Prediction tools
# ---------------------------------------------------------------------------

@mcp.tool()
def predict_hemolysis(sequences: str) -> str:
    """
    Predict hemolysis probability for one or more peptide sequences using a
    locally trained PeptideBERT checkpoint (hemo-0209_1531).

    Args:
        sequences: One of:
            - Plain amino-acid string:  "ACDEFGHIK"
            - Comma/newline separated:  "ACDEFGHIK,MKWVTF"
            - JSON object:              '{"sequences": ["ACDEFGHIK", "MKWVTF"]}'
            - Path to a JSON file

    Returns:
        JSON array of predictions, each with "sequence" and "pred_index"/"scores".
    """
    return peptidebert_hemolysis(sequences)


@mcp.tool()
def predict_solubility(sequences: str) -> str:
    """
    Predict solubility for one or more peptide sequences using a locally trained
    PeptideBERT checkpoint (sol-0114_1359).

    Args:
        sequences: One of:
            - Plain amino-acid string:  "ACDEFGHIK"
            - Comma/newline separated:  "ACDEFGHIK,MKWVTF"
            - JSON object:              '{"sequences": ["ACDEFGHIK", "MKWVTF"]}'
            - Path to a JSON file

    Returns:
        JSON array of predictions, each with "sequence" and "pred_index"/"scores".
    """
    return peptidebert_soluability(sequences)


@mcp.tool()
def predict_non_fouling(sequences: str) -> str:
    """
    Predict non-fouling properties for one or more peptide sequences using a locally
    trained PeptideBERT checkpoint (nf-0210_0959).

    Args:
        sequences: One of:
            - Plain amino-acid string:  "ACDEFGHIK"
            - Comma/newline separated:  "ACDEFGHIK,MKWVTF"
            - JSON object:              '{"sequences": ["ACDEFGHIK", "MKWVTF"]}'
            - Path to a JSON file

    Returns:
        JSON array of predictions, each with "sequence" and "pred_index"/"scores".
    """
    return peptidebert_non_fouling(sequences)


# ---------------------------------------------------------------------------
# Refinement / mutation tools
# ---------------------------------------------------------------------------

@mcp.tool()
def llm_mutate_sequence(
    sequence: str,
    hemolysis_score: float,
    solubility_score: float,
    non_fouling_score: float,
    target_property: str,
) -> str:
    """
    Use an LLM to propose a single targeted amino-acid substitution.
    Unlike ESM2, this considers the current property scores and reasons about
    which residue to change and why. This is the PRIMARY mutation tool.

    Args:
        sequence:          Amino-acid sequence (no spaces), e.g. "GSAEKRG".
        hemolysis_score:   Current hemolysis p_positive (want < 0.5).
        solubility_score:  Current solubility p_positive (want >= 0.5).
        non_fouling_score: Current non-fouling p_positive (want >= 0.5).
        target_property:   Which property to prioritise: "hemolysis",
                           "solubility", or "non_fouling".

    Returns:
        Mutated sequence with rationale, e.g.
        "GSAERKG (pos 3: K→E, reason: reduce cationic charge to lower hemolysis)"
    """
    import json as _j, re as _re

    seq = sequence.replace(" ", "").upper()
    import shutil

    mutation_prompt = f"""You are an expert peptide chemist. Given a peptide sequence and its
current property scores, propose exactly ONE single amino-acid substitution
to improve the target property while preserving the others as much as possible.

Sequence: {seq}
Current scores:
  - Hemolysis p_positive: {hemolysis_score:.4f}  (want < 0.5; lower = less hemolytic)
  - Solubility p_positive: {solubility_score:.4f}  (want >= 0.5; higher = more soluble)
  - Non-fouling p_positive: {non_fouling_score:.4f}  (want >= 0.5; higher = more non-fouling)

Target property to improve: {target_property}

Rules:
- Change EXACTLY one residue (substitution only, no insertions or deletions).
- The output sequence must be the same length as the input ({len(seq)} residues).
- Use only standard amino acids: A C D E F G H I K L M N P Q R S T V W Y.
- Think about which residue is most likely hurting the target property
  and what substitution would help:
  · To REDUCE hemolysis: replace hydrophobic / cationic-amphipathic residues
    (L, I, V, F, W, K in hydrophobic face) with polar/neutral ones (S, T, G, N, Q).
  · To INCREASE solubility: replace hydrophobic residues (L, I, V, F, W, A)
    with charged/polar ones (K, R, E, D, S, T, N, Q).
  · To INCREASE non-fouling: replace sticky/hydrophobic residues with
    hydrophilic/zwitterionic ones (S, G, E, K, D).

Respond with ONLY a JSON object, nothing else:
{{"mutated_sequence": "<new sequence>", "position": <0-indexed>, "original_aa": "<old>", "new_aa": "<new>", "rationale": "<brief reason>"}}"""

    try:
        import subprocess
        claude_bin = shutil.which("claude") or "claude"
        proc = subprocess.run(
            [claude_bin, "-p", mutation_prompt],
            capture_output=True, text=True, timeout=60
        )
        raw = proc.stdout.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = _j.loads(raw)
        mutated = result.get("mutated_sequence", "").replace(" ", "").upper()

        if not _re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]+', mutated):
            return f"Error: LLM returned invalid sequence '{mutated}'. Use esm2_mutate_sequence as fallback."
        if len(mutated) != len(seq):
            return f"Error: LLM changed sequence length ({len(seq)} → {len(mutated)}). Use esm2_mutate_sequence as fallback."

        rationale = result.get("rationale", "no rationale given")
        pos = result.get("position", "?")
        old_aa = result.get("original_aa", "?")
        new_aa = result.get("new_aa", "?")
        return f"{mutated} (pos {pos}: {old_aa}→{new_aa}, reason: {rationale})"

    except Exception as e:
        return f"Error: LLM mutation failed ({e}). Use esm2_mutate_sequence as fallback."


@mcp.tool()
def esm2_mutate_sequence(sequence: str) -> str:
    """
    Fallback mutation tool: perform a single amino-acid substitution using
    ESM2 delta log-likelihood scoring (esm2_t6_8M_UR50D). This is a blind
    mutator — it does NOT consider property scores. Use only if
    llm_mutate_sequence fails.

    Args:
        sequence: Amino-acid sequence in single-letter code, e.g. "GSAEKRG".
                  Spaces are stripped automatically.

    Returns:
        Mutated amino-acid sequence (one residue changed).
    """
    clean = sequence.replace(" ", "")
    return esm2_mutate(clean)


# ---------------------------------------------------------------------------
# Result aggregation tool
# ---------------------------------------------------------------------------

@mcp.tool()
def evaluate_results(sol_res: str, hem_res: str, nf_res: str) -> str:
    """
    Combine predictions from the three property tools into a single unified
    summary per sequence.

    Args:
        sol_res: JSON string returned by predict_solubility.
        hem_res: JSON string returned by predict_hemolysis.
        nf_res:  JSON string returned by predict_non_fouling.

    Returns:
        JSON array, one object per sequence, with keys:
        sequence, solubility_pred, hemolysis_pred, non_fouling_pred.
    """
    return result_eval(sol_res, hem_res, nf_res)


# ---------------------------------------------------------------------------
# MCP Prompts — content lives in prompts/ directory
# ---------------------------------------------------------------------------

@mcp.prompt()
def peptide_generation() -> str:
    """Guide for selecting and invoking the correct peptide generation tool."""
    return PROMPT_GENERATION


@mcp.prompt()
def property_prediction() -> str:
    """Guide for running all three property predictors on a peptide sequence."""
    return PROMPT_PREDICTION


@mcp.prompt()
def iterative_refinement() -> str:
    """Strict multi-step iterative refinement protocol for peptide optimisation."""
    return PROMPT_REFINEMENT


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AgentP MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "cli"],
        default=None,
        help="Transport type (default: cli in a terminal, otherwise stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    args = parser.parse_args()

    transport = args.transport
    if transport is None:
        transport = "cli" if sys.stdin.isatty() and sys.stdout.isatty() else "stdio"

    if transport == "cli":
        _run_cli()
    elif transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")
