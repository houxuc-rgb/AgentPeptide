# AgentPeptide

AgentPeptide is an MCP server and local CLI for peptide design workflows. It combines:

- PeptideGPT generators for non-hemolytic, soluble, and non-fouling sequence proposals
- Local PeptideBERT predictors for hemolysis, solubility, and non-fouling scoring
- Mutation tools for iterative refinement with either an LLM or ESM2
- MCP prompts that guide tool selection and multi-step optimisation

## Requirements

- Python 3.10+
- Local checkpoint files for the PeptideBERT predictors
- `OPENAI_API_KEY` if you want to use `--transport cli`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment

Set the predictor checkpoint directory explicitly before starting the server:

```bash
export AGENTPMCP_CHECKPOINTS_DIR=/path/to/checkpoints
```

The directory should contain subfolders such as:

- `hemo-0328_1451`
- `sol-0114_1359`
- `nf-0210_0959`

If you want to use the interactive CLI refinement loop, also set:

```bash
export OPENAI_API_KEY=...
```

## Running

Start in CLI mode when launched from a terminal:

```bash
python server.py
```

Run as an MCP stdio server:

```bash
python server.py --transport stdio
```

Run as an MCP SSE server:

```bash
python server.py --transport sse --port 8000
```

## Exposed tools

- `generate_non_hemolytic_peptide`
- `generate_soluble_peptide`
- `generate_non_fouling_peptide`
- `predict_hemolysis`
- `predict_solubility`
- `predict_non_fouling`
- `llm_mutate_sequence`
- `esm2_mutate_sequence`
- `evaluate_results`

## Notes

- The predictors expect amino-acid sequences as plain text, newline/comma-separated text, or JSON.
- The generator models are loaded from Hugging Face on first use.
- Checkpoints are not included in this repository.
