import json


def result_eval(sol_res: str, hem_res: str, nf_res: str) -> str:
    """
    Summarize peptide property predictions from three evaluation tools.

    Args:
        sol_res: JSON string output from peptidebert_soluability
        hem_res: JSON string output from peptidebert_hemolysis
        nf_res:  JSON string output from peptidebert_non_fouling

    Returns:
        JSON string with unified per-sequence predictions.
    """
    summary = {}

    def update_summary(data_list, property_name):
        for entry in data_list:
            seq = entry["sequence"]
            if seq not in summary:
                summary[seq] = {"sequence": seq}

            p_positive = entry.get("p_positive")
            if p_positive is not None:
                summary[seq][f"{property_name}_p_positive"] = float(p_positive)
            elif entry.get("pred_index") is not None:
                summary[seq][f"{property_name}_pred_index"] = entry.get("pred_index")

    try:
        sol_list = json.loads(sol_res) if isinstance(sol_res, str) else sol_res
        hem_list = json.loads(hem_res) if isinstance(hem_res, str) else hem_res
        nf_list = json.loads(nf_res) if isinstance(nf_res, str) else nf_res
    except json.JSONDecodeError as e:
        return f"Error: Failed to parse input JSON: {e}"

    update_summary(sol_list, "solubility")
    update_summary(hem_list, "hemolysis")
    update_summary(nf_list, "non_fouling")

    for seq_summary in summary.values():
        sol = seq_summary.get("solubility_p_positive")
        hem = seq_summary.get("hemolysis_p_positive")
        nf = seq_summary.get("non_fouling_p_positive")

        if sol is not None:
            seq_summary["is_soluble"] = sol >= 0.5
        if hem is not None:
            seq_summary["is_non_hemolytic"] = hem <= 0.5
            seq_summary["non_hemolysis_score"] = 1.0 - hem
        if nf is not None:
            seq_summary["is_non_fouling"] = nf >= 0.5

        if sol is not None and hem is not None and nf is not None:
            seq_summary["passes_all_constraints"] = (
                (sol >= 0.5) and (hem <= 0.5) and (nf >= 0.5)
            )
            seq_summary["overall_score"] = sol + nf + (1.0 - hem)

    return json.dumps(list(summary.values()), indent=2)
