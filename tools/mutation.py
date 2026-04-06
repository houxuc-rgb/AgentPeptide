import torch
from esm import pretrained as esm_pretrained

_esm_mutator = None


class FastESM2Mutator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = esm_pretrained.esm2_t6_8M_UR50D()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

    def _tokenize(self, sequence):
        data = [("protein", sequence)]
        _, _, tokens = self.batch_converter(data)
        return tokens.to(self.device)

    def delta_ll_scan(self, sequence):
        tokens = self._tokenize(sequence)

        with torch.no_grad():
            outputs = self.model(tokens)
            logits = outputs["logits"]

        log_probs = torch.log_softmax(logits, dim=-1)

        best_score = float("-inf")
        best_sequence = sequence

        for pos in range(len(sequence)):
            wt_aa = sequence[pos]
            wt_idx = self.alphabet.get_idx(wt_aa)

            for aa in "ACDEFGHIKLMNPQRSTVWY":
                if aa == wt_aa:
                    continue

                mut_idx = self.alphabet.get_idx(aa)

                delta_ll = (
                    log_probs[0, pos + 1, mut_idx]
                    - log_probs[0, pos + 1, wt_idx]
                ).item()

                if delta_ll > best_score:
                    best_score = delta_ll
                    best_sequence = sequence[:pos] + aa + sequence[pos + 1:]

        return best_sequence

def _get_esm_mutator() -> FastESM2Mutator:
    global _esm_mutator
    if _esm_mutator is None:
        _esm_mutator = FastESM2Mutator()
    return _esm_mutator


def esm2_mutate(sequence: str) -> str:
    """
    Perform minimal single-site mutation using ESM2 delta log-likelihood scoring.
    Designed for short peptides (<10 aa). Returns the mutated amino acid sequence.
    """
    if not sequence or len(sequence) < 1:
        raise ValueError("Invalid sequence.")
    return _get_esm_mutator().delta_ll_scan(sequence)
