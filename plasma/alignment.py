from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class AlignmentPair:
    baseline_label: Optional[str]
    baseline_idx: Optional[int]
    stream_label: Optional[str]
    stream_idx: Optional[int]
    op: str  # match, delete, insert


def align_sequences_match_delete_insert(
    baseline: List[str],
    stream: List[str],
) -> List[AlignmentPair]:
    """
    Deterministic alignment with only:
    - exact match (cost 0)
    - delete baseline symbol (cost 1)
    - insert stream symbol (cost 1)

    No substitution op is used. A mismatch is forced to become delete+insert.
    This makes a substituted baseline phoneme count as not preserved / omitted.
    """
    n = len(baseline)
    m = len(stream)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "insert"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = []

            if baseline[i - 1] == stream[j - 1]:
                candidates.append((dp[i - 1][j - 1], "match"))

            candidates.append((dp[i - 1][j] + 1, "delete"))
            candidates.append((dp[i][j - 1] + 1, "insert"))

            # Deterministic tie-breaking
            op_priority = {"match": 0, "delete": 1, "insert": 2}
            best_cost, best_op = min(candidates, key=lambda x: (x[0], op_priority[x[1]]))
            dp[i][j] = best_cost
            back[i][j] = best_op

    pairs: List[AlignmentPair] = []
    i, j = n, m

    while i > 0 or j > 0:
        op = back[i][j]

        if op == "match":
            pairs.append(
                AlignmentPair(
                    baseline_label=baseline[i - 1],
                    baseline_idx=i - 1,
                    stream_label=stream[j - 1],
                    stream_idx=j - 1,
                    op="match",
                )
            )
            i -= 1
            j -= 1
        elif op == "delete":
            pairs.append(
                AlignmentPair(
                    baseline_label=baseline[i - 1],
                    baseline_idx=i - 1,
                    stream_label=None,
                    stream_idx=None,
                    op="delete",
                )
            )
            i -= 1
        elif op == "insert":
            pairs.append(
                AlignmentPair(
                    baseline_label=None,
                    baseline_idx=None,
                    stream_label=stream[j - 1],
                    stream_idx=j - 1,
                    op="insert",
                )
            )
            j -= 1
        else:
            raise RuntimeError("Unexpected alignment backpointer state.")

    pairs.reverse()
    return pairs