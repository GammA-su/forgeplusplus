from __future__ import annotations

from prooftape.ptv1 import PTv1Runtime

TOK_CSP = ["<BOS>", "BEGIN",
    "OP", "APPLY_TOPO", "DEST", "STR:order",
    "OP", "APPLY_CUMSUM", "DEST", "STR:schedule",
    "OP", "EMIT_SCHEDULE",
"END", "<EOS>"]

TOK_MATH_ADD = ["<BOS>", "BEGIN",
    "OP", "EXTRACT_INT", "DEST", "STR:a", "ARG", "index", "VAL", "INT:0",
    "OP", "EXTRACT_INT", "DEST", "STR:b", "ARG", "index", "VAL", "INT:1",
    "OP", "APPLY_ARITH", "DEST", "STR:result", "ARG", "a", "VAL", "STR:a", "ARG", "b", "VAL", "STR:b", "ARG", "op", "VAL", "STR:+",
    "OP", "EMIT_NUM", "ARG", "value", "VAL", "STR:result",
"END", "<EOS>"]

def test_csp_apply_topo_cumsum_emit_schedule():
    rec_constraints = [{
        "id": "csp", "type": "csp",
        "args": {"tasks": {"D": 2, "B": 3, "A": 2}, "constraints": [["D","B"], ["B","A"]]}
    }]
    rt = PTv1Runtime()
    got = rt.run("[CSP] Tasks: D=2,B=3,A=2. Constraints: D<B,B<A.", rec_constraints, TOK_CSP)
    assert got == {"schedule": {"D": 0, "B": 2, "A": 5}, "status": "ok"}

def test_math_extract_apply_emit_num():
    rt = PTv1Runtime()
    got = rt.run("[MATH] What is 7 plus 7?", [], TOK_MATH_ADD)
    assert got == 14
