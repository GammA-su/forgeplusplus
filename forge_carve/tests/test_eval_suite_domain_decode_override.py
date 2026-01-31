from __future__ import annotations

import json
from pathlib import Path

import torch

from fc.dsl.tokens import build_default_vocab
from fc.eval.suite import EvalState, run_eval
from fc.train.data import TextVocab


class _DummyModel:
    def __init__(self, max_prog_len: int, vocab_size: int) -> None:
        self.max_prog_len = max_prog_len
        self.vocab_size = vocab_size

    def __call__(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = input_ids.shape[0]
        logits = torch.zeros((batch, self.max_prog_len, self.vocab_size), dtype=torch.float32)
        return {"logits": logits}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_eval_suite_domain_decode_override(tmp_path: Path, monkeypatch) -> None:
    data_path = tmp_path / "mix.jsonl"
    rows = [
        {
            "id": "schema-1",
            "domain": "schema",
            "domain_tag": "schema",
            "x": "name: Alice age: 30 city: Paris",
            "y": None,
            "constraints": [],
            "proof": {"dsl": "PTv1", "tokens": []},
            "orbit": [],
            "flips": [],
        },
        {
            "id": "math-1",
            "domain": "math",
            "domain_tag": "math",
            "x": "2 + 2",
            "y": None,
            "constraints": [],
            "proof": {"dsl": "PTv1", "tokens": []},
            "orbit": [],
            "flips": [],
        },
        {
            "id": "csp-1",
            "domain": "csp",
            "domain_tag": "csp",
            "x": "schedule tasks A,B with constraints",
            "y": None,
            "constraints": [],
            "proof": {"dsl": "PTv1", "tokens": []},
            "orbit": [],
            "flips": [],
        },
    ]
    _write_jsonl(data_path, rows)

    texts = [row["x"] for row in rows if isinstance(row.get("x"), str)]
    text_vocab = TextVocab.build(texts)
    prog_vocab_obj = build_default_vocab()
    prog_vocab = prog_vocab_obj.token_to_id
    cfg = {"train": {"max_text_len": 16, "max_prog_len": 8}}
    state = EvalState(
        ckpt_path="dummy",
        ckpt={"max_prog_len": 8},
        cfg=cfg,
        text_vocab=text_vocab,
        prog_vocab=prog_vocab,
        prog_vocab_obj=prog_vocab_obj,
        model=_DummyModel(max_prog_len=8, vocab_size=len(prog_vocab)),
        device=torch.device("cpu"),
    )

    calls: list[tuple[str | None, bool]] = []

    def _stub_decode(
        logits_seq: torch.Tensor,
        vocab: object,
        *,
        enforce_opcode: bool,
        min_tokens: int,
        max_tokens: int,
        domain_tag: str | None,
        return_stats: bool,
    ) -> tuple[list[int], None]:
        calls.append((domain_tag, enforce_opcode))
        eos_id = prog_vocab_obj.token_to_id["<EOS>"]
        return [eos_id], None

    monkeypatch.setattr("fc.eval.suite.greedy_decode_with_opcode_mask", _stub_decode)

    run_eval(
        str(data_path),
        "dummy",
        out_path=str(tmp_path / "report.json"),
        constrained_op=True,
        decode_overrides={"math": {"constrained_op": False}},
        max_prog_len=8,
        batch_size=2,
        state=state,
    )

    seen = {domain: flag for domain, flag in calls}
    assert seen.get("math") is False
    assert seen.get("schema") is True
    assert seen.get("csp") is True
