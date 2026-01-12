from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from fc.adv.mutator import ProgramMutator
from fc.dsl.codec import alignment_distance, decode_program
from fc.dsl.program import Program
from fc.eval.metrics import (
    attack_success_rate,
    flip_sensitivity_score,
    orbit_invariance_pass_rate,
    proof_validity_correlation,
    repair_success_rate,
    selective_accuracy,
    verified_accuracy,
)
from fc.interp.core import Interpreter
from fc.morph.equiv import outputs_equivalent
from fc.dsl.tokens import TokenVocab
from fc.train.data import Example, TextVocab
from fc.util.jsonl import read_jsonl
from fc.verify.mesh import VerifierMesh
from fc.dsl.repair import propose_repairs
from fc.train.losses import kl_divergence


def _load_text_vocab(mapping: dict[str, int]) -> TextVocab:
    id_to_token = {i: t for t, i in mapping.items()}
    return TextVocab(token_to_id=mapping, id_to_token=id_to_token)

def _load_prog_vocab(mapping: dict[str, int]) -> TokenVocab:
    id_to_token = {i: t for t, i in mapping.items()}
    return TokenVocab(token_to_id=mapping, id_to_token=id_to_token)


def run_eval(
    data_path: str,
    ckpt_path: str,
    out_path: str,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    rows = [Example.model_validate(r) for r in read_jsonl(data_path)]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    text_vocab = _load_text_vocab(ckpt["text_vocab"])
    prog_vocab = ckpt["prog_vocab"]
    prog_vocab_obj = _load_prog_vocab(prog_vocab)
    cfg = ckpt["config"]

    from fc.model.forge import ForgeModel, ModelConfig
    from fc.model.backbone import BackboneConfig
    from fc.model.primal_dual import PrimalDualConfig
    from fc.model.slots import SlotConfig

    bcfg = BackboneConfig(vocab_size=cfg["text_vocab_size"], **cfg["backbone"])
    scfg = SlotConfig(**cfg["slots"])
    pcfg = PrimalDualConfig(**cfg["primal_dual"])
    mcfg = ModelConfig(
        vocab_size=len(prog_vocab),
        max_prog_len=cfg["max_prog_len"],
        backbone=bcfg,
        slots=scfg,
        primal_dual=pcfg,
    )
    model = ForgeModel(mcfg)
    model.load_state_dict(ckpt["model"])
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = str(device)
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
    torch_device = torch.device(device_str)
    model.to(torch_device)
    model.eval()

    interp = Interpreter()
    mesh = VerifierMesh()
    mutator = ProgramMutator()

    correct = []
    valid = []
    orbit_pass = []
    flip_pass = []
    repaired = []
    confs = []
    adv_success = 0
    adv_total = 0

    for ex in rows:
        input_ids = torch.tensor(
            [text_vocab.encode(ex.x, cfg["train"]["max_text_len"])],
            dtype=torch.long,
            device=torch_device,
        )
        with torch.inference_mode():
            base_out = model(input_ids)
        pred_ids = torch.argmax(base_out["logits"], dim=-1)[0].detach().cpu().tolist()
        prog = decode_program(pred_ids, prog_vocab_obj)
        out, _, _ = interp.execute(prog, ex.x)
        orbits = [o.x for o in ex.orbit]
        flips = [f.x for f in ex.flips]
        report = mesh.run(
            ex.x,
            prog,
            out,
            domain=ex.domain,
            expected=ex.y,
            orbits=orbits,
            flips=flips,
            mutator=mutator,
            constraints=ex.constraints,
        )
        c_sum = sum(report.c)
        conf = float(1.0 / (1.0 + c_sum))
        confs.append(conf)

        is_correct = outputs_equivalent(out, ex.y)
        correct.append(is_correct)

        formal_ok = (
            report.c[mesh.constraint_names.index("schema")]
            + report.c[mesh.constraint_names.index("arith")]
            + report.c[mesh.constraint_names.index("csp")]
            == 0.0
        )
        valid.append(formal_ok)

        # Internal invariance: mu + program skeleton stability across orbits
        orbit_ok = True
        for otext in orbits:
            orbit_ids = torch.tensor(
                [text_vocab.encode(otext, cfg["train"]["max_text_len"])],
                dtype=torch.long,
                device=torch_device,
            )
            with torch.inference_mode():
                orbit_out = model(orbit_ids)
            orbit_pred = torch.argmax(orbit_out["logits"], dim=-1)[0].detach().cpu().tolist()
            orbit_prog = decode_program(orbit_pred, prog_vocab_obj)
            orbit_exec, _, _ = interp.execute(orbit_prog, otext)
            if not outputs_equivalent(orbit_exec, out):
                orbit_ok = False
            if alignment_distance(prog, orbit_prog) > 1:
                orbit_ok = False
            if kl_divergence(base_out["mu"], orbit_out["mu"]).item() > 0.5:
                orbit_ok = False
        orbit_pass.append(orbit_ok)

        flip_ok = True
        for ftext in flips:
            flip_ids = torch.tensor(
                [text_vocab.encode(ftext, cfg["train"]["max_text_len"])],
                dtype=torch.long,
                device=torch_device,
            )
            with torch.inference_mode():
                flip_out = model(flip_ids)
            flip_pred = torch.argmax(flip_out["logits"], dim=-1)[0].detach().cpu().tolist()
            flip_prog = decode_program(flip_pred, prog_vocab_obj)
            flip_exec, _, _ = interp.execute(flip_prog, ftext)
            if outputs_equivalent(flip_exec, out):
                flip_ok = False
            if kl_divergence(base_out["mu"], flip_out["mu"]).item() < 0.1:
                flip_ok = False
        flip_pass.append(flip_ok)

        # Repair attempt
        if c_sum > 0:
            best = c_sum
            fixed = False
            for candidate in propose_repairs(prog, report.meta.get("formal", {}).get("meta", {})):
                out2, _, _ = interp.execute(candidate, ex.x)
                rep2 = mesh.run(ex.x, candidate, out2, domain=ex.domain, expected=ex.y, constraints=ex.constraints)
                c2 = sum(rep2.c)
                if c2 < best:
                    best = c2
                if c2 == 0:
                    fixed = True
                    break
            repaired.append(fixed)
        else:
            repaired.append(True)

        # Adversarial success count
        mutants = mutator.mutate(prog)
        for mprog in mutants:
            adv_total += 1
            mout, _, _ = interp.execute(mprog, ex.x)
            rep = mesh.run(ex.x, mprog, mout, domain=ex.domain, constraints=ex.constraints)
            if sum(rep.c[:3]) == 0:
                adv_success += 1

    report = {
        "verified_accuracy": verified_accuracy(correct),
        "orbit_invariance_pass_rate": orbit_invariance_pass_rate(orbit_pass),
        "flip_sensitivity_score": flip_sensitivity_score(flip_pass),
        "proof_validity_correlation": proof_validity_correlation(valid, correct),
        "repair_success_rate": repair_success_rate(repaired),
        "attack_success_rate": attack_success_rate(adv_success, adv_total),
        "selective_accuracy": selective_accuracy(confs, correct, thresholds=[0.2, 0.4, 0.6, 0.8]),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


def load_eval_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_eval_suite(config_path: str, device: str | torch.device | None = None) -> dict[str, Any]:
    cfg = load_eval_config(config_path)
    ckpt = cfg.get("ckpt", "out/ckpt.pt")
    schema_path = cfg.get("schema_path", "out/data/schema.jsonl")
    math_path = cfg.get("math_path", "out/data/math.jsonl")
    csp_path = cfg.get("csp_path", "out/data/csp.jsonl")
    out_path = cfg.get("out_path", "out/report.json")
    report = {
        "schema": run_eval(schema_path, ckpt, out_path="out/schema_report.json", device=device),
        "math": run_eval(math_path, ckpt, out_path="out/math_report.json", device=device),
        "csp": run_eval(csp_path, ckpt, out_path="out/csp_report.json", device=device),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report
