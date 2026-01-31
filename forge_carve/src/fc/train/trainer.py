from __future__ import annotations

import json
import random
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.optim import Optimizer
import yaml
from torch import nn


class _AdafactorFallback(Optimizer):
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        *,
        lr: float,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            eps = float(group["eps"])
            clip_threshold = float(group["clip_threshold"])
            decay_rate = float(group["decay_rate"])
            weight_decay = float(group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adafactor fallback does not support sparse gradients.")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    if p.ndim >= 2:
                        state["exp_avg_sq_row"] = torch.zeros(
                            p.shape[:-1], device=p.device, dtype=torch.float32
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            p.shape[-1], device=p.device, dtype=torch.float32
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                grad_fp32 = grad.detach().float()
                if weight_decay != 0.0:
                    grad_fp32 = grad_fp32.add(p.data.float(), alpha=weight_decay)
                beta2 = 1.0 - (state["step"] ** decay_rate)
                if p.ndim >= 2:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2).add_(grad_fp32.pow(2).mean(dim=-1), alpha=1 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(grad_fp32.pow(2).mean(dim=0), alpha=1 - beta2)
                    r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean()).rsqrt().unsqueeze(-1)
                    c_factor = exp_avg_sq_col.rsqrt().unsqueeze(0)
                    update = grad_fp32 * (r_factor * c_factor)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)
                    update = grad_fp32 * (exp_avg_sq.rsqrt().add_(eps))
                update_rms = update.pow(2).mean().sqrt()
                clip_denom = max(1.0, float(update_rms / clip_threshold))
                update = update / clip_denom
                p.data.add_(update.to(p.dtype), alpha=-lr)
        return loss


def _build_optimizer(
    name: str,
    params: list[torch.nn.Parameter],
    lr: float,
    betas: tuple[float, float],
):
    def _adamw_kwargs() -> dict[str, Any]:
        kwargs: dict[str, Any] = {"lr": lr, "betas": betas}
        try:
            torch.optim.AdamW(params, **kwargs, foreach=False, capturable=False)
            kwargs["foreach"] = False
            kwargs["capturable"] = False
        except TypeError:
            pass
        return kwargs

    if name == "adamw":
        return torch.optim.AdamW(params, **_adamw_kwargs())
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("adamw8bit requested but bitsandbytes not available; use adafactor instead.") from exc
        return bnb.optim.AdamW8bit(params, lr=lr, betas=betas)
    if name == "adafactor":
        try:
            from transformers import Adafactor  # type: ignore
        except Exception:
            return _AdafactorFallback(params, lr=lr)
        return Adafactor(
            params,
            lr=lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    raise ValueError(f"Unknown optimizer: {name}")

from fc.dsl.codec import alignment_distance, decode_program
from fc.dsl.tokens import TokenVocab
from fc.interp.core import Interpreter
from fc.model.backbone import BackboneConfig
from fc.model.forge import ForgeModel, ModelConfig
from fc.model.primal_dual import PrimalDualConfig
from fc.model.slots import SlotConfig
from fc.train.data import (
    Example,
    TextVocab,
    audit_proof_tokens,
    audit_proof_tokens_against_vocab,
    audit_proof_tokens_from_paths,
    build_program_vocab_from_examples,
    collate_batch,
    load_dataset,
    load_dataset_with_variants,
)
from fc.train.losses import (
    causal_faithfulness_loss,
    kkt_loss,
    mdl_loss,
    orbit_invariance_loss,
    regret_loss,
    state_progress_loss,
)
from fc.util.jsonl import write_jsonl
from fc.util.vocab_identity import assert_vocab_match, vocab_identity
from fc.util.logging import configure_logging, get_logger
from fc.util.seed import set_seed
from fc.verify.mesh import VerifierMesh


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    steps: int
    batch_size: int
    lr: float
    max_text_len: int
    max_prog_len: int
    mdl_alpha: float
    mdl_beta: float
    regret_margin: float
    causal_delta: float
    shuffle: bool = False
    proof_supervision: bool = True
    proof_supervision_source: str = "proof"
    precision: str = "auto"
    microbatch: int = 1
    grad_accum: int = 1
    grad_checkpoint: bool = True
    optimizer: str = "adafactor"
    adamw_betas: tuple[float, float] = (0.9, 0.999)


@dataclass(frozen=True)
class WeightConfig:
    ce: float = 1.0
    proof: float = 1.0
    kkt: float = 0.2
    mdl: float = 0.1
    regret: float = 0.1
    orbit: float = 0.1
    causal: float = 0.1
    state: float = 0.1


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(cfg: dict[str, Any], vocab_size: int) -> ForgeModel:
    bcfg = BackboneConfig(vocab_size=cfg["text_vocab_size"], **cfg["backbone"])
    scfg = SlotConfig(**cfg["slots"])
    pcfg = PrimalDualConfig(**cfg["primal_dual"])
    mcfg = ModelConfig(vocab_size=vocab_size, max_prog_len=cfg["max_prog_len"], backbone=bcfg, slots=scfg, primal_dual=pcfg)
    return ForgeModel(mcfg)


def _sync_max_prog_len(cfg: dict[str, Any], train_cfg: TrainConfig) -> None:
    cfg["max_prog_len"] = train_cfg.max_prog_len


def _resize_policy_head(
    model: ForgeModel,
    state: dict[str, torch.Tensor],
    *,
    old_max_prog_len: int,
    new_max_prog_len: int,
    vocab_size: int,
) -> None:
    if old_max_prog_len == new_max_prog_len:
        return
    weight_key = "policy.head.weight"
    bias_key = "policy.head.bias"
    if weight_key not in state or bias_key not in state:
        return
    old_weight = state[weight_key]
    old_bias = state[bias_key]
    expected_old = old_max_prog_len * vocab_size
    expected_new = new_max_prog_len * vocab_size
    if old_weight.shape[0] != expected_old or old_bias.shape[0] != expected_old:
        raise ValueError(
            "init_ckpt policy head shape mismatch "
            f"expected_rows={expected_old} got_weight_rows={old_weight.shape[0]}"
        )
    new_weight = model.policy.head.weight.detach().clone()
    new_bias = model.policy.head.bias.detach().clone()
    if new_weight.shape[0] != expected_new or new_bias.shape[0] != expected_new:
        raise ValueError(
            "policy head resize failed "
            f"expected_rows={expected_new} got_weight_rows={new_weight.shape[0]}"
        )
    copy_len = min(old_max_prog_len, new_max_prog_len) * vocab_size
    if copy_len > 0:
        new_weight[:copy_len] = old_weight[:copy_len]
        new_bias[:copy_len] = old_bias[:copy_len]
    state[weight_key] = new_weight
    state[bias_key] = new_bias


def _resize_positional_embeddings(
    model: ForgeModel,
    state: dict[str, torch.Tensor],
    logger: Any,
) -> None:
    model_state = model.state_dict()
    for key, old_weight in list(state.items()):
        if not key.endswith("pos_emb.weight") and ".pos_emb." not in key:
            continue
        if key not in model_state:
            continue
        target_weight = model_state[key]
        if old_weight.shape == target_weight.shape:
            continue
        if old_weight.ndim != 2 or target_weight.ndim != 2:
            raise ValueError(
                "init_ckpt pos_emb rank mismatch "
                f"key={key} expected_rank={target_weight.ndim} got_rank={old_weight.ndim}"
            )
        old_len, dim = old_weight.shape
        new_len, new_dim = target_weight.shape
        if dim != new_dim:
            raise ValueError(
                "init_ckpt pos_emb dim mismatch "
                f"key={key} expected_dim={new_dim} got_dim={dim}"
            )
        if new_len <= old_len:
            new_weight = old_weight[:new_len].clone()
            method = "truncate"
        else:
            new_weight = target_weight.detach().clone()
            new_weight[:old_len] = old_weight
            extra = torch.randn(new_len - old_len, dim, device=old_weight.device) * 0.02
            new_weight[old_len:] = extra
            method = "noise_init"
        state[key] = new_weight
        logger.warning(
            "resized init pos_emb: key=%s old_len=%d new_len=%d method=%s",
            key,
            old_len,
            new_len,
            method,
        )


def _load_init_checkpoint(
    init_ckpt: dict[str, Any],
    init_ckpt_path: str,
    *,
    model: ForgeModel,
    prog_vocab: TokenVocab,
    cfg: dict[str, Any],
    logger: Any,
) -> None:
    if "model" not in init_ckpt:
        raise ValueError(f"init_ckpt missing model state: {init_ckpt_path}")
    if "prog_vocab" in init_ckpt:
        assert_vocab_match(
            prog_vocab.token_to_id,
            init_ckpt["prog_vocab"],
            expected_label="train.prog_vocab",
            actual_label=f"{init_ckpt_path}:prog_vocab",
        )
    if "prog_vocab_sha256" in init_ckpt:
        current_id = vocab_identity(prog_vocab.token_to_id)
        if init_ckpt["prog_vocab_sha256"] != current_id.sha256:
            raise ValueError(
                "init_ckpt prog_vocab_sha256 mismatch "
                f"expected={current_id.sha256} got={init_ckpt['prog_vocab_sha256']}"
            )
    init_cfg = init_ckpt.get("config", {})
    init_text_vocab_size = init_cfg.get("text_vocab_size")
    if init_text_vocab_size is not None and init_text_vocab_size != cfg.get("text_vocab_size"):
        logger.warning(
            "init_ckpt text_vocab_size mismatch expected=%s got=%s",
            cfg.get("text_vocab_size"),
            init_text_vocab_size,
        )
    init_max_prog_len = init_ckpt.get("max_prog_len") or init_cfg.get("max_prog_len")
    if init_max_prog_len is None:
        init_max_prog_len = model.cfg.max_prog_len
    state = dict(init_ckpt["model"])
    _resize_positional_embeddings(model, state, logger)
    _resize_policy_head(
        model,
        state,
        old_max_prog_len=int(init_max_prog_len),
        new_max_prog_len=model.cfg.max_prog_len,
        vocab_size=model.cfg.vocab_size,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            "init_ckpt state mismatch "
            f"missing={missing} unexpected={unexpected}"
        )
    logger.info("loaded init_ckpt=%s", init_ckpt_path)


def _apply_mode(cfg: dict[str, Any]) -> str:
    mode = cfg.get("mode", "forge")
    if mode == "ablation":
        cfg.setdefault("primal_dual", {})
        cfg["primal_dual"]["steps"] = 1
        weights = cfg.setdefault("weights", {})
        weights["kkt"] = 0.0
        weights["orbit"] = 0.0
    cfg["mode"] = mode
    return mode

def load_examples(
    schema_path: str,
    math_path: str,
    csp_path: str,
    include_variants: bool = True,
) -> list[Example]:
    loader = load_dataset_with_variants if include_variants else load_dataset
    return loader(schema_path) + loader(math_path) + loader(csp_path)


def _epoch_indices(size: int, rng: random.Random, shuffle: bool) -> Iterator[int]:
    indices = list(range(size))
    if shuffle:
        rng.shuffle(indices)
    return iter(indices)


def _next_batch(
    examples: list[Example],
    iterator: Iterator[int],
    batch_size: int,
    rng: random.Random,
    shuffle: bool,
) -> tuple[list[Example], Iterator[int]]:
    if not examples:
        raise ValueError("Cannot iterate empty dataset.")
    batch: list[Example] = []
    while len(batch) < batch_size:
        try:
            idx = next(iterator)
        except StopIteration:
            iterator = _epoch_indices(len(examples), rng, shuffle)
            continue
        batch.append(examples[idx])
    return batch, iterator


def train(
    examples: list[Example],
    config_path: str,
    out_dir: str,
    device: str | torch.device | None = None,
    *,
    cfg_override: dict[str, Any] | None = None,
    init_ckpt_path: str | None = None,
) -> Path:
    cfg = cfg_override if cfg_override is not None else load_config(config_path)
    mode = _apply_mode(cfg)
    weights = cfg.setdefault("weights", {})
    if "proof" not in weights and "ce" in weights:
        weights["proof"] = weights["ce"]
    train_cfg = TrainConfig(**cfg["train"])
    _sync_max_prog_len(cfg, train_cfg)
    weight_cfg = WeightConfig(**cfg.get("weights", {}))
    set_seed(train_cfg.seed)
    configure_logging()
    logger = get_logger(__name__)

    init_ckpt: dict[str, Any] | None = None
    init_text_vocab: TextVocab | None = None
    init_prog_vocab: TokenVocab | None = None
    if init_ckpt_path:
        init_ckpt = torch.load(init_ckpt_path, map_location="cpu")
        init_text_map = init_ckpt.get("text_vocab")
        if not isinstance(init_text_map, dict):
            raise ValueError(f"init_ckpt missing text_vocab: {init_ckpt_path}")
        init_text_vocab = TextVocab(
            token_to_id=init_text_map,
            id_to_token={i: t for t, i in init_text_map.items()},
        )
        init_prog_map = init_ckpt.get("prog_vocab")
        if isinstance(init_prog_map, dict):
            init_prog_vocab = TokenVocab(
                token_to_id=init_prog_map,
                id_to_token={i: t for t, i in init_prog_map.items()},
            )

    texts = [ex.x for ex in examples]
    for ex in examples:
        texts.extend([o.x for o in ex.orbit])
        texts.extend([f.x for f in ex.flips])
    dataset_text_vocab = TextVocab.build(texts)
    if init_text_vocab is not None:
        if dataset_text_vocab.token_to_id != init_text_vocab.token_to_id:
            extra = sorted(set(dataset_text_vocab.token_to_id) - set(init_text_vocab.token_to_id))
            logger.warning(
                "dataset text_vocab differs from init_ckpt; reusing init vocab extra_tokens=%d sample=%s",
                len(extra),
                extra[:5],
            )
        text_vocab = init_text_vocab
    else:
        text_vocab = dataset_text_vocab

    if init_prog_vocab is not None:
        audit_proof_tokens_against_vocab(
            examples,
            init_prog_vocab,
            proof_source=train_cfg.proof_supervision_source,
        )
        prog_vocab = init_prog_vocab
    else:
        audit_proof_tokens(examples, proof_source=train_cfg.proof_supervision_source)
        prog_vocab = build_program_vocab_from_examples(
            examples,
            proof_source=train_cfg.proof_supervision_source,
        )

    cfg["text_vocab_size"] = len(text_vocab.token_to_id)
    model = _build_model(cfg, vocab_size=len(prog_vocab.token_to_id))
    model.set_prog_decoder(lambda ids: decode_program(ids, prog_vocab))
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = str(device)
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
    torch_device = torch.device(device_str)
    model.to(torch_device)
    model.train()
    logger.info("trainer device=%s", torch_device)
    param_dtype = None
    for param in model.parameters():
        param_dtype = param.dtype
        break
    logger.info("trainer optimizer=%s param_dtype=%s", train_cfg.optimizer, param_dtype)
    if train_cfg.grad_checkpoint:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("trainer grad_checkpoint=on")
        elif hasattr(model, "backbone") and hasattr(model.backbone, "gradient_checkpointing_enable"):
            model.backbone.gradient_checkpointing_enable()
            logger.info("trainer grad_checkpoint=on (backbone)")
        else:
            logger.info("trainer grad_checkpoint=on (no-op)")
    if init_ckpt_path:
        _load_init_checkpoint(
            init_ckpt if init_ckpt is not None else torch.load(init_ckpt_path, map_location="cpu"),
            init_ckpt_path,
            model=model,
            prog_vocab=prog_vocab,
            cfg=cfg,
            logger=logger,
        )

    optimizer = _build_optimizer(
        train_cfg.optimizer,
        list(model.parameters()),
        lr=train_cfg.lr,
        betas=train_cfg.adamw_betas,
    )
    precision = train_cfg.precision
    if precision == "auto":
        if torch_device.type == "cuda":
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            precision = "fp32"
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unknown precision: {precision}")
    use_autocast = torch_device.type == "cuda" and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16" and torch_device.type == "cuda")
    microbatch = max(1, int(train_cfg.microbatch))
    grad_accum = max(1, int(train_cfg.grad_accum))
    logger.info(
        "trainer precision=%s microbatch=%d grad_accum=%d grad_checkpoint=%s",
        precision,
        microbatch,
        grad_accum,
        train_cfg.grad_checkpoint,
    )
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=prog_vocab.token_to_id["<PAD>"])
    pad_id = prog_vocab.token_to_id["<PAD>"]

    adjacency = torch.eye(cfg["slots"]["num_states"], dtype=torch.float32, device=torch_device)
    interp = Interpreter()
    mesh = VerifierMesh()
    train_logs: list[dict[str, Any]] = []

    rng = random.Random(train_cfg.seed)
    batch_iter = _epoch_indices(len(examples), rng, train_cfg.shuffle)
    for step in range(train_cfg.steps):
        batch, batch_iter = _next_batch(examples, batch_iter, train_cfg.batch_size, rng, train_cfg.shuffle)
        micro_sums = {
            "loss": 0.0,
            "prog_ce": 0.0,
            "kkt": 0.0,
            "mdl": 0.0,
            "regret": 0.0,
            "orbit": 0.0,
            "causal": 0.0,
            "state": 0.0,
            "c_hard_mean": 0.0,
            "c_soft_mean": 0.0,
        }
        micro_count = 0
        optimizer.zero_grad(set_to_none=True)
        for micro_idx in range(0, len(batch), microbatch):
            micro = batch[micro_idx : micro_idx + microbatch]
            batch_data = collate_batch(
                micro,
                text_vocab,
                prog_vocab,
                train_cfg.max_text_len,
                train_cfg.max_prog_len,
                proof_source=train_cfg.proof_supervision_source,
            )
            input_ids = batch_data["input_ids"].to(torch_device)
            program_ids = batch_data["program_ids"].to(torch_device)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else contextlib.nullcontext()
            )
            with autocast_ctx:
                outputs = model(input_ids)
                logits = outputs["logits"]
                mu = outputs["mu"]
                c_soft = outputs["c_soft"]
                state_logits = outputs["state_logits"]
                pred_program_ids = outputs["program_ids"]
                candidate_scores = outputs["candidate_scores"]
                chosen_idx = outputs["chosen_index"]
                programs = outputs["programs"]

                if train_cfg.proof_supervision and weight_cfg.proof > 0.0 and (program_ids != pad_id).any():
                    prog_ce = ce_loss_fn(logits.view(-1, logits.size(-1)), program_ids.view(-1))
                else:
                    prog_ce = torch.tensor(0.0, device=torch_device)
                program_len = (pred_program_ids != pad_id).sum(dim=1)

                if weight_cfg.kkt > 0.0:
                    c_hard_list: list[list[float]] = []
                    for ex, prog in zip(micro, programs):
                        out, _, _ = interp.execute(prog, ex.x)
                        orbits = [o.x for o in ex.orbit]
                        flips = [f.x for f in ex.flips]
                        report = mesh.run(
                            ex.x,
                            prog,
                            out,
                            domain=ex.domain,
                            orbits=orbits,
                            flips=flips,
                            constraints=ex.constraints,
                        )
                        c_hard_list.append(list(report.c))
                    c_hard = torch.tensor(c_hard_list, dtype=torch.float32, device=torch_device)
                    kkt = kkt_loss(c_hard, mu)
                else:
                    c_hard = torch.zeros(program_ids.size(0), len(mesh.constraint_names), device=torch_device)
                    kkt = torch.tensor(0.0, device=torch_device)
                mdl = mdl_loss(program_len, mu, train_cfg.mdl_alpha, train_cfg.mdl_beta)

                reg = regret_loss(-candidate_scores, chosen_idx, train_cfg.regret_margin)

                if weight_cfg.orbit > 0.0:
                    orbit_texts = [ex.orbit[0].x if ex.orbit else ex.x for ex in micro]
                    orbit_ids = torch.tensor(
                        [text_vocab.encode(t, train_cfg.max_text_len) for t in orbit_texts],
                        dtype=torch.long,
                        device=torch_device,
                    )
                    orbit_out = model(orbit_ids)
                    orbit_prog_dist: list[float] = []
                    for base_prog, orb_prog in zip(programs, orbit_out["programs"]):
                        if base_prog.instructions and orb_prog.instructions:
                            orbit_prog_dist.append(float(alignment_distance(base_prog, orb_prog)))
                        else:
                            orbit_prog_dist.append(0.0)
                    orbit_dist = torch.tensor(orbit_prog_dist, device=torch_device, dtype=torch.float32)
                    orbit = orbit_invariance_loss(mu, orbit_out["mu"], orbit_dist)
                else:
                    orbit = torch.tensor(0.0, device=torch_device)

                flip_texts = [ex.flips[0].x if ex.flips else ex.x for ex in micro]
                flip_ids = torch.tensor(
                    [text_vocab.encode(t, train_cfg.max_text_len) for t in flip_texts],
                    dtype=torch.long,
                    device=torch_device,
                )
                flip_out = model(flip_ids)
                causal = causal_faithfulness_loss(mu, flip_out["mu"], train_cfg.causal_delta)

                state_targets = torch.zeros(program_ids.size(0), dtype=torch.long, device=torch_device)
                state_loss = state_progress_loss(state_logits, state_targets, adjacency)

                loss = (
                    weight_cfg.proof * prog_ce
                    + weight_cfg.kkt * kkt
                    + weight_cfg.mdl * mdl
                    + weight_cfg.regret * reg
                    + weight_cfg.orbit * orbit
                    + weight_cfg.causal * causal
                    + weight_cfg.state * state_loss
                )

            micro_sums["loss"] += float(loss.item())
            micro_sums["prog_ce"] += float(prog_ce.item())
            micro_sums["kkt"] += float(kkt.item())
            micro_sums["mdl"] += float(mdl.item())
            micro_sums["regret"] += float(reg.item())
            micro_sums["orbit"] += float(orbit.item())
            micro_sums["causal"] += float(causal.item())
            micro_sums["state"] += float(state_loss.item())
            micro_sums["c_hard_mean"] += float(c_hard.mean().item())
            micro_sums["c_soft_mean"] += float(c_soft.mean().item())
            micro_count += 1

            loss = loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_step = ((micro_idx // microbatch) + 1) % grad_accum == 0 or (micro_idx + microbatch) >= len(batch)
            if is_step:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        denom = max(1, micro_count)
        log_row = {
            "step": step,
            "loss": micro_sums["loss"] / denom,
            "prog_ce": micro_sums["prog_ce"] / denom,
            "kkt": micro_sums["kkt"] / denom,
            "mdl": micro_sums["mdl"] / denom,
            "regret": micro_sums["regret"] / denom,
            "orbit": micro_sums["orbit"] / denom,
            "causal": micro_sums["causal"] / denom,
            "state": micro_sums["state"] / denom,
            "c_hard_mean": micro_sums["c_hard_mean"] / denom,
            "c_soft_mean": micro_sums["c_soft_mean"] / denom,
        }
        train_logs.append(log_row)
        if step % 10 == 0:
            logger.info(
                "step=%d loss=%.4f prog_ce=%.4f kkt=%.4f orbit=%.4f",
                step,
                loss.item(),
                prog_ce.item(),
                kkt.item(),
                orbit.item(),
            )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    prog_id = vocab_identity(prog_vocab.token_to_id)
    ckpt = {
        "mode": mode,
        "model": model.state_dict(),
        "text_vocab": text_vocab.token_to_id,
        "prog_vocab": prog_vocab.token_to_id,
        "prog_vocab_sha256": prog_id.sha256,
        "prog_vocab_tokens": prog_id.tokens_by_id,
        "max_prog_len": cfg.get("max_prog_len", train_cfg.max_prog_len),
        "config": cfg,
    }
    ckpt_path = out_path / "ckpt.pt"
    torch.save(ckpt, ckpt_path)
    # Save vocab as JSON for inspection
    (out_path / "text_vocab.json").write_text(json.dumps(text_vocab.token_to_id, indent=2))
    (out_path / "prog_vocab.json").write_text(json.dumps(prog_vocab.token_to_id, indent=2))
    write_jsonl(out_path / "train_log.jsonl", train_logs)
    return ckpt_path


def train_from_paths(
    config_path: str,
    out_dir: str,
    schema_path: str = "out/data/schema.jsonl",
    math_path: str = "out/data/math.jsonl",
    csp_path: str = "out/data/csp.jsonl",
    device: str | torch.device | None = None,
    *,
    init_ckpt_path: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    proof_source = cfg.get("train", {}).get("proof_supervision_source", "proof")
    audit_proof_tokens_from_paths(
        [schema_path, math_path, csp_path],
        proof_source=proof_source,
    )
    examples = load_examples(schema_path, math_path, csp_path, include_variants=True)
    return train(
        examples,
        config_path=config_path,
        out_dir=out_dir,
        device=device,
        init_ckpt_path=init_ckpt_path,
    )


def train_from_dataset(
    config_path: str,
    data_path: str,
    out_dir: str,
    device: str | torch.device | None = None,
    *,
    init_ckpt_path: str | None = None,
    max_prog_len: int | None = None,
    steps: int | None = None,
    include_variants: bool = True,
    proof_source: str | None = None,
    precision: str | None = None,
    microbatch: int | None = None,
    grad_accum: int | None = None,
    grad_checkpoint: bool | None = None,
    optimizer: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    if max_prog_len is not None:
        cfg.setdefault("train", {})["max_prog_len"] = int(max_prog_len)
        cfg["max_prog_len"] = int(max_prog_len)
        cfg.setdefault("eval", {}).setdefault("max_prog_len", int(max_prog_len))
    if steps is not None:
        cfg.setdefault("train", {})["steps"] = int(steps)
    if precision is not None:
        cfg.setdefault("train", {})["precision"] = precision
    else:
        cfg.setdefault("train", {}).setdefault("precision", "auto")
    if microbatch is not None:
        cfg.setdefault("train", {})["microbatch"] = int(microbatch)
    else:
        cfg.setdefault("train", {}).setdefault("microbatch", 1)
    if grad_accum is not None:
        cfg.setdefault("train", {})["grad_accum"] = int(grad_accum)
    else:
        cfg.setdefault("train", {}).setdefault("grad_accum", 1)
    if grad_checkpoint is not None:
        cfg.setdefault("train", {})["grad_checkpoint"] = bool(grad_checkpoint)
    else:
        cfg.setdefault("train", {}).setdefault("grad_checkpoint", True)
    if optimizer is not None:
        cfg.setdefault("train", {})["optimizer"] = optimizer
    else:
        cfg.setdefault("train", {}).setdefault("optimizer", "adafactor")
    resolved_proof_source = proof_source or cfg.get("train", {}).get("proof_supervision_source", "proof")
    cfg.setdefault("train", {})["proof_supervision_source"] = resolved_proof_source
    if init_ckpt_path is None:
        audit_proof_tokens_from_paths([data_path], proof_source=resolved_proof_source)
    loader = load_dataset_with_variants if include_variants else load_dataset
    examples = loader(data_path)
    return train(
        examples,
        config_path=config_path,
        out_dir=out_dir,
        device=device,
        cfg_override=cfg,
        init_ckpt_path=init_ckpt_path,
    )
