import torch

from fc.model.backbone import BackboneConfig
from fc.model.forge import ForgeModel, ModelConfig
from fc.model.primal_dual import PrimalDualConfig
from fc.model.slots import SlotConfig


def test_model_forward_shapes() -> None:
    cfg = ModelConfig(
        vocab_size=32,
        max_prog_len=16,
        backbone=BackboneConfig(vocab_size=20, d_model=32, n_heads=2, n_layers=1, d_ff=64, max_len=32),
        slots=SlotConfig(num_slots=2, d_slot=16, num_states=3),
        primal_dual=PrimalDualConfig(num_constraints=5, steps=2, d_slot=16),
    )
    model = ForgeModel(cfg)
    input_ids = torch.randint(0, cfg.backbone.vocab_size, (4, 10))
    outputs = model(input_ids)
    assert outputs["logits"].shape == (4, cfg.max_prog_len, cfg.vocab_size)
    assert outputs["mu"].shape == (4, cfg.primal_dual.num_constraints)
    assert outputs["c_soft"].shape == (4, cfg.primal_dual.num_constraints)
    assert outputs["program_ids"].shape == (4, cfg.max_prog_len)
    assert outputs["candidate_ids"].shape[:2] == (4, 3)
    assert outputs["candidate_scores"].shape == (4, 3)
    assert outputs["chosen_index"].shape == (4,)
