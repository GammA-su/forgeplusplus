import numpy as np
import torch

import fc
from fc.dsl import codec
from fc.eval import metrics
from fc.model import forge
from fc.train import data
from fc.util import seed as seed_mod


def test_imports_and_seed_determinism() -> None:
    assert fc.__version__
    assert codec is not None
    assert metrics is not None
    assert forge is not None
    assert data is not None

    seed_mod.set_seed(123)
    torch_a = torch.rand(4)
    np_a = np.random.rand(4)
    seed_mod.set_seed(123)
    torch_b = torch.rand(4)
    np_b = np.random.rand(4)

    assert torch.allclose(torch_a, torch_b)
    assert np.allclose(np_a, np_b)
