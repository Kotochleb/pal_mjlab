"""Tests for `DepthFlickerNoiseCfg`.

The corruption is silent when its *direction* is wrong: a dropped pixel on the real
camera reads as far, because `clipped_depth` maps both "no return" and "beyond range" to
`max_depth`. Noise that pushed pixels toward the lens instead would teach the opposite
reflex, and nothing downstream would complain.

The noise deliberately does not bound its own output -- the observation term's `clip`
does, in metres, before the 1 / max_depth scaling. The last test here pins that pairing,
since noise without the clip would put out-of-range values in front of the policy.
"""

import pytest
import torch
from conftest import get_test_device
from pal_mjlab.tasks.velocity.mdp import DepthFlickerNoiseCfg

ROWS, COLS = 12, 16
MAX_DEPTH = 2.0
DROPOUT_PROB = 0.05
RANGE_NOISE_COEFF = 0.007


def _cfg(**overrides) -> DepthFlickerNoiseCfg:
  kwargs = dict(
    max_depth=MAX_DEPTH,
    dropout_prob=DROPOUT_PROB,
    range_noise_coeff=RANGE_NOISE_COEFF,
  )
  kwargs.update(overrides)
  return DepthFlickerNoiseCfg(**kwargs)  # type: ignore[arg-type]


def _mixed_image(device: str) -> torch.Tensor:
  """One image spanning every branch: very near, mid, just under the cap, saturated."""
  row = torch.tensor(
    [0.06, 0.3, 0.8, 1.2, 1.6, 1.99, MAX_DEPTH, MAX_DEPTH], device=device
  )
  return row.repeat(4, ROWS * COLS // row.numel())


def test_output_is_finite_and_shape_preserving() -> None:
  device = get_test_device()
  torch.manual_seed(0)
  data = _mixed_image(device)

  out = _cfg().apply(data)

  assert out.shape == data.shape
  assert torch.isfinite(out).all()
  # Bounding is the term's `clip`; all this owes is a value near the truth or at the cap.
  assert (out >= 0.0).all()


def test_saturated_pixels_stay_exactly_at_max() -> None:
  device = get_test_device()
  torch.manual_seed(0)
  data = torch.full((8, ROWS * COLS), MAX_DEPTH, device=device)

  out = _cfg(dropout_prob=0.0).apply(data)

  # Jittering these would let the policy tell "nothing there" from "far" by looking for
  # a value just under the cap, which the real camera does not offer.
  assert torch.equal(out, data)


def test_zeroed_parameters_are_the_identity() -> None:
  device = get_test_device()
  torch.manual_seed(0)
  data = _mixed_image(device)

  out = _cfg(dropout_prob=0.0, range_noise_coeff=0.0).apply(data)

  assert torch.equal(out, data)


def test_full_dropout_pins_everything_to_max() -> None:
  device = get_test_device()
  torch.manual_seed(0)
  data = _mixed_image(device)

  out = _cfg(dropout_prob=1.0).apply(data)

  assert torch.equal(out, torch.full_like(data, MAX_DEPTH))


def test_dropout_rate_matches_the_configured_probability() -> None:
  device = get_test_device()
  torch.manual_seed(0)
  # Mid-range and jitter-free, so the only way to land on max_depth is a dropout.
  data = torch.full((4096, ROWS * COLS), 1.0, device=device)

  out = _cfg(range_noise_coeff=0.0).apply(data)

  rate = (out >= MAX_DEPTH).float().mean().item()
  assert rate == pytest.approx(DROPOUT_PROB, abs=0.005)


def test_dropout_only_ever_pushes_pixels_toward_far() -> None:
  device = get_test_device()
  torch.manual_seed(0)
  data = torch.full((256, ROWS * COLS), 1.0, device=device)

  out = _cfg().apply(data)

  # Every pixel is either jittered by a few sigma or pinned to max_depth; none may read
  # substantially nearer than the truth, which is what a hole-as-zero model would do.
  sigma = RANGE_NOISE_COEFF * 1.0**2
  assert (out > 1.0 - 6.0 * sigma).all()


def test_jitter_grows_with_the_square_of_range() -> None:
  device = get_test_device()
  torch.manual_seed(0)
  near = torch.full((8192, ROWS * COLS), 0.5, device=device)
  far = torch.full((8192, ROWS * COLS), 1.5, device=device)

  cfg = _cfg(dropout_prob=0.0)
  near_std = (cfg.apply(near) - near).std().item()
  far_std = (cfg.apply(far) - far).std().item()

  assert near_std == pytest.approx(RANGE_NOISE_COEFF * 0.5**2, rel=0.05)
  assert far_std == pytest.approx(RANGE_NOISE_COEFF * 1.5**2, rel=0.05)
  # sigma ~ z^2, so tripling the range multiplies the spread by nine.
  assert far_std / near_std == pytest.approx(9.0, rel=0.1)


@pytest.mark.parametrize(
  "overrides",
  [
    {"operation": "scale"},
    {"dropout_prob": 1.5},
    {"dropout_prob": -0.1},
    {"range_noise_coeff": -0.001},
    {"max_depth": 0.0},
  ],
)
def test_invalid_configuration_is_rejected(overrides: dict) -> None:
  with pytest.raises(ValueError):
    _cfg(**overrides)


def test_actor_depth_term_clips_the_noise_it_carries() -> None:
  from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import (
    CLIPPED_DEPTH_MAX,
    CLIPPED_DEPTH_MIN,
    pal_kangaroo_rough_env_cfg,
  )

  cfg = pal_kangaroo_rough_env_cfg()
  actor = cfg.observations["actor"].terms["clipped_depth"]

  assert isinstance(actor.noise, DepthFlickerNoiseCfg)
  assert actor.noise.max_depth == CLIPPED_DEPTH_MAX, (
    "the noise must saturate at the same value `clipped_depth` fills misses with"
  )
  # The manager runs noise -> clip -> scale, so without this the jitter would reach the
  # policy unbounded and 1 / CLIPPED_DEPTH_MAX would stop normalising.
  assert actor.clip == (CLIPPED_DEPTH_MIN, CLIPPED_DEPTH_MAX)
  assert CLIPPED_DEPTH_MIN > 0.0, "a zero depth reads as a surface against the lens"

  # The critic stays privileged: clean image, no corruption.
  assert cfg.observations["critic"].terms["clipped_depth"].noise is None
