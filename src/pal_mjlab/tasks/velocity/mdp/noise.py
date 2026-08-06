"""Noise models for PAL exteroception observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.noise import NoiseCfg
from typing_extensions import override

__all__ = ["DepthFlickerNoiseCfg"]


@dataclass
class DepthFlickerNoiseCfg(NoiseCfg):
  """D435i-like corruption of an already-filtered, metric depth image.

  Operates on the output of `clipped_depth`: a flattened image in metres where both
  "no return" and "beyond range" are already `max_depth`. Two effects, matching what
  survives that filtering:

  - Stereo matching drops pixels frame to frame on low-texture, specular or dark
    patches. After filtering a dropped pixel is indistinguishable from far, so it is
    pinned to `max_depth` rather than zeroed.
  - Axial stereo error grows with the square of range (sigma = z^2 * sigma_disparity /
    (focal * baseline)), so the jitter is range-dependent rather than uniform.

  Pixels already at `max_depth` are left exactly there: the policy should not be able
  to tell a saturated pixel from a noisy one by looking for a value just under the cap.

  The observation manager applies noise before the term's `clip` and `scale`, so all
  parameters here are in metres, and the jitter may push a pixel a little past either
  end of the range. Bounding it is the term's job: pair this with
  `clip=(min_depth, max_depth)`.
  """

  max_depth: float
  dropout_prob: float = 0.05
  range_noise_coeff: float = 0.007

  def __post_init__(self) -> None:
    if self.operation != "add":
      raise ValueError(
        f"{type(self).__name__} defines its own corruption and only supports "
        f"operation='add'; got {self.operation!r}"
      )
    if not 0.0 <= self.dropout_prob <= 1.0:
      raise ValueError(f"dropout_prob ({self.dropout_prob}) must be in [0, 1]")
    if self.range_noise_coeff < 0.0:
      raise ValueError(f"range_noise_coeff ({self.range_noise_coeff}) must be >= 0")
    if self.max_depth <= 0.0:
      raise ValueError(f"max_depth ({self.max_depth}) must be positive")

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    # Pixels the camera reports as far, for either reason: already a miss or beyond
    # range, or dropped by stereo matching on this frame.
    saturated = data >= self.max_depth
    dropped = torch.rand_like(data) < self.dropout_prob

    # Axial stereo error, standard deviation growing with the square of range.
    sigma = self.range_noise_coeff * data.square()
    depth = data + sigma * torch.randn_like(data)

    # `depth` is a fresh tensor, so filling it in place cannot reach the sensor buffer.
    depth.masked_fill_(saturated | dropped, self.max_depth)
    return depth
