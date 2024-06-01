from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from lrv_test.config.contour import ContourConfig


@dataclass(frozen=True)
class Contour:
    z: Callable[[float], complex]
    dz: Callable[[float], complex]
    t_range: tuple[float, float]

    @classmethod
    def from_circle_parameters(cls, center: float, radius: float) -> Contour:
        """Complex a circle countour, with edge points at min_x + min_y i,
        min_x + max_y i, max_x + min_y i, min_x + min_y i.
        Oriented counter-clockwise.
        """
        z = lambda t: center + radius * np.exp(-1j * t * 2 * np.pi)
        dz = lambda t: -1j * 2 * np.pi * radius * np.exp(-1j * t * 2 * np.pi)
        t_range = (0, 1)
        return cls(z, dz, t_range)


def create_contour(
    contour_config: ContourConfig, eig_range: tuple[float, float]
) -> Contour:
    min_eig, max_eig = eig_range
    eig_diameter = max_eig - min_eig
    contour_range = (
        min_eig - contour_config.real_slack * 0.5 * eig_diameter,
        max_eig + contour_config.real_slack * 0.5 * eig_diameter,
    )

    if contour_config.type_ == "circle":
        center = (contour_range[0] + contour_range[1]) / 2
        radius = (contour_range[1] - contour_range[0]) / 2
        contour = Contour.from_circle_parameters(center, radius)
    else:
        raise ValueError(f"Unknown contour type: {contour_config.type_}")

    return contour
