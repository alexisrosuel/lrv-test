import numpy as np

""" 
This file compiles the functions used in the paper to compute the test statistics. 
It is mainly about the Stieltjes transform of the Marchenko-Pastur distribution, and
some variations on it. 
"""


def support_MP(c: float) -> tuple[float, float]:
    low = max(0, (1 - np.sqrt(c)) ** 2)
    high = (1 + np.sqrt(c)) ** 2
    return (low, high)


def t(z: complex, c: float) -> complex:
    """
    Stieltjes transform of the Marchenko-Pastur distribution
    """
    # Compute the discriminant for the square root term
    discriminant = (z - 1 - c) ** 2 - 4 * c

    # Compute the two solutions of the square root
    sqrt_discriminant_1 = np.sqrt(discriminant)
    sqrt_discriminant_2 = -sqrt_discriminant_1

    # Compute the two possible solutions for G(z)
    G1 = -((1 - c) - z + sqrt_discriminant_1) / (2 * c * z)
    G2 = -((1 - c) - z + sqrt_discriminant_2) / (2 * c * z)

    # Return the solution with positive imaginary part
    return G1 if np.imag(G1) > 0 else G2


def t_tilde(z: complex, c: float) -> complex:
    return -1 / (z * (1 + c * t(z, c)))


def z_t_t_tilde(z: complex, c: float) -> complex:
    return z * t(z, c) * t_tilde(z, c)
