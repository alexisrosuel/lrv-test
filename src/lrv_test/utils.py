from typing import Callable


def derivative(
    f: Callable[[float], float], epsilon: float = 1e-6
) -> Callable[[float], float]:
    return lambda x: (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)
