from typing import Callable

import numpy as np
from scipy.integrate import quad

from lrv_test.contour import Contour
from lrv_test.functions import support_MP, z_t_t_tilde
from lrv_test.types import real_function
from lrv_test.utils import contour_integral, psi


def s(z: complex, c: float) -> complex:
    z_t_t_tilde_square = z_t_t_tilde(z, c) ** 2
    return np.sqrt(c) * z_t_t_tilde_square / (1 - c * z_t_t_tilde_square)



def rectangle_integral(f: real_function, min_x: float, max_x: float, min_y: float, max_y: float) -> float: 
    # compute the four integrals
    integral_1 = quad(lambda x: f(x + 1j * max_y), min_x, max_x, complex_func=True)[0]
    integral_2 = quad(lambda y: f(max_x + 1j * y), max_y, min_y, complex_func=True)[0]
    integral_3 = quad(lambda x: f(x + 1j * min_y), max_x, min_x, complex_func=True)[0]
    integral_4 = quad(lambda y: f(min_x + 1j * y), min_y, max_y, complex_func=True)[0]

    # return the sum of the four integrals
    return integral_1 + integral_2 + integral_3 + integral_4



def compute_sigma(f: real_function, c: float) -> float:
    # create the contour used in contour integration
    support = support_MP(c)
    # radius = (support[1] - support[0]) / 2 
    # center = (support[0] + support[1]) / 2
    # contour = Contour.from_circle_parameters(center, radius)

    # we integrate over a rectangle 
    # epsilon = 1e-6
    # contour_integral = lambda n: rectangle_integral(
    #     lambda z: np.sqrt(n+1) * f(z) * s(z, c) * (np.sqrt(c) * z_t_t_tilde(z, c)) ** n,
    #      support[0] - epsilon,
    #      support[1] + epsilon,
    #      -epsilon,
    #      epsilon,
    # )
    
    radius = np.sqrt(c)+0.1
    center = 0
    contour = Contour.from_circle_parameters(center, radius)
    def integrand(w, n): 
        return - np.sqrt(n+1) * c * f(psi(w, c)) / w**(n+2)
    
    ci = lambda n: contour_integral(integrand=lambda w: integrand(w, n), contour=contour)

    # for i in range(1, 10): 
    #     print(ci(i))

    # import pdb; pdb.set_trace()


    # compute the integrals until a term is small enough
    cis = [] 
    n = 1
    integral_value = ci(n)
    while np.abs(integral_value) > 1e-4: 
        cis.append(integral_value)
        n += 1
        integral_value = ci(n)
    cis.append(integral_value)
            
    result = -np.sum([ci**2 for ci in cis]) / (4*np.pi**2)
    # assert np.imag(result) < 1e-1
    return np.sqrt(np.real(result)) 
