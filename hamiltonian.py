import numpy as np
from referenceqvm.gates import X, Y, Z, I


def get_operator():
    return np.kron(Z, Z)

def expected_value(rho):
    """
    Returns Hamiltonian expectation and drho/dt

    """
    if not isinstance(rho, np.ndarray):
        raise TypeError("""I only take ndarray""")
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("""RHO IS NOT SQUARE. WHAT HAPPENED?""")

    ham = get_operator()
    return np.trace(ham.dot(rho))


def gradient(rho):
    """
    blah
    """
    if not isinstance(rho, np.ndarray):
        raise TypeError("""I only take ndarray""")
    if rho.shape[0] != rho.shape[1]:
        raise ValueError("""RHO IS NOT SQUARE. WHAT HAPPENED?""")

    ham = get_operator()
    return 1j* (ham.dot(rho) - rho.dot(ham))
