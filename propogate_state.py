import numpy as np
from scipy.integrate import ode
from referenceqvm.gates import H

from hamiltonian import expected_value, gradient
from referenceqvm.gates import Z, I, X

from referenceqvm.reference import Connection

from pyquil.quil import Program
from pyquil.paulis import trotterize

from scipy.linalg import expm

if __name__ == "__main__":

    num_qubits = 1
    state = np.ones((2**num_qubits, 1)) * (1.0/np.sqrt(2))
    rho_initial = state.dot(state.T)

    rqvm = Connection(type='unitary')

    expected_z = []
    for t in np.linspace(0, 2 * np.pi, 100):
        unitary = expm(1j*t*Z)
        rho = np.conj(unitary).T.dot(rho_initial).dot(unitary)
        print rho
        expected_z.append(np.trace(X.dot(rho)))

    # r = ode(expected_value, gradient).set_integrator('dopri5')
    # t0 = 0.0
    # dt = 0.001
    # r.set_initial_value(rho_initial, t0)
    # while r.t < 10:
    #     print r.t + dt, r.integrate(r.t + dt)
