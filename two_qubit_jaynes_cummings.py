import numpy as np
from scipy.linalg import expm
from referenceqvm.gates import X, Y, I, Z

import matplotlib.pyplot as plt

sigmap = 0.5 * (X - 1j*Y)
sigmam = 0.5 * (X + 1j*Y)

def euler_integrated(rho_initial, hamiltonian):
    """
    propogate density matrix for t = [0, 2 pi, 100]
    """
    rhos = []
    dt = 2 * np.pi / 100
    rho = np.copy(rho_initial)
    for t in xrange(100):
        rhos.append(rho)
        rho = rho + dt * commutator(hamiltonian, rho)

    return rhos

def commutator(operator_a, operator_b):
    """
    compute commutator of two operators
    """
    return operator_a.dot(operator_b) - operator_b.dot(operator_a)


def trace_bath(rho):
    I = np.eye(2)
    state0 = np.array([1., 0.])
    state1 = np.array([0., 1.])
    t0 = np.kron(state0, I)
    t1 = np.kron(state1, I)
    return np.dot(t0, np.dot(rho, t0.T.conj())) + np.dot(t1, np.dot(rho, t1.T.conj()))

if __name__ == "__main__":

    alpha = 1.0
    jc_driver = alpha * (np.kron(sigmam, sigmap) + np.kron(sigmap, sigmam))
    w, v = np.linalg.eigh(jc_driver)

    np.random.seed(42)
    vec = np.random.random(3)
    vec = vec/np.linalg.norm(vec)
    rho_system = np.array([[1 + vec[2], vec[0] - 1j*vec[1]],
                           [vec[0] + 1j*vec[1], 1 - vec[2]]]) * 0.5

    rho_bath = np.eye(2) * 0.5

    rho = np.kron(rho_bath, rho_system)

    local_observable_x = []
    local_observable_y = []
    local_observable_z = []
    Xs = np.kron(I, X)
    Ys = np.kron(I, Y)
    Zs = np.kron(I, Z)
    time = np.linspace(0, alpha * 2 * np.pi, 100)
    for t in time:
        propogator = expm(1j*t*jc_driver)
        rho_final = np.conj(propogator).T.dot(rho).dot(propogator)
        local_observable_x.append(np.trace(Xs.dot(rho_final)))
        local_observable_y.append(np.trace(Ys.dot(rho_final)))
        local_observable_z.append(np.trace(Zs.dot(rho_final)))

    rhos = euler_integrated(rho, jc_driver)
    local_observable_x_euler = map(lambda x: np.trace(Xs.dot(x)), rhos)
    local_observable_y_euler = map(lambda x: np.trace(Ys.dot(x)), rhos)
    local_observable_z_euler = map(lambda x: np.trace(Zs.dot(x)), rhos)

    plt.plot(time, local_observable_x, 'C0-', label=r'$X_{s}$')
    plt.plot(time, local_observable_y, 'C1-', label=r'$Y_{s}$')
    plt.plot(time, local_observable_z, 'C2-', label=r'$Z_{s}$')

    plt.plot(time, local_observable_x, 'C0o', mfc=None, ms=3, label=r'$X_{s}\;\mathrm{Euler}$')
    plt.plot(time, local_observable_y, 'C1o', mfc=None, ms=3,label=r'$Y_{s}\;\mathrm{Euler}$')
    plt.plot(time, local_observable_z, 'C2o', mfc=None, ms=3, label=r'$Z_{s}\;\mathrm{Euler}$')

    plt.legend(loc='lower right')
    plt.show()

