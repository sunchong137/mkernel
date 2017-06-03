import numpy as np
from scipy.linalg import expm
from referenceqvm.gates import X, Y, I, Z

import matplotlib.pyplot as plt
import sys
from itertools import product

sigmap = 0.5 * (X - 1j*Y)
sigmam = 0.5 * (X + 1j*Y)

def euler_integrated(rho_initial, hamiltonian, time_steps):
    """
    propogate density matrix for t = [0, 2 pi, 100]
    """
    rhos = []
    dt = 2 * np.pi / time_steps
    rho = np.copy(rho_initial)
    for t in xrange(time_steps):
        rhos.append(rho)
        rho = rho  + commutator(hamiltonian, rho) * dt

    return rhos

def commutator(operator_a, operator_b):
    """
    compute commutator of two operators
    """
    return -1j*(operator_a.dot(operator_b) - operator_b.dot(operator_a))


def trace_bath(rho):
    rho_total = rho.reshape((2, 2, 2, 2))
    test_system_density = np.einsum('jijk', rho_total)

    rho_nick = np.zeros((2, 2, 2, 2), dtype=complex)
    for p, q, i, j in product(range(2), repeat=4):
        rho_nick[p, q, i, j] = rho[p * 2 + q, i * 2 + j]

    assert np.allclose(rho_total, rho_nick)

    I = np.eye(2)
    state0 = np.array([1., 0.])
    state1 = np.array([0., 1.])
    t0 = np.kron(state0, I)
    t1 = np.kron(state1, I)
    system_density = np.dot(t0, np.dot(rho, t0.T.conj())) + np.dot(t1, np.dot(rho, t1.T.conj()))

    if not np.allclose(test_system_density, system_density):
        raise AssertionError("""Total density contracted incorrectly""")

    constructed_system_density = np.zeros((2, 2), dtype=complex)
    constructed_system_density[0, 0] = rho_nick[0, 0, 0, 0] + rho_nick[1, 0, 1, 0]
    constructed_system_density[0, 1] = rho_nick[0, 0, 0, 1] + rho_nick[1, 0, 1, 1]
    constructed_system_density[1, 0] = rho_nick[0, 1, 0, 0] + rho_nick[1, 1, 1, 0]
    constructed_system_density[1, 1] = rho_nick[0, 1, 0, 1] + rho_nick[1, 1, 1, 1]
    assert np.allclose(constructed_system_density, system_density)

    return system_density

if __name__ == "__main__":

    alpha = 1.0
    # bath, system
    jc_driver = alpha * (np.kron(sigmam, sigmap) + np.kron(sigmap, sigmam))
    w, v = np.linalg.eigh(jc_driver)

    # np.random.seed(42)
    vec = np.random.random(3)
    vec = vec/np.linalg.norm(vec)
    rho_system = np.array([[1 + vec[2], vec[0] - 1j*vec[1]],
                           [vec[0] + 1j*vec[1], 1 - vec[2]]]) * 0.5

    rho_bath = np.eye(2) * 0.5

    rho = np.kron(rho_bath, rho_system)

    local_observable_x = []
    local_observable_y = []
    local_observable_z = []

    local_observable_x_bath = []
    local_observable_y_bath = []
    local_observable_z_bath = []

    local_observable_x_system = []
    local_observable_y_system = []
    local_observable_z_system = []


    Xs = np.kron(I, X)
    Ys = np.kron(I, Y)
    Zs = np.kron(I, Z)

    Xb = np.kron(X, I)
    Yb = np.kron(Y, I)
    Zb = np.kron(Z, I)

    time_steps = 100
    time = np.linspace(0, alpha * 2 * np.pi, time_steps)
    rho_system = []
    for t in time:
        propogator = expm(1j*t*jc_driver)
        rho_final = np.conj(propogator).T.dot(rho).dot(propogator)
        local_observable_x.append(np.trace(Xs.dot(rho_final)))
        local_observable_y.append(np.trace(Ys.dot(rho_final)))
        local_observable_z.append(np.trace(Zs.dot(rho_final)))

        local_observable_x_bath.append(np.trace(Xb.dot(rho_final)))
        local_observable_y_bath.append(np.trace(Yb.dot(rho_final)))
        local_observable_z_bath.append(np.trace(Zb.dot(rho_final)))

        # check my local_observable
        x_check = rho_final[0 * 2 + 0, 0 * 2 + 1] + rho_final[1 * 2 + 0, 1 * 2 + 1]
        x_check += rho_final[0 * 2 + 1, 0 * 2 + 0] + rho_final[1 * 2 + 1, 1 * 2 + 0]
        print x_check, local_observable_x[-1],
        assert np.isclose(x_check, local_observable_x[-1])

        rho_system.append(trace_bath(rho_final))

        reduced_rho = np.zeros((2, 2), dtype=complex)
        reduced_rho[0, 0] = rho_final[0 * 2 + 0, 0 * 2 + 0] + rho_final[1 * 2 + 0, 1 * 2 + 0]
        reduced_rho[0, 1] = rho_final[0 * 2 + 0, 0 * 2 + 1] + rho_final[1 * 2 + 0, 1 * 2 + 1]
        reduced_rho[1, 0] = rho_final[0 * 2 + 1, 0 * 2 + 0] + rho_final[1 * 2 + 1, 1 * 2 + 0]
        reduced_rho[1, 1] = rho_final[0 * 2 + 1, 0 * 2 + 1] + rho_final[1 * 2 + 1, 1 * 2 + 1]

        assert np.allclose(reduced_rho, rho_system[-1])
        print np.trace(rho_system[-1].dot(X))
        local_observable_x_system.append(np.trace(rho_system[-1].dot(X)))

    rhos = euler_integrated(rho, jc_driver, time_steps)


    local_observable_x_euler = map(lambda x: np.trace(Xs.dot(x)), rhos)
    local_observable_y_euler = map(lambda x: np.trace(Ys.dot(x)), rhos)
    local_observable_z_euler = map(lambda x: np.trace(Zs.dot(x)), rhos)

    local_observable_x_subsystem = map(lambda x: np.trace(X.dot(x)), rho_system)
    local_observable_y_subsystem = map(lambda x: np.trace(Y.dot(x)), rho_system)
    local_observable_z_subsystem = map(lambda x: np.trace(Z.dot(x)), rho_system)

    plt.plot(time, local_observable_x, 'C0-', label=r'$X_{s}$')
    plt.plot(time, local_observable_y, 'C1-', label=r'$Y_{s}$')
    plt.plot(time, local_observable_z, 'C2-', label=r'$Z_{s}$')

    plt.plot(time, local_observable_x_euler, 'C0o', mfc=None, ms=3, label=r'$X_{s}\;\mathrm{Euler}$')
    plt.plot(time, local_observable_y_euler, 'C1o', mfc=None, ms=3,label=r'$Y_{s}\;\mathrm{Euler}$')
    plt.plot(time, local_observable_z_euler, 'C2o', mfc=None, ms=3, label=r'$Z_{s}\;\mathrm{Euler}$')

    assert np.allclose(local_observable_x_subsystem, local_observable_x)
    assert np.allclose(local_observable_y_subsystem, local_observable_y)
    assert np.allclose(local_observable_z_subsystem, local_observable_z)

    # plt.plot(time, local_observable_x_subsystem, 'C0^', mfc=None, ms=3, label=r'$X_{s}\;\mathrm{subsystem}$')
    # plt.plot(time, local_observable_y_subsystem, 'C1^', mfc=None, ms=3,label=r'$Y_{s}\;\mathrm{subsystem}$')
    # plt.plot(time, local_observable_z_subsystem, 'C2^', mfc=None, ms=3, label=r'$Z_{s}\;\mathrm{subsystem}$')

    plt.legend(loc='lower right')
    plt.show()
