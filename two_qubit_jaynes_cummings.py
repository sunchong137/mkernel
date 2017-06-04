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
    propogate density matrix for t = [0, 2 pi, time_steps] by euler integration

    :param rho_initial: ndarray square matrix representing the density initial density matrix
    :param hamiltonian: ndarray square matrix represeniing the Hamiltonian
    :param Int time_steps: Total number of time steps to propogate the density matrix.
    """
    rhos = []
    dt = 2 * np.pi / time_steps
    rho = np.copy(rho_initial)
    for t in xrange(time_steps):
        rhos.append(rho)
        rho = rho + commutator(hamiltonian, rho) * dt

    return rhos

def commutator(operator_a, operator_b):
    """
    compute commutator of two operators
    """
    return -1j*(operator_a.dot(operator_b) - operator_b.dot(operator_a))


def trace_bath(rho):
    """
    Trace out the bath degrees of freedom of an operator

    The ordering is always bath indices then system indices so that the
    system is always indexed by the least significant digits

    :param rho: general bath + system operator
    :returns: system density operator
    """
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

def gen_one_qubit_superoperator(unitary):
    """
    Generaty the Kraus operator form of the super operator acting on a density matrix

    Kraus operators are formed by taking the expected value of the propogator with states all N^{2}
    states in the bath space.   Rememebr that most significant bits are the bath.  Least significant
    bits are the system
    """
    # states in the bath
    bath_states = [np.array([[1], [0]]), np.array([[0], [1]])]
    bath_state_pairs = product(bath_states, repeat=2)
    kraus_ops = []
    for bra, ket in bath_state_pairs:
        super_bra = np.kron(bra, np.eye(2))
        super_ket = np.kron(ket, np.eye(2))
        # not sure why the sqrt of 2 term is needed...my kraus operators seem
        # to be follow a completeness relation of 2 * I instead of just
        # I...weird...what don't I understand about the construction of these
        # operatersators.
        kraus_ops.append(np.conj(super_bra).T.dot(unitary).dot(super_ket)/np.sqrt(2))

    # check for closure
    assert np.allclose(sum(map(lambda x: np.conj(x).T.dot(x), kraus_ops)), np.eye(2))
    return kraus_ops

def generate_initial_densities(num_system_spins, num_bath_spins, random_seed=42):
    """
    Generate a non-entangled system bath density matrix as our initial state

    :param Int num_system_spins: number of spins in the `system' piece
    :param Int num_bath_spins: number of spins in the `bath' piece
    :returns: ndarray that is 2**(num_system_spins + num_bath_spins)
    """
    if num_system_spins != 1:
        raise ValueError("""Right now I'm only supporting a single spin in the system""")

    # Generate initial random density matrix
    np.random.seed(random_seed)
    vec = np.random.random(3)
    vec = vec/np.linalg.norm(vec)
    rho_system = np.array([[1 + vec[2], vec[0] - 1j*vec[1]],
                           [vec[0] + 1j*vec[1], 1 - vec[2]]]) * 0.5

    # Generate initial bath density matrix
    rho_bath = np.eye(2) * 0.5

    # Generate initial total system density matrices
    rho = np.kron(rho_bath, rho_system)
    assert np.isclose(np.trace(rho), 1.0)
    return rho, rho_system, rho_bath

def superoperator_map(rho, kraus_ops):
    """
    Map the density matrix via superoperator
    """
    rho_new = np.zeros_like(rho)
    for kraus in kraus_ops:
        rho_new += kraus.dot(rho).dot(np.conj(kraus).T)

    return rho_new


if __name__ == "__main__":

    alpha = 1.0
    # bath, system
    jc_driver = alpha * (np.kron(sigmam, sigmap) + np.kron(sigmap, sigmam))
    w, v = np.linalg.eigh(jc_driver)

    rho, rho_system, rho_bath = generate_initial_densities(1, 1)

    local_observable_x = []
    local_observable_y = []
    local_observable_z = []

    local_observable_x_bath = []
    local_observable_y_bath = []
    local_observable_z_bath = []

    local_observable_x_system = []
    local_observable_y_system = []
    local_observable_z_system = []

    local_observable_x_reduced_dynamics = []
    local_observable_y_reduced_dynamics = []
    local_observable_z_reduced_dynamics = []

    Xs = np.kron(I, X)
    Ys = np.kron(I, Y)
    Zs = np.kron(I, Z)

    Xb = np.kron(X, I)
    Yb = np.kron(Y, I)
    Zb = np.kron(Z, I)

    time_steps = 3000
    time = np.linspace(0, alpha * 2 * np.pi, time_steps)
    rhos_system = []
    for t in time:
        # the true propogator of the system.
        # expm will exponentiate eigenvalues the rotated back from eigenbasis
        propogator = expm(1j*t*jc_driver)

        # map the reduced dynamics by superoperator method (via Kraus)
        kraus_ops = gen_one_qubit_superoperator(propogator)
        rho_system_current = superoperator_map(rho_system, kraus_ops)
        local_observable_x_reduced_dynamics.append(np.trace(X.dot(rho_system_current)))
        local_observable_y_reduced_dynamics.append(np.trace(Y.dot(rho_system_current)))
        local_observable_z_reduced_dynamics.append(np.trace(Z.dot(rho_system_current)))


        rho_final = np.conj(propogator).T.dot(rho).dot(propogator)

        # collect local observables.
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

        rhos_system.append(trace_bath(rho_final))


        reduced_rho = np.zeros((2, 2), dtype=complex)
        reduced_rho[0, 0] = rho_final[0 * 2 + 0, 0 * 2 + 0] + rho_final[1 * 2 + 0, 1 * 2 + 0]
        reduced_rho[0, 1] = rho_final[0 * 2 + 0, 0 * 2 + 1] + rho_final[1 * 2 + 0, 1 * 2 + 1]
        reduced_rho[1, 0] = rho_final[0 * 2 + 1, 0 * 2 + 0] + rho_final[1 * 2 + 1, 1 * 2 + 0]
        reduced_rho[1, 1] = rho_final[0 * 2 + 1, 0 * 2 + 1] + rho_final[1 * 2 + 1, 1 * 2 + 1]

        assert np.allclose(reduced_rho, rhos_system[-1])
        print np.trace(rhos_system[-1].dot(X))
        local_observable_x_system.append(np.trace(rhos_system[-1].dot(X)))

    # integrate my system forward in time with euler integration
    rhos = euler_integrated(rho, jc_driver, time_steps)

    local_observable_x_euler = map(lambda x: np.trace(Xs.dot(x)), rhos)
    local_observable_y_euler = map(lambda x: np.trace(Ys.dot(x)), rhos)
    local_observable_z_euler = map(lambda x: np.trace(Zs.dot(x)), rhos)

    local_observable_x_subsystem = map(lambda x: np.trace(X.dot(x)), rhos_system)
    local_observable_y_subsystem = map(lambda x: np.trace(Y.dot(x)), rhos_system)
    local_observable_z_subsystem = map(lambda x: np.trace(Z.dot(x)), rhos_system)

    plt.plot(time[::30], local_observable_x[::30], 'C0-', label=r'$X_{s}$')
    plt.plot(time[::30], local_observable_y[::30], 'C1-', label=r'$Y_{s}$')
    plt.plot(time[::30], local_observable_z[::30], 'C2-', label=r'$Z_{s}$')

    # plt.plot(time[::30], local_observable_x_euler[::30], 'C0o', mfc='black', ms=3, label=r'$X_{s}\;\mathrm{Euler}$')
    # plt.plot(time[::30], local_observable_y_euler[::30], 'C1o', mfc='black', ms=3,label=r'$Y_{s}\;\mathrm{Euler}$')
    # plt.plot(time[::30], local_observable_z_euler[::30], 'C2o', mfc='black', ms=3, label=r'$Z_{s}\;\mathrm{Euler}$')

    assert np.allclose(local_observable_x_subsystem, local_observable_x)
    assert np.allclose(local_observable_y_subsystem, local_observable_y)
    assert np.allclose(local_observable_z_subsystem, local_observable_z)

    # plt.plot(time[::30], local_observable_x_subsystem[::30], 'C0^', mfc=None, ms=3, label=r'$X_{s}\;\mathrm{subsystem}$')
    # plt.plot(time[::30], local_observable_y_subsystem[::30], 'C1^', mfc=None, ms=3,label=r'$Y_{s}\;\mathrm{subsystem}$')
    # plt.plot(time[::30], local_observable_z_subsystem[::30], 'C2^', mfc=None, ms=3, label=r'$Z_{s}\;\mathrm{subsystem}$')

    plt.plot(time[::30], local_observable_x_reduced_dynamics[::30], 'C0D', mfc=None, ms=3, label=r'$X_{s}\;\mathrm{reduced}$')
    plt.plot(time[::30], local_observable_y_reduced_dynamics[::30], 'C1D', mfc=None, ms=3,label=r'$Y_{s}\;\mathrm{reduced}$')
    plt.plot(time[::30], local_observable_z_reduced_dynamics[::30], 'C2D', mfc=None, ms=3, label=r'$Z_{s}\;\mathrm{reduced}$')



    plt.legend(loc='lower right')
    plt.show()
