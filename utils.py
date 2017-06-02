# author: Chong Sun 05/26/2017
import numpy as np


def trace_bath(rho):
    # trace off bath from rho
    # rho = rho_s x rho_b
    Imat = np.eye(2)
    bath0 = np.kron(Imat, np.array([1.,0.]))
    bath1 = np.kron(Imat, np.array([0.,1.]))
    #bath0 = np.kron(np.array([1.,0.]), Imat)
    #bath1 = np.kron(np.array([0.,1.]), Imat)
    return np.dot(bath0, np.dot(rho, bath0.T.conj())) + \
           np.dot(bath1, np.dot(rho, bath1.T.conj()))

def euler_integrated(rho_initial, hamiltonian, Nstep=100):
    # propogate rho
    rhos = []
    dt = 2*np.pi/Nstep
    rho = rho_initial.copy()
    for t in xrange(100):
        rhos.append(rho)
        rho += dt*commutator(hamiltonian, rho)

    return rhos

def liouville(H, O):
    return -1.j*(H.dot(O) - O.dot(H))

def projectP(rho, bath_rho):
    return np.kron(trace_bath(rho), bath_rho)

def projectQ(rho, bath_rho):
    matI = np.eye(rho.shape)
    return matI-projectP
