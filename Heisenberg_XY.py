#author: Chong Sun, May 27, 2016
#Reference: PHYSICAL REVIEW B 70, 045323 (2004), "The theory of Open quantum systems"
#Interaction picture is used.
#Fragment-bath interaction Hamiltonian:
#############################################
#####   aH = 2a( s_+ J_- + s_- J_+ )    #####
#############################################
#The time evolution is based on Nakajima-Zwanzig GQME
#2 spin system: rho_s x rho_b

import numpy as np
import matplotlib.pyplot as plt
import utils

def get_hamiltonian(alpha=0.5):
    sp = np.array([[0, 1], [0, 0]], dtype=np.complex)
    sm = np.array([[0, 0], [1, 0]], dtype=np.complex)
    ham = 2 * alpha * (np.kron(sp,sm) + np.kron(sm,sp))
    return ham

def closed_evolution(rho, rho_bath, alpha=0.5, npoint=100):
    # get P(rho) in time range (0, 2pi)
    ham = get_hamiltonian(alpha=alpha)
    rho_n = utils.projectP(rho, rho_bath)
    rho_array = []
    dt = 2*np.pi/npoint
    for t in xrange(npoint):
        rho_array.append(rho_n)
        incre = utils.projectP(utils.liouville(ham, rho_n), rho_bath)
        rho_n = rho_n +  dt * incre
        rho_array.append(rho_n)

    rho_array=np.asarray(rho_array)
    return rho_array

def plot_properties(rho_array):
    npoint = len(rho_array)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex)/2.
    Y = np.array([[0, 1], [-1, 0]], dtype=np.complex)/2.j
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex)/2.
    sx, sy, sz = [], [], []
    tarray = xrange(npoint)

    for t in tarray:
        rhos = utils.trace_bath(rho_array[t])
        sx.append(np.trace(X.dot(rhos)))
        sy.append(np.trace(Y.dot(rhos)))
        sz.append(np.trace(Z.dot(rhos)))
    plt.plot(tarray, sz, c='r')
    plt.plot(tarray, sx, c='b')
    plt.plot(tarray, sy, c='g')
    plt.show()

#    return np.array(sx), np.array(sy), np.array(sz)

if __name__ == "__main__":
    vec = np.random.random(3)
    vec = vec/np.linalg.norm(vec)
    rho_system = np.array([[1 + vec[2], vec[0] - 1j*    vec[1]],
                           [vec[0] + 1j*vec[1], 1 -     vec[2]]]) * 0.5
    rho_bath = np.eye(2) * 0.5
    rho = np.kron(rho_system, rho_bath)
    rho_array=closed_evolution(rho,rho_bath)
    plot_properties(rho_array)
