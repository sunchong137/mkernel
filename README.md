Memory Kernel Project
---------------------

Task [1] Integrate a one-qubit system in time.  We should observe oscillations
corresponding to the Rabi frequency.

Task [2] Integrate a two-qubit system in time.

Procedure:

A) Integrator selection. Scipy.integrate.ode(f, jac=fprime).set_integrator('dopri5')

B) Initial state rho = |psi><psi|  |psi> = 1/sqrt(2) * np.array([[1], [1]])

C) f = Tr[H.rho], fprime = -i/hbar * [H, rho]

D) Observables to track <Z>, <X>, <Y>
