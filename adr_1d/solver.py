import numpy as np
import scipy as sp

# Setup the simulation

class Solver(object):

    def __init__(self, imax=100, jmax=1000, dt=0.01, dx=1.,
                 v = None, D=5., s=1., fi_orig=None):

        self.imax = imax
        self.jmax = jmax
        self.dt = dt
        self.dx = dx

        self.xgrid = dx*np.arange(jmax)
        self.tgrid = dt*np.arange(imax)

        if v is None:
            self.v = 10.*np.ones(jmax, dtype=np.double)
        else:
            self.v = v
        self.D =  D
        self.s = s
        if fi_orig is None:
            self.fi_orig = sp.stats.norm(loc=dx*jmax/2, scale=50*dx).pdf(self.xgrid)
        else:
            self.fi_orig = fi_orig


        self.A = None
        self.zeta = None
        self.I = None
        self.setup_matrices()

    def setup_matrices(self):
        # Define the advection operator

        self.A = np.zeros((self.jmax, self.jmax), dtype=np.double)
        for r in range(self.A.shape[0]):
            for c in range(self.A.shape[1]):
                to_the_right = (c+1)%self.A.shape[1]
                to_the_left = (c-1)%self.A.shape[1]
                if r == to_the_right:
                    self.A[r, c] = self.v[r]/(2*self.dx)
                elif r == to_the_left:
                    self.A[r,c] = -self.v[r]/(2*self.dx)

        # Define the diffusion operator

        self.zeta = np.zeros((self.jmax, self.jmax), dtype=np.double)
        for r in range(self.zeta.shape[0]):
            for c in range(self.zeta.shape[1]):
                to_the_right = (c+1)%self.zeta.shape[1]
                to_the_left = (c-1)%self.zeta.shape[1]
                if r == to_the_right:
                    self.zeta[r, c] = self.D/self.dx**2
                elif r == to_the_left:
                    self.zeta[r,c] = self.D/self.dx**2
                elif r == c:
                    self.zeta[r, c] = -2*self.D/self.dx**2

        # Define the identity operator
        self.I = np.identity(self.jmax, dtype=np.double)

    def run(self):
        sol_in_time = np.zeros((self.jmax, self.imax), dtype=np.double)
        sol_in_time[:, 0] = self.fi_orig

        fi = np.array([self.fi_orig]).T

        for i in range(self.imax):
            inv = np.linalg.inv(self.I - (self.dt/2.)*self.zeta)
            propagation = (self.I + self.dt*self.A + (self.dt/2.)*self.zeta).dot(fi)
            growth = self.dt*self.s*fi*(1-fi)

            fi_plus_1 = inv.dot(propagation + growth)
            sol_in_time[:, i] = fi_plus_1[:, 0]

            fi = fi_plus_1

        return sol_in_time