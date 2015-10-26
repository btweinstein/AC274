import numpy as np
import scipy as sp

# Setup the simulation

class Solver(object):

    def __init__(self, imax=100, jmax=100, kmax=10, dt=0.01, dr=1.,
                 u=None, v=None, D=5., s=1., fi_orig=None):

        self.imax = imax
        self.jmax = jmax
        self.kmax = kmax
        self.dt = dt
        self.dr = dr

        self.rgrid = dr*np.arange(imax)
        self.cgrid = dr*np.arange(jmax)
        self.tgrid = dt*np.arange(kmax)

        if v is None:
            self.v = 10.*np.ones((imax, jmax), dtype=np.double)
        else:
            self.v = v
        if u is None:
            self.u = 10.*np.ones((imax, jmax), dtype=np.double)
        else:
            self.u = u

        self.D =  D
        self.s = s
        if fi_orig is None:
            self.fi_orig = np.zeros((imax, jmax), dtype=np.double)
            self.fi_orig[imax/2, jmax/2] = 0.5
        else:
            self.fi_orig = fi_orig


        self.A = None
        self.zeta = None
        self.I = None
        # self.setup_matrices()

    def get_A(self):
        """Returns the advection operator"""

        max_logical_index = self.get_logical_index(self.imax, self.jmax)

        A = np.zeros((max_logical_index, max_logical_index), dtype=np.int) #TODO: Convert to sparse matrix!
        for i in range(self.imax):
            for j in range(self.jmax):


    def logical_dd(self, i1, j1, i2, j2):
        # A dirac delta that converts ij coordinates to logical index coordinates
        log1 = self.get_logical_index(i1, j1)
        log2 = self.get_logical_index(i2, j2)
        if log1 == log2:
            return 1
        else:
            return 0

    def get_logical_index(self, i, j):
        return i*self.jmax + j

    # def setup_matrices(self):
    #     # Define the advection operator
    #
    #     self.A = np.zeros((self.jmax, self.jmax), dtype=np.double)
    #     for r in range(self.A.shape[0]):
    #         for c in range(self.A.shape[1]):
    #             to_the_right = (c+1)%self.A.shape[1]
    #             to_the_left = (c-1)%self.A.shape[1]
    #             if r == to_the_right:
    #                 self.A[r, c] = self.v[r]/(2*self.dx)
    #             elif r == to_the_left:
    #                 self.A[r,c] = -self.v[r]/(2*self.dx)
    #
    #     # Define the diffusion operator
    #
    #     self.zeta = np.zeros((self.jmax, self.jmax), dtype=np.double)
    #     for r in range(self.zeta.shape[0]):
    #         for c in range(self.zeta.shape[1]):
    #             to_the_right = (c+1)%self.zeta.shape[1]
    #             to_the_left = (c-1)%self.zeta.shape[1]
    #             if r == to_the_right:
    #                 self.zeta[r, c] = self.D/self.dx**2
    #             elif r == to_the_left:
    #                 self.zeta[r,c] = self.D/self.dx**2
    #             elif r == c:
    #                 self.zeta[r, c] = -2*self.D/self.dx**2
    #
    #     # Define the identity operator
    #     self.I = np.identity(self.jmax, dtype=np.double)
    #
    # def run(self):
    #     sol_in_time = np.zeros((self.jmax, self.imax), dtype=np.double)
    #     sol_in_time[:, 0] = self.fi_orig
    #
    #     fi = np.array([self.fi_orig]).T
    #
    #     for i in range(self.imax):
    #         left_side = self.I - (self.dt/2.)*self.zeta - (self.dt/2.)*self.A
    #
    #         propagation = (self.I + (self.dt/2.)*self.A + (self.dt/2.)*self.zeta).dot(fi)
    #         growth = self.dt*self.s*fi*(1-fi)
    #         right_side = propagation + growth
    #
    #         fi_plus_1 = sp.sparse.linalg.bicgstab(left_side, right_side, x0=fi, tol=10.**-6)[0]
    #         sol_in_time[:, i] = fi_plus_1
    #
    #         fi = fi_plus_1
    #
    #     return sol_in_time