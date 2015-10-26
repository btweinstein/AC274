import numpy as np
import scipy as sp

# Setup the simulation

class Solver(object):

    def __init__(self, imax=10, jmax=10, kmax=20, dt=0.01, dr=1.0,
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

        # Now convert u and v to logical indices! We are using a simple technique to do this; it could be time
        # consuming to do this in general for an arbitrary index
        self.v = self.v.ravel() # In c order automatically, which matches our logical indexing
        self.u = self.u.ravel()

        self.D =  D
        self.s = s
        if fi_orig is None:
            self.fi_orig = np.zeros((imax, jmax), dtype=np.double)
            self.fi_orig[imax/2, jmax/2] = 0.1
        else:
            self.fi_orig = fi_orig


        self.A = self.get_A()
        self.zeta = None
        self.I = None
        # self.setup_matrices()

    def get_A(self):
        """Returns the advection operator"""

        max_logical_index = self.get_logical_index(self.imax, self.jmax)

        A = np.zeros((max_logical_index, max_logical_index), dtype=np.double) #TODO: Convert to sparse matrix!
        for i in range(self.imax):
            for j in range(self.jmax):
                uij = self.u[self.get_logical_index(i, j)]
                vij = self.v[self.get_logical_index(i, j)]

                first_term = (uij/(2.*self.dr))*(self.logical_dd(i, j, i+1, j) - self.logical_dd(i, j, i-1, j))
                print self.logical_dd(i, j, i+1, j)
                second_term = (vij/(2*self.dr))*(self.logical_dd(i,j,i,j+1) - self.logical_dd(i,j,i,j-1))
                #print (vij/(2*self.dr))
                A[i, j] = first_term + second_term

        return A

    def logical_dd(self, r, c, irow, jrow, icol, jcol):
        # A dirac delta that converts ij coordinates to logical index coordinates
        desired_row = self.get_logical_index(irow, jrow)
        desired_col = self.get_logical_index(icol, jcol)
        if (r == desired_row) and (c == desired_col):
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