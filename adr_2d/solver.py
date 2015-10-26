import numpy as np
import scipy as sp
from morton import zorder

# Setup the simulation

class Solver(object):

    def __init__(self, imax=10, jmax=10, kmax=20, dt=0.1, dr=1.0,
                 u=None, v=None, D=1., s=0.3, fi_orig=None, use_morton=True):

        self.imax = imax
        self.jmax = jmax
        self.kmax = kmax
        self.dt = dt
        self.dr = dr

        self.use_morton = use_morton

        self.rgrid = dr*np.arange(imax)
        self.cgrid = dr*np.arange(jmax)
        self.tgrid = dt*np.arange(kmax)

        if v is None:
            self.v = 1.*np.ones((imax, jmax), dtype=np.double)
        else:
            self.v = v
        if u is None:
            self.u = 1.*np.ones((imax, jmax), dtype=np.double)
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

        # Setup the logical indices
        self.logical_index_mat = self.get_logical_index_matrix()
        # Convert the logical indices into a matrix
        self.position_to_logical_dict = {}
        for i in range(self.logical_index_mat.shape[0]):
            for j in range(self.logical_index_mat.shape[1]):
                self.position_to_logical_dict[i, j] = self.logical_index_mat[i, j]
        # Invert the list
        self.logical_to_position_dict = {v: k for k, v in self.position_to_logical_dict.items()}

        self.A = self.get_A()
        self.zeta = self.get_zeta()
        self.I = np.identity(self.logical_index_mat.max() + 1, dtype=np.double)
        # self.setup_matrices()

    def get_logical_index_matrix(self):
        index_mat = np.arange(self.imax * self.jmax).reshape((self.imax, self.jmax))
        if self.use_morton:
            zorder(index_mat)
        # We could do more complex things like morton ordering, but will keep it simple
        return index_mat

    def get_A(self):
        """Returns the advection operator"""

        max_logical_index = self.logical_index_mat.max() + 1

        A = np.zeros((max_logical_index, max_logical_index), dtype=np.double) #TODO: Convert to sparse matrix!
        for r in range(max_logical_index):
            i1, j1 = self.logical_to_position_dict[r]
            for c in range(max_logical_index):
                i2, j2 = self.logical_to_position_dict[c]

                uij = self.u[r]
                vij = self.v[r]

                first_term = (uij/(2.*self.dr))*(self.dd(i1 + 1, j1, i2, j2) - self.dd(i1 - 1, j1, i2, j2))
                second_term = (vij/(2*self.dr))*(self.dd(i1, j1+1,i2,j2) - self.dd(i1,j1-1, i2, j2))
                A[r, c] = first_term + second_term

        return A

    def get_zeta(self):
        max_logical_index = self.logical_index_mat.max() + 1

        a_stencil = 4./36.
        b_stencil = 1./36.
        c_stencil = -20./36. # Be careful, a, b, and c may appear in loops

        zeta = np.zeros((max_logical_index, max_logical_index), dtype=np.double) #TODO: Convert to sparse matrix!
        for r in range(max_logical_index):
            i1, j1 = self.logical_to_position_dict[r]
            for c in range(max_logical_index):
                i2, j2 = self.logical_to_position_dict[c]

                first_term = self.dd(i1+1,j1+1, i2, j2) + \
                             self.dd(i1+1,j1-1, i2, j2) + \
                             self.dd(i1-1, j1-1, i2, j2) + \
                             self.dd(i1-1, j1+1, i2, j2)
                first_term *= b_stencil

                second_term = self.dd(i1, j1+1, i2, j2) + \
                              self.dd(i1+1,j1,i2,j2) + \
                              self.dd(i1,j1-1,i2,j2) + \
                              self.dd(i1-1,j1,i2,j2)

                second_term *= a_stencil

                third_term = c_stencil*self.dd_1d(r, c)

                zeta[r, c] = (self.D/self.dr**2)*(first_term + second_term + third_term)

        return zeta

    def dd_1d(self, i, j):
        """Standrad dirac delta."""
        if i ==j:
            return 1
        return 0

    def dd(self, i1, j1, i2, j2):
        """A dirac delta that makes sure i's and j's are the same."""
        if (i1== i2) and (j1 == j2):
            return 1
        else:
            return 0

    def convert_fi_real_to_logical(self, fi):
        fi_logical = np.zeros((self.logical_index_mat.max() + 1, 1))
        for i in range(fi.shape[0]):
            for j in range(fi.shape[1]):
                fi_logical[self.position_to_logical_dict[i, j], 0] = fi[i, j]
        return fi_logical

    def convert_fi_logical_to_real(self, fi):
        real_space = np.zeros((self.imax, self.jmax), dtype=np.double)
        for i in range(fi.shape[0]):
            real_space[self.logical_to_position_dict[i]] = fi[i]
        return real_space


    def run(self):
        sol_in_time = np.zeros((self.imax , self.jmax, self.kmax), dtype=np.double)
        sol_in_time[:, :, 0] = self.fi_orig
        # We need to convert the original solution to the new form.
        fi = self.convert_fi_real_to_logical(self.fi_orig)

        for i in range(self.imax - 1):
            left_side = self.I - (self.dt/2.)*self.zeta - (self.dt/2.)*self.A

            propagation = (self.I + (self.dt/2.)*self.A + (self.dt/2.)*self.zeta).dot(fi)
            growth = self.dt*self.s*fi*(1-fi)
            right_side = propagation + growth

            fi_plus_1 = sp.sparse.linalg.bicgstab(left_side, right_side, x0=fi, tol=10.**-6)[0]

            # Now get the solution in space
            sol_in_time[:, :, i + 1] = self.convert_fi_logical_to_real(fi_plus_1)

            fi = fi_plus_1

        return sol_in_time

if __name__=='__main__':
    # Test script
    sol = Solver(imax=10, jmax=10, use_morton=False)
    print sol.run()
