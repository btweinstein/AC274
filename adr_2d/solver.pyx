#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
import scipy as sp
from libc.math cimport fabs
import skimage as ski
import zorder
import seaborn as sns
import matplotlib.pyplot as plt

# Setup the simulation

cdef double TOLERANCE = 10.**-11

cdef int dd_1d(int i, int j) nogil:
        """Standrad dirac delta."""
        if i ==j:
            return 1
        return 0

cdef int dd(int i1, int j1, int i2, int j2) nogil:
        """A dirac delta that makes sure i's and j's are the same."""
        if (i1== i2) and (j1 == j2):
            return 1
        else:
            return 0

cdef long c_pos_mod(long num1, long num2) nogil:
    if num1 < 0:
        return num1 + num2
    else:
        return num1 % num2

class Solver(object):

    def __init__(self, imax=10, jmax=10, kmax=20, dt=0.01, dr=1.0,
                 u=None, v=None, D=10., s=0.8, fi_orig=None, use_morton=False, covX=None, covY = None):

        self.imax = imax
        self.jmax = jmax
        self.kmax = kmax
        self.dt = dt
        self.dr = dr

        self.use_morton = use_morton

        self.rgrid = dr*np.arange(imax)
        self.cgrid = dr*np.arange(jmax)
        self.tgrid = dt*np.arange(kmax)

        # Setup the logical indices
        self.logical_index_mat = self.get_logical_index_matrix()
        # Convert the logical indices into a matrix
        self.position_to_logical_dict = {}
        for i in range(self.logical_index_mat.shape[0]):
            for j in range(self.logical_index_mat.shape[1]):
                self.position_to_logical_dict[i, j] = self.logical_index_mat[i, j]
        # Invert the list
        self.logical_to_position_dict = {v: k for k, v in self.position_to_logical_dict.items()}


        if v is None: # v is right/left
            self.v = 50.*np.ones((imax, jmax), dtype=np.double)
        else:
            self.v = v
        if u is None: # u is down/up
            self.u = 50.*np.ones((imax, jmax), dtype=np.double)
        else:
            self.u = u

        # Now convert u and v to logical indices! We are using a simple technique to do this; it could be time
        # consuming to do this in general for an arbitrary index
        self.v = self.convert_fi_real_to_logical(self.v)
        self.v = self.v[:, 0] # Convert to a 1d list, don't need the vector
        self.u = self.convert_fi_real_to_logical(self.u)
        self.u = self.u[:, 0]

        self.D =  D
        self.s = s
        print 'Creating initial gaussian condition...'
        if fi_orig is None:
            self.fi_orig = np.zeros((imax, jmax), dtype=np.double)
            if covX is None:
                covX = self.imax/30
            if covY is None:
                covY = self.jmax/30
            dist = sp.stats.multivariate_normal(mean=[self.imax/2, self.jmax/2], cov=[[covX,0],[0,covY]])
            for i in range(self.fi_orig.shape[0]):
                for j in range(self.fi_orig.shape[1]):
                    self.fi_orig[i, j] = dist.pdf([i, j])
            self.fi_orig = .1*self.fi_orig/self.fi_orig.max()
        else:
            self.fi_orig = fi_orig
        print 'Done!'

        print 'Creating advection operator...'
        self.A = self.get_A()
        print 'Done!'
        print 'Creating diffusion operator...'
        self.zeta = self.get_zeta()
        print 'Done!'

        self.I = sp.sparse.eye(self.logical_index_mat.max() + 1, dtype=np.double, format='csc')

    def get_logical_index_matrix(self):

        index_mat = np.arange(self.imax * self.jmax).reshape((self.imax, self.jmax))
        if self.use_morton:
            zorder.zorder(index_mat)
        assert np.unique(index_mat).shape[0] == self.imax*self.jmax # Ensure that the indexing does not remove elements...
        # We could do more complex things like morton ordering, but will keep it simple
        return index_mat

    def get_A(self):
        """Returns the advection operator"""

        cdef int max_logical_index = self.logical_index_mat.max() + 1

        A = sp.sparse.lil_matrix((max_logical_index, max_logical_index), dtype=np.double)
        cdef int r, c
        cdef int i1, j1
        cdef int i2, j2

        cdef double[:] u = self.u
        cdef double[:] v = self.v
        cdef double uij, vij
        cdef double first_term, second_term, result

        cdef double dr = self.dr

        cdef int ip1, im1, jp1, jm1

        cdef int imax = self.imax
        cdef int jmax = self.jmax

        cdef dict position_to_logical = self.position_to_logical_dict

        cdef int[:] i_possible = np.zeros(3, dtype=np.intc)
        cdef int[:] j_possible = np.zeros(3, dtype=np.intc)

        cdef int i_count, j_count

        cdef float uip1, uim1, vjp1, vjm1

        # Loop over space...as the real matrix is *way* larger
        for i1 in range(imax):
            for j1 in range(jmax):
                r = position_to_logical[i1, j1]
                # Get neighbors
                ip1 = c_pos_mod(i1 + 1, imax)
                im1 = c_pos_mod(i1 - 1, imax)
                jp1 = c_pos_mod(j1 + 1, jmax)
                jm1 = c_pos_mod(j1 - 1, jmax)

                # Loop over all possible stencils
                i_possible[0] = ip1
                i_possible[1] = i1
                i_possible[2] = im1

                j_possible[0] = jp1
                j_possible[1] = j1
                j_possible[2] = jm1



                uip1 = u[position_to_logical[ip1, j1]]
                uim1 = u[position_to_logical[im1, j1]]
                vjp1 = v[position_to_logical[i1, jp1]]
                vjm1 = v[position_to_logical[i1, jm1]]

                for i_count in range(3):
                    i2 = i_possible[i_count]
                    for j_count in range(3):
                        j2 = j_possible[j_count]

                        # We must be careful here...mod's in c don't become positive which leads to bad things

                        first_term = 1./(2.*dr)*(uip1*dd(ip1, j1, i2, j2) - uim1*dd(im1, j1, i2, j2))
                        second_term = 1./(2.*dr)*(vjp1*dd(i1, jp1,i2,j2) - vjm1*dd(i1,jm1, i2, j2))

                        result = first_term + second_term

                        if fabs(result) > TOLERANCE:
                            c = position_to_logical[i2, j2]
                            A[r, c] = first_term + second_term
        return sp.sparse.csc_matrix(A)

    def get_zeta(self):
        cdef int max_logical_index = self.logical_index_mat.max() + 1

        cdef double a_stencil = 4./36.
        cdef double b_stencil = 1./36.
        cdef double c_stencil = -20./36. # Be careful, a, b, and c may appear in loops

        zeta = sp.sparse.lil_matrix((max_logical_index, max_logical_index), dtype=np.double)
        cdef int r, c
        cdef int i1, j1, i2, j2
        cdef double first_term, second_term, third_term, result

        cdef double D = self.D
        cdef double dr = self.dr

        cdef int imax = self.imax
        cdef int jmax = self.jmax

        cdef int ip1, im1, jp1, jm1

        cdef dict position_to_logical = self.position_to_logical_dict

        cdef int[:] i_possible = np.zeros(3, dtype=np.intc)
        cdef int[:] j_possible = np.zeros(3, dtype=np.intc)

        cdef int i_count, j_count

        for i1 in range(imax):
            for j1 in range(jmax):
                r = position_to_logical[i1, j1]
                # Get neighbors
                ip1 = c_pos_mod(i1 + 1, imax)
                im1 = c_pos_mod(i1 - 1, imax)
                jp1 = c_pos_mod(j1 + 1, jmax)
                jm1 = c_pos_mod(j1 - 1, jmax)

                # Loop over all possible stencils
                i_possible[0] = ip1
                i_possible[1] = i1
                i_possible[2] = im1

                j_possible[0] = jp1
                j_possible[1] = j1
                j_possible[2] = jm1

                for i_count in range(3):
                    i2 = i_possible[i_count]
                    for j_count in range(3):
                        j2 = j_possible[j_count]

                        first_term = dd(ip1,jp1, i2, j2) + \
                                     dd(ip1,jm1, i2, j2) + \
                                     dd(im1, jm1, i2, j2) + \
                                     dd(im1, jp1, i2, j2)
                        first_term *= b_stencil

                        second_term = dd(i1, jp1, i2, j2) + \
                                      dd(ip1, j1,i2,j2) + \
                                      dd(i1, jm1,i2,j2) + \
                                      dd(im1, j1,i2,j2)

                        second_term *= a_stencil
                        third_term = c_stencil*dd(i1, j1, i2, j2) # They have to be the same

                        result = (D/dr**2)*(first_term + second_term + third_term)

                        if fabs(result) > TOLERANCE:
                            c = position_to_logical[i2, j2]
                            zeta[r, c] = result

        return sp.sparse.csc_matrix(zeta)

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


    def run(self, record_images=True):
        sol_in_time = np.zeros((self.imax, self.jmax, self.kmax + 1), dtype=np.double)
        sol_in_time[:, :, 0] = self.fi_orig
        # We need to convert the original solution to the new form.
        fi = self.convert_fi_real_to_logical(self.fi_orig)
        # Convert fi to a sparse matrix

        # The left side is constant in time
        left_side = self.I - (self.dt/2.)*self.zeta + (self.dt/2.)*self.A
        propagator = self.I + (self.dt/2.)*self.zeta - (self.dt/2.)*self.A

        colormap = sns.cubehelix_palette(n_colors=1024, as_cmap=True)

        for i in range(self.kmax):
            if i % 50 == 0:
                print 'Done with iteration', i
                print 'Minimum (to check for stability):' , sol_in_time[:, :, i].min()
                if record_images:
                    ski.io.imshow(sol_in_time[:, :, i], cmap=colormap)
                    plt.grid(False)
                    plt.clim([0, 1])
                    plt.savefig('%05d'%i + str('.png'), dpi=200, bbox_inches='tight')
                    plt.clf()

            propagation = (propagator).dot(fi)
            growth = self.dt*self.s*fi*(1-fi)
            right_side = propagation + growth

            fi_plus_1 = sp.sparse.linalg.bicgstab(left_side, right_side, x0=fi, tol=10.**-1*TOLERANCE)[0]

            # Now get the solution in space
            sol_in_time[:, :, i+1] = self.convert_fi_logical_to_real(fi_plus_1)

            fi = fi_plus_1

        return sol_in_time

if __name__=='__main__':
    # Test script
    sol = Solver(imax=10, jmax=10, use_morton=False)
    print sol.run()