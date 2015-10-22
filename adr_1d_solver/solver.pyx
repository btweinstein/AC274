#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np

import scipy as sp

# Setup the simulation

cdef class one_d_adr(object):

    cdef:
        public int imax
        public int jmax
        public double dt
        public double dx
        public double v
        public double D
        public double s

        double[:] xgrid
        double[:] tgrid

        double[:] fi

        double[:, :] A
        double[:, :] zeta
        double[:, :] I

    def __init__(self, int imax=100, int jmax=1000, double dt=0.01, double dx=1.,
                double[:] v = None, double D=5., double s=1., double[:] fi=None):

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
        if fi is None:
            self.fi = sp.stats.norm(loc=dx*jmax/2, scale=50*dx).pdf(self.xgrid)
        else:
            self.fi = fi

        self.setup_matrices()

    cdef setup_matrices(self):
        # Define the advection operator

        self.A = np.zeros((self.jmax, self.jmax), dtype=np.double)
        for r in range(self.A.shape[0]):
            for c in range(self.A.shape[1]):
                to_the_right = (c+1)%self.A.shape[1]
                to_the_left = (c-1)%self.A.shape[1]
                if r == to_the_right:
                    self.A[r, c] = -self.v[r]/(2*self.dx)
                elif r == to_the_left:
                    self.A[r,c] = self.v[r]/(2*self.dx)

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
        I = np.identity(self.jmax, dtype=np.double)

    cdef double[:] G(self, double[:] f):
        return self.s*f*(1-f)







cpdef do_sim(int imax=100, int jmax=1000, double dt=0.01, double dx=1.,
             double[:] v = None, double D=5., double s=1., double[:] fi=None):

    if v is None:
        v = 10.*np.ones(jmax, dtype=np.double)

    cdef double[:]
    if fi is None:
        print 'waka'


    # Define the advection operator


