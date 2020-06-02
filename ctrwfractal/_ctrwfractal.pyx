# -*- coding: utf-8 -*-
# Copyright 2016-2020 Tom Furnival
#
# This file is part of CTRWfractal.
#
# CTRWfractal is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CTRWfractal is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CTRWfractal.  If not, see <http://www.gnu.org/licenses/>.

# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libc.stdint cimport uint32_t

from .arma cimport Mat, numpy_to_mat_d, numpy_from_mat_d, numpy_to_mat_f, numpy_from_mat_f


cdef extern from "ctrw.hpp":
    cdef uint32_t c_ctrw "CTRWwrapper"[T] (Mat[T] &, Mat[T] &, Mat[T] &,
                                            double, double, bool, double, double,
                                            uint32_t, uint32_t, int)


def ctrw_fractal_double(np.ndarray[np.float64_t, ndim=2] X,
                    double lambda1 = 1.0,
                    double tol = 1e-7,
                    bool subsample = False,
                    double sampleL = 1.0,
                    double sampleH = 1.0,
                    uint32_t maxRank = 10,
                    uint32_t maxIter = 1000,
                    int randomSeed = -1):

    cdef np.ndarray[double, ndim=2] A
    cdef np.ndarray[double, ndim=2] E

    cdef Mat[double] _A
    cdef Mat[double] _E

    cdef uint32_t rankEstimate

    _A = Mat[double]()
    _E = Mat[double]()

    rankEstimate = c_rosl_lrs[double](numpy_to_mat_d(X), _A, _E,
                                      lambda1, tol, subsample, sampleL, sampleH,
                                      maxRank, maxIter, randomSeed)

    A = numpy_from_mat_d(_A)
    E = numpy_from_mat_d(_E)

    return A, E, rankEstimate

