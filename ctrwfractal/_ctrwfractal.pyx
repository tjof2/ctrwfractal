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
from libc.stdint cimport uint8_t, uint64_t, int64_t

from .arma cimport Mat, Cube, numpy_from_mat_d, numpy_from_cube_d


cdef extern from "_ctrw.hpp":
    cdef uint64_t c_ctrw "CTRWwrapper"[T] (Mat[T] &, Mat[T] &, Cube[T] &,
                                           uint64_t, uint64_t, uint64_t,
                                           double, double, double, double,
                                           uint8_t, uint8_t, int64_t, int64_t)


def ctrw_fractal(uint64_t grid_size = 128,
                 uint64_t n_walks = 0,
                 uint64_t n_steps = 1,
                 double threshold = -1.0,
                 double beta = 0.0,
                 double tau0 = 1.0,
                 double noise = 0.0,
                 uint8_t lattice_type = 0,
                 uint8_t walk_type = 0,
                 int64_t random_seed = 0,
                 int64_t n_jobs = -1):

    cdef np.ndarray[double, ndim=2] lattice
    cdef np.ndarray[double, ndim=2] analysis
    cdef np.ndarray[double, ndim=3] walks

    cdef Mat[double] _lattice
    cdef Mat[double] _analysis
    cdef Cube[double] _walks

    cdef uint64_t result

    _lattice = Mat[double]()
    _analysis = Mat[double]()
    _walks = Cube[double]()

    result = c_ctrw[double](_lattice,
                            _analysis,
                            _walks,
                            grid_size,
                            n_walks,
                            n_steps,
                            threshold,
                            beta,
                            tau0,
                            noise,
                            lattice_type,
                            walk_type,
                            n_jobs,
                            random_seed)

    lattice = numpy_from_mat_d(_lattice)
    analysis = numpy_from_mat_d(_analysis)
    walks = numpy_from_cube_d(_walks)

    return lattice, analysis, walks, result

