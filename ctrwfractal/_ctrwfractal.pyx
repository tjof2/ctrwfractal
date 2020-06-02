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
from libc.stdint cimport uint8_t, uint32_t

from .arma cimport (
    Mat,
    Cube,
    numpy_to_mat_d,
    numpy_from_mat_d,
    numpy_to_mat_f,
    numpy_from_mat_f,
    numpy_to_cube_d,
    numpy_from_cube_d,
    numpy_to_cube_f,
    numpy_from_cube_f
)


cdef extern from "_ctrw.hpp":
    cdef uint32_t c_ctrw "CTRWwrapper"[T] (Mat[T] &, Mat[T] &, Cube[T] &,
                                           uint32_t, uint32_t, uint32_t,
                                           double, double, double, double,
                                           uint8_t, uint8_t, int, int)


def ctrw_fractal_double(uint32_t gridSize = 128,
                        uint32_t nWalks = 0,
                        uint32_t walkLength = 1,
                        double threshold = -1.0,
                        double beta = 0.0,
                        double tau0 = 1.0,
                        double noise = 0.0,
                        uint8_t latticeMode = 0,
                        uint8_t walkMode = 0,
                        int randomSeed = 0,
                        int nJobs = -1):

    cdef np.ndarray[double, ndim=2] lattice
    cdef np.ndarray[double, ndim=2] analysis
    cdef np.ndarray[double, ndim=3] walks

    cdef Mat[double] _lattice
    cdef Mat[double] _analysis
    cdef Cube[double] _walks

    cdef uint32_t result

    _lattice = Mat[double]()
    _analysis = Mat[double]()
    _walks = Cube[double]()

    result = c_ctrw[double](_lattice,
                            _analysis,
                            _walks,
                            gridSize,
                            nWalks,
                            walkLength,
                            threshold,
                            beta,
                            tau0,
                            noise,
                            latticeMode,
                            walkMode,
                            nJobs,
                            randomSeed)

    lattice = numpy_from_mat_d(_lattice)
    analysis = numpy_from_mat_d(_analysis)
    walks = numpy_from_cube_d(_walks)

    return lattice, analysis, walks, result

