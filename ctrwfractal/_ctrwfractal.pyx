# Copyright 2016-2020 Tom Furnival
#
# This file is part of ctrwfractal.
#
# ctrwfractal is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ctrwfractal is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ctrwfractal.  If not, see <http://www.gnu.org/licenses/>.

# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libc.stdint cimport uint64_t, int64_t

np.import_array()


cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void PyArray_CLEARFLAGS(np.ndarray arr, int flags)


cdef extern from "<armadillo>" namespace "arma" nogil:
    ctypedef int uword

    cdef cppclass Col[T]:
        const uword n_rows
        const uword n_elem

        Col() nogil

        Col(uword n_rows) nogil

        Col(T* aux_mem,
            uword n_rows,
            bool copy_aux_mem,
            bool strict) nogil

        T *memptr() nogil

    cdef cppclass Mat[T]:
        const uword n_rows
        const uword n_cols
        const uword n_elem

        Mat() nogil

        Mat(uword n_rows, uword n_cols) nogil

        Mat(T* aux_mem,
            uword n_rows,
            uword n_cols,
            bool copy_aux_mem,
            bool strict) nogil

        T *memptr() nogil

    cdef cppclass Cube[T]:
        const uword n_rows
        const uword n_cols
        const uword n_slices
        const uword n_elem

        Cube() nogil

        Cube(uword n_rows, uword n_cols, uword n_slices) nogil

        Cube(T* aux_mem,
            uword n_rows,
            uword n_cols,
            uword n_slices,
            bool copy_aux_mem,
            bool strict) nogil

        T *memptr() nogil


cdef extern from "utils/utils.hpp":
    void SetMemState[T](T& m, int state)
    void SetMemStateCube[T](T& m, int state)
    size_t GetMemState[T](T& m)
    double* GetMemory(Col[double]& m)
    double* GetMemory(Mat[double]& m)
    double* GetMemory(Cube[double]& m)
    int64_t* GetMemory(Col[int64_t]& m)
    int64_t* GetMemory(Mat[int64_t]& m)
    int64_t* GetMemory(Cube[int64_t]& m)


cdef np.ndarray[np.int64_t, ndim=1] numpy_from_col_i(Col[int64_t] &m) except +:
    cdef np.npy_intp dim = <np.npy_intp> m.n_elem
    cdef np.ndarray[np.int64_t, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &dim, np.NPY_INT64, GetMemory(m))

    if GetMemState[Col[int64_t]](m) == 0:
        SetMemState[Col[int64_t]](m, 1)
        PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)

    return arr


cdef np.ndarray[np.double_t, ndim=2] numpy_from_mat_d(Mat[double] &m) except +:
    cdef np.npy_intp dims[2]
    dims[0] = <np.npy_intp> m.n_cols
    dims[1] = <np.npy_intp> m.n_rows
    cdef np.ndarray[np.double_t, ndim=2] arr = np.PyArray_SimpleNewFromData(2, &dims[0], np.NPY_DOUBLE, GetMemory(m))

    if GetMemState[Mat[double]](m) == 0:
        SetMemState[Mat[double]](m, 1)
        PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)

    return arr


cdef np.ndarray[np.double_t, ndim=3] numpy_from_cube_d(Cube[double] &m) except +:
    cdef np.npy_intp dims[3]
    dims[0] = <np.npy_intp> m.n_slices
    dims[1] = <np.npy_intp> m.n_cols
    dims[2] = <np.npy_intp> m.n_rows
    cdef np.ndarray[np.double_t, ndim=3] arr = np.PyArray_SimpleNewFromData(3, &dims[0], np.NPY_DOUBLE, GetMemory(m))

    if GetMemState[Cube[double]](m) == 0:
        SetMemStateCube[Cube[double]](m, 1)
        PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)

    return arr


cdef extern from "_ctrw.hpp":
    cdef uint64_t c_ctrw "CTRWwrapper"[T] (Col[int64_t] &, Mat[T] &, Mat[T] &, Cube[T] &,
                                           uint64_t, uint64_t, double,
                                           uint64_t, uint64_t, uint64_t,
                                           double, double, double,
                                           int64_t, int64_t)


def ctrw_fractal(uint64_t grid_size = 32,
                 uint64_t lattice_type = 0,
                 double threshold = 0.0,
                 uint64_t walk_type = 0,
                 uint64_t n_walks = 0,
                 uint64_t n_steps = 0,
                 double beta = 0.0,
                 double tau0 = 1.0,
                 double noise = 0.0,
                 int64_t random_seed = -1,
                 int64_t n_jobs = -1):

    cdef uint64_t result

    cdef np.ndarray[np.int64_t, ndim=1] clusters
    cdef np.ndarray[np.double_t, ndim=2] lattice
    cdef np.ndarray[np.double_t, ndim=2] analysis
    cdef np.ndarray[np.double_t, ndim=3] walks

    cdef Col[int64_t] _clusters
    cdef Mat[double] _lattice
    cdef Mat[double] _analysis
    cdef Cube[double] _walks

    _clusters = Col[int64_t]()
    _lattice = Mat[double]()
    _analysis = Mat[double]()
    _walks = Cube[double]()

    result = c_ctrw[double](_clusters,
                            _lattice,
                            _analysis,
                            _walks,
                            grid_size,
                            lattice_type,
                            threshold,
                            walk_type,
                            n_walks,
                            n_steps,
                            beta,
                            tau0,
                            noise,
                            random_seed,
                            n_jobs)

    clusters = numpy_from_col_i(_clusters)
    lattice = numpy_from_mat_d(_lattice)
    analysis = numpy_from_mat_d(_analysis)
    walks = numpy_from_cube_d(_walks)

    return clusters, lattice, walks, analysis, result

