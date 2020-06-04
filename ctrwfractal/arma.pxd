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
from libc.stdint cimport int64_t


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


cdef inline Col[int64_t] numpy_to_col_i(np.ndarray[int64_t, ndim=1] X):
    if not X.flags.f_contiguous:
        X = X.copy(order="F")
    return Col[int64_t](<int64_t*> X.data, X.shape[0], False, False)


cdef inline Mat[double] numpy_to_mat_d(np.ndarray[double, ndim=2] X):
    if not X.flags.f_contiguous:
        X = X.copy(order="F")
    return Mat[double](<double*> X.data, X.shape[0], X.shape[1], False, False)


cdef inline Mat[float] numpy_to_mat_f(np.ndarray[float, ndim=2] X):
    if not X.flags.f_contiguous:
        X = X.copy(order="F")
    return Mat[float](<float*> X.data, X.shape[0], X.shape[1], False, False)


cdef inline Cube[double] numpy_to_cube_d(np.ndarray[double, ndim=3] X):
    if not X.flags.f_contiguous:
        X = X.copy(order="F")
    return Cube[double](<double*> X.data, X.shape[0], X.shape[1], X.shape[2], False, False)


cdef inline Cube[float] numpy_to_cube_f(np.ndarray[float, ndim=3] X):
    if not X.flags.f_contiguous:
        X = X.copy(order="F")
    return Cube[float](<float*> X.data, X.shape[0], X.shape[1], X.shape[2], False, False)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[int64_t, ndim=1] numpy_from_col_i(Col[int64_t] &m):
    cdef np.ndarray[int64_t, ndim=1] arr
    cdef int64_t *pArr
    cdef int64_t *pM
    arr = np.ndarray((m.n_rows), dtype=np.int64, order='F')
    pArr = <int64_t *>arr.data
    pM = m.memptr()

    for i in range(m.n_rows):
        pArr[i] = pM[i]

    return arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[double, ndim=2] numpy_from_mat_d(Mat[double] &m):
    cdef np.ndarray[double, ndim=2] arr
    cdef double *pArr
    cdef double *pM
    arr = np.ndarray((m.n_rows, m.n_cols), dtype=np.float64, order='F')
    pArr = <double *>arr.data
    pM = m.memptr()

    for i in range(m.n_rows * m.n_cols):
        pArr[i] = pM[i]

    return arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[float, ndim=2] numpy_from_mat_f(Mat[float] &m):
    cdef np.ndarray[float, ndim=2] arr
    cdef float *pArr
    cdef float *pM
    arr = np.ndarray((m.n_rows, m.n_cols), dtype=np.float32, order='F')
    pArr = <float *>arr.data
    pM = m.memptr()

    for i in range(m.n_rows * m.n_cols):
        pArr[i] = pM[i]

    return arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[double, ndim=3] numpy_from_cube_d(Cube[double] &m):
    cdef np.ndarray[double, ndim=3] arr
    cdef double *pArr
    cdef double *pM
    arr = np.ndarray((m.n_rows, m.n_cols, m.n_slices), dtype=np.float64, order='F')
    pArr = <double *>arr.data
    pM = m.memptr()

    for i in range(m.n_rows * m.n_cols * m.n_slices):
        pArr[i] = pM[i]

    return arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[float, ndim=3] numpy_from_cube_f(Cube[float] &m):
    cdef np.ndarray[float, ndim=3] arr
    cdef float *pArr
    cdef float *pM
    arr = np.ndarray((m.n_rows, m.n_cols, m.n_slices), dtype=np.float32, order='F')
    pArr = <float *>arr.data
    pM = m.memptr()

    for i in range(m.n_rows * m.n_cols * m.n_slices):
        pArr[i] = pM[i]

    return arr