# -*- coding: utf-8 -*-
# distutils: language = c++
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport INFINITY
from libcpp.vector cimport vector

cdef inline double min3(double a, double b, double c):
    if a < b and a < c:
        return a
    elif b < c:
        return b
    else:
        return c

cdef inline int argmin3(double a, double b, double c):
    if a < b and a < c:
        return 0
    elif b < c:
        return 1
    else:
        return 2

def cost_matrix(x, y, dist_function):
    if len(x.shape) == len(y.shape):
        cost_matrix = create_cost_mat(x, y, dist_function)
    return cost_matrix

cdef double[:,:] create_cost_mat(x, y, dist_func):
    cdef double[:, :] cost_matrix
    cost_matrix = np.empty((x.shape[0], y.shape[0]), dtype=np.float64)
    # cost_matrix[:] = INFINITY
    cost_matrix[0,0] = dist_func(x[0], y[0])
    cdef int i, j
    for i in range(1, cost_matrix.shape[0]):
        cost_matrix[i, 0] = dist_func(x[i], y[0]) + cost_matrix[i-1, 0]
    
    for j in range(1, cost_matrix.shape[1]):
        cost_matrix[0, j] = dist_func(x[0], y[j]) + cost_matrix[0, j-1]

    for i in range(1, cost_matrix.shape[0]):
        for j in range(1, cost_matrix.shape[1]):
            cost_matrix[i, j] = dist_func(x[i],y[j])+ \
            min3(cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1])
    return cost_matrix[0:,0:]

def path(cost_matrix):
    cdef int x_len, y_len
    cdef double[:,:] cost_mat
    cost_mat = np.ascontiguousarray(cost_matrix)
    x_len = cost_mat.shape[0]
    y_len = cost_mat.shape[1]
    path_x, path_y = get_path(cost_mat, x_len, y_len)
    path = np.flip(np.array([[i, j] for i, j in zip(path_x, path_y)]), axis=0)
    return path


cdef get_path(double[:,:] cost_mat, int x_len, int y_len):
    cdef int i = x_len -1
    cdef int j = y_len -1
    cdef vector[int] path_x, path_y
    path_x.push_back(i)
    path_y.push_back(j)
    cdef int match
    while (i > 0 or j > 0):
        if (i == 0):
            j -= 1
        elif (j == 0):
            i -= 1
        else:
            match = argmin3(cost_mat[i - 1, j - 1], cost_mat[i - 1, j], cost_mat[i, j - 1])
            if match == 0:
                i -= 1
                j -= 1
            elif match == 1:
                i -= 1
            else:
                j -= 1
        path_x.push_back(i)
        path_y.push_back(j)

    return path_x, path_y
    
