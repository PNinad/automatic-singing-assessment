# -*- coding: utf-8 -*-
# distutils: language = c++
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs, sqrt

params = dict()

cdef inline int modulo(int a, int b, int m):
    return min((a-b)%m, (b-a)%m)

cdef inline float factor(float alpha, int modulo):
    # cdef int mod = modulo
    cdef float a = alpha
    cdef float factor = 1.0
    for i in range(modulo):
        factor = factor*a
    return factor

def euclidean(x, y):
    if x.shape and x.shape[0] > 1:
        return eval_euclidean(x, y)
    else:
        return fabs(x-y)

cdef float eval_euclidean(vector1, vector2):
    cdef float[:] x = vector1
    cdef float[:] y = vector2
    cdef int i
    cdef float tmp, d
    d = 0
    for i in range(x.shape[0]):
        tmp = x[i] - y[i]
        d += tmp * tmp
    return sqrt(d)

def cosine(x, y):
    return eval_cosine(x, y)

cdef float eval_cosine(vector1, vector2):
    cdef float[:] x = vector1
    cdef float[:] y = vector2
    cdef int i
    cdef float dot=0, mod_x=0, mod_y=0
    for i in range(x.shape[0]):
        dot += x[i]*y[i]
        mod_x += x[i]*x[i]
        mod_y += y[i]*y[i]
    
    mod_x =sqrt(mod_x)
    mod_y= sqrt(mod_y)

    if mod_x==0 and mod_y == 0:
        return 0
    elif mod_x ==0 and mod_y !=0:
        return 1
    elif mod_x !=0 and mod_y ==0:
        return 1
    else:
        return (1- dot/(mod_x*mod_y))

# def cos_W(x, y):
#     pass
def cos_W(x, y):
    alpha = params['alpha']
    hpcp_size = params['hpcp_size']
    if(alpha <= 0 or alpha > 1):
         return cosine(x,y)
    return eval_cos_W(x, y, alpha, hpcp_size)

cdef float eval_cos_W(vector1, vector2, alpha, hpcp_size):
    cdef float[:] x = vector1
    cdef float[:] y = vector2
    cdef float[:] x_roll
    cdef vector[float] num
    cdef float _alpha = alpha
    cdef int _hpcp_size = hpcp_size
    cdef int i, j
    cdef float dot=0, mod_x=0, mod_y=0
    cdef float max_num=0
    cdef int max_shift = int(hpcp_size/12)
    
    for i in range(x.shape[0]):
        mod_x += x[i]*x[i]
        mod_y += y[i]*y[i]

    mod_x = sqrt(mod_x)
    mod_y = sqrt(mod_y)
      
    if (mod_x==0 and mod_y == 0):
        return 0        
    elif (mod_x==0 and mod_y != 0):
        return 1
    elif (mod_x!=0 and mod_y==0):
        return 1
    else:
        for j in range(-1*max_shift, max_shift+1):
            x_roll = np.roll(x, j)
            dot = 0
            for i in range(x.shape[0]):
                dot += x_roll[i]*y[i]
            num.push_back(factor(_alpha , modulo(0, j, _hpcp_size))*dot)
        max_num = max(num)
        return (1 - max_num/(mod_x*mod_y))
        # # 1-(i/6)**1.7 
        # # #num =[np.cos((np.pi/2)*(analysis.modulo(0,i,HPCP_SIZE)/(HPCP_SIZE/2)))*np.dot(np.roll(x,i),y) for i in range(-1*max_shift, max_shift+1)]
        # num =[(_alpha**modulo(0,i,_hpcp_size))*np.dot(np.roll(x,i),y) for i in range(-1*max_shift, max_shift+1)]
        # #num= np.dot(x,y)
        # num = max(num)
        # dist = 1 - num/(mod_x*mod_y)    
    # return dist