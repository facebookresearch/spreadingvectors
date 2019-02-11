// -*- c++ -*-
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */
%module c_lattices

// release GIL by default for all functions
%exception {
    Py_BEGIN_ALLOW_THREADS
    $action
    Py_END_ALLOW_THREADS
}

%{
#include <omp.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "lattice_Zn.h"
%}

// simplified interface for vector
namespace std {

    template<class T>
    class vector {
    public:
        vector();
        ~vector();
        void push_back(T);
        void clear();
        T * data();
        size_t size();
        T at (size_t n) const;
        void resize (size_t n);
    };
};


%template(FloatVector) std::vector<float>;
%template(RepeatVector) std::vector<Repeat>;


typedef unsigned long uint64_t;
typedef unsigned int int32_t;
typedef unsigned char uint8_t;

%include "lattice_utils.h"
%include "lattice_Zn.h"


%inline %{
    // make sure all these pointer types are instanciated
void dummy_func(double * , int *, unsigned char *, long * , unsigned long *) {
}

%}

%{
PyObject *swig_ptr (PyObject *a)
{
    if(!PyArray_Check(a)) {
        PyErr_SetString(PyExc_ValueError, "input not a numpy array");
        return NULL;
    }
    PyArrayObject *ao = (PyArrayObject *)a;

    if(!PyArray_ISCONTIGUOUS(ao)) {
        PyErr_SetString(PyExc_ValueError, "array is not C-contiguous");
        return NULL;
    }
    void * data = PyArray_DATA(ao);
    if(PyArray_TYPE(ao) == NPY_FLOAT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_float, 0);
    }
    if(PyArray_TYPE(ao) == NPY_FLOAT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_double, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_int, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_long, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_long, 0);
    }
    PyErr_SetString(PyExc_ValueError, "did not recognize array type");
    return NULL;
}


%}

void omp_set_num_threads(int );
void *memcpy(void *dest, const void *src, size_t n);

%exception;

%exception;


%init %{
/* needed, else crash at runtime */
import_array();
%}

// return a pointer usable as input for functions that expect pointers
PyObject *swig_ptr (PyObject *a);

%pythoncode %{

import numpy as np

vector_name_map = {
    'Float': 'float32',
    'Byte': 'uint8',
    'Uint64': 'uint64',
    'Long': 'int64',
    'Int': 'int32',
    'Double': 'float64'
    }

def vector_to_array(v):
    """ convert a C++ vector to a numpy array """
    classname = v.__class__.__name__
    assert classname.endswith('Vector')
    dtype = np.dtype(vector_name_map[classname[:-6]])
    a = np.empty(v.size(), dtype=dtype)
    memcpy(swig_ptr(a), v.data(), a.nbytes)
    return a


def vector_float_to_array(v):
    return vector_to_array(v)


def copy_array_to_vector(a, v):
    """ copy a numpy array to a vector """
    n, = a.shape
    classname = v.__class__.__name__
    assert classname.endswith('Vector')
    dtype = np.dtype(vector_name_map[classname[:-6]])
    assert dtype == a.dtype, (
        'cannot copy a %s array to a %s (should be %s)' % (
            a.dtype, classname, dtype))
    v.resize(n)
    memcpy(v.data(), swig_ptr(a), a.nbytes)

%}
