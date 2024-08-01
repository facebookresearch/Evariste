# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string

cdef extern from "earley.hpp":
    cdef cppclass Grammar:
        Grammar(string)
        pair[vector[int], vector[string]] parse(string)

