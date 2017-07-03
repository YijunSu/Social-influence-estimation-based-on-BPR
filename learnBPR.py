#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

"""
===============================================================================
author: 赵明星
desc:   学习BPR代码。
===============================================================================
"""

import numpy as np
from math import log, exp
import random
from scipy.sparse import coo_matrix
from numpy import array


if __name__ == '__main__':
    l = []
    for i in xrange(100):
        l.append(i)
    print l
    row = array(l)
    l = []
    for i in xrange(100):
        l.append(i)
    column = array(l)
    l = []
    for i in xrange(100):
        l.append(1)
    d = array(l)
    data = coo_matrix((d, (row, column)))
    csr_data = data.tocsr()
    print csr_data
    i = random.randint(1, 10)
    while i in csr_data[1].indices:
        i = random.randint(1, 10)
        print i

    x = np.random.random_sample((2, 4))
    y, z = x.nonzero()
    print "y:", y
    print "z:", z
    l = range(8)
    print l
    random.shuffle(l)
    print l
    z = z[l]
    print "z:", z
    print np.dot(x[0, :], x[1, :])
    print x[0, :], x[1, :]
    print x[0, 0], x[0, 1], x[1, 0], x[1, 1]
    print x[0, 0] * x[1, 0] +  \
          x[0, 1] * x[1, 1]

    print x.shape
    print log(2.8)
