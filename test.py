# coding=utf-8

import numpy
import sys
import random
from scipy.sparse import coo_matrix
from numpy import array
import matplotlib.pyplot as plt
import beautifulsoup4 as bs4

l = []
for i in xrange(10):
    l.append(random.randint(0, 9))
row = array(l)
l = []
for _ in xrange(10):
    l.append(random.randint(0, 9))
column = array(l)
l = []
for o in xrange(10):
    l.append(random.randint(0, 1))
d = array(l)
data = coo_matrix((d, (row, column)), (10, 10))
print(data)
print(row)
print(column)
print(d)
print data.nnz, "data.nnz"
print data.nonzero(), 'data.nonzero()'
print data.nonzero()[0].size
csr_data = data.tocsr()
print(csr_data), 'csr data'
print(csr_data.shape), 'data shape'
index = range(10)
print(index)


def flatten(mode):
    for element in mode:
        try:
            element.append()
            flatten(element)
        except TypeError:
                yield element
l = [[1, 2], [4, 6], [[1, 2, 4]], [2, 4]]
for x in flatten(l):
    print "x:", x

a = numpy.random.random_sample((5, 4))
print a
print(numpy.dot(a[1], a[2]))
print(a[1])
print(a[2])
print 'numpy.dot(a[1], a[2]):', numpy.dot(a[1], a[2])
print 'numpy.dot(a[1], a[2].T):', numpy.dot(a[1], a[2].T)
x = a[1][0]*a[2][0] + a[1][1]*a[2][1] + a[1][2]*a[2][2] + a[1][3]*a[2][3]
print 'calculate x myself :', x
a[1] += a[2]
print a

s = set()
for i in range(10):
    s.add(i)
print(s)
for i in list(s):
    print i

a = array(l)
print(l)
print(a)

x = numpy.linspace(-5, 5, 1000)
y = 1.0 / (1.0 + numpy.exp(x))
z = numpy.exp(x) / (1.0 + numpy.exp(x))
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='$sin(x)$', color='red')
plt.plot(x, z, label='$cos(x^2)$')
plt.title('figure example 1')
plt.ylim(-2, 2)
plt.legend()
plt.show()

my_file = file('test_file.txt', 'w')
for i in range(200):
    my_file.write(str(i) + ',')
    if i % 10 == 0:
        my_file.write('\n')
my_file.close()
'''
print 'read file begin!'
with open('test_file.txt') as f:
    for line in f:
        print line
'''
print sys.maxunicode

with open('diary_file.txt') as f:
    for line in f:
        print line
