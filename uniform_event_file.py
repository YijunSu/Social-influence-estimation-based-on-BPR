#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

"""
===============================================================================
author: 赵明星
desc:   保存event真实id到数字id的映射。
===============================================================================
"""


class UniformEventFile(object):
    def __init__(self):
        self.event_list = []

    def fill_set(self):
        with open('event_info.txt') as f:
            for line in f:
                result = line.split()
                self.event_list.append(int(result[0]))
        print self.event_list
        print 'length of list:', len(self.event_list)

    def alter_event_file(self):
        i = 0
        f = file('event_file.txt', 'w')
        for item in self.event_list:
            print >>f, item, i
            i += 1

if __name__ == '__main__':
    a = UniformEventFile()
    a.fill_set()
    a.alter_event_file()
