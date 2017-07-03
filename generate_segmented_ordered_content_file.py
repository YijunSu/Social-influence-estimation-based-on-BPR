#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

"""
===============================================================================
author: 赵明星
desc:   切分文本，不如结巴分词好用，鉴定完毕。
===============================================================================
"""

from mmseg import seg_txt


class FileGenerator(object):
    def __init__(self):
        self.count = 0

    def generate_segmented_content_file(self):
        my_file = file('ordered_segmented_content_file.txt', 'w')
        with open('ordered_content_file.txt') as f:
            for line in f:
                print "正在对第{0}行进行分词操作……".format(self.count)
                for segment in seg_txt(line):
                    my_file.write(segment + ' ')
                my_file.write('\n')
                self.count += 1
        my_file.close()

if __name__ == '__main__':
    generator = FileGenerator()
    generator.generate_segmented_content_file()
