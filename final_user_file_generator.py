#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

"""
===============================================================================
@author:    赵明星
@desc:      生成模型需要的数据集的所有用户的id文件。
===============================================================================
"""

user_set = set()
with open('event_user_pairs_after_washing.txt') as f:
    for line in f:
        event, user = line.split()
        user_set.add(user)
f.close()
print(len(user_set))

with open('final_user_id.txt', 'w+') as f:
    for user in user_set:
        f.write(user + '\n')
f.close()
