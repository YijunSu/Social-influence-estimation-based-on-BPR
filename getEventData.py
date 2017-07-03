#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

"""
===============================================================================
author: 赵明星
desc:   连接数据库，并将需要的数据从数据库中提取出来。
===============================================================================
"""

import pymysql as mdb


class GenerateFile(object):
    def __init__(self):
        self.event_info = []
        self.useful_event = []

    def get_data_from_sql(self):
        # TODO: get data we need from mysql server

        host = "127.0.1"
        port = 3306
        user = "root"
        passwd = "123456"
        db = "event"
        # 使用该代码时请注意修改以上一个字段，以及后面的sql查询语句。

        con = mdb.connect(host, port, user, passwd, db)
        cur = con.cursor()
        sql = """
        """
        cur.execute(sql)
        self.event_info = list(cur)
        print 'event info:', self.event_info
        print 'size of original event data:', len(self.event_info)
        # item_data is a list of tuples
        # each tuple includes event_id, event_category, location_id and owner_id
        cur.close()
        con.close()

    def get_useful_event(self):
        result = []
        with open('event_file.txt') as f:
            for line in f:
                result.append(line.split())
        for r in result:
            for event in r:
                self.useful_event.append(int(event))
        print 'size of useful event data:', len(self.useful_event)

    def generate_file(self):
        event_info_file = file('event_info.txt', 'w')
        i = 0
        for event in self.event_info:
            if int(event[0]) in self.useful_event:
                print >>event_info_file, event[0], event[1], event[2], event[3]
                print(i)
                i += 1
        event_info_file.close()

if __name__ == '__main__':
    m = GenerateFile()
    m.get_data_from_sql()
    m.get_useful_event()
    m.generate_file()
