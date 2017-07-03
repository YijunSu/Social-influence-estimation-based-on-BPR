# coding=utf-8

import csv
import pymysql as mdb


class ContentFileGenerator(object):
    def __init__(self):
        self.ordered_content_list = []
        self.all_event_list = []
        self.event_content_dict = {}
        # event_content_dict'key is t_event_id(str type) and its value is is a list
        # the list shows the content of the event
        self.ordered_event_list = []
        self.all_event_content_dict = {}
        self.count = 0

    # TODO: get data(t_event_id and event_content) from csv file(exported from mysql server)
    def get_data_from_csv(self):
        self.generate_all_event_list()
        self.generate_ordered_event_list()
        self.init_event_content_dict()
        with open('event_content.csv', 'rU') as f:
            csv_reader = csv.reader(f, quoting=csv.QUOTE_NONE)
            event_id = None
            for row in csv_reader:
                print '正在处理csv文件的第{0}行……'.format(self.count)
                self.count += 1
                line = ','.join(row)
                result = line.split('\t')
                if result[0] in self.all_event_list:
                    event_id = result[0]
                    for i in xrange(len(result)):
                        if i == 0:
                            pass
                        else:
                            self.all_event_content_dict[event_id].append(result[i])
                else:
                    for content in result:
                        self.all_event_content_dict[event_id].append(content)
        for event in self.ordered_event_list:
            self.event_content_dict[event] = self.all_event_content_dict[event]
            for content in self.all_event_content_dict[event]:
                print content

    def generate_ordered_event_list(self):
        with open('event_file.txt') as f:
            for line in f:
                result = line.split()
                self.ordered_event_list.append(result[0])

    def init_event_content_dict(self):
        for event in self.ordered_event_list:
            self.event_content_dict[event] = []
        for event in self.all_event_list:
            self.all_event_content_dict[event] = []

    def generate_all_event_list(self):
        # TODO: get all_event list from mysql server
        con = mdb.connect(host, port, user, passwd, db)
        cur = con.cursor()
        sql = """SELECT id
        FROM event_list_new
        """
        cur.execute(sql)
        event_list = list(cur)
        cur.close()
        con.close()
        i = 0
        for event in event_list:
            self.all_event_list.append(event[0])
            print '第{0}个event:{1}'.format(i, event)
            i += 1

    def generate_content_file(self):
        my_file = file('ordered_content_file.txt', 'w')
        i = 0
        for event in self.ordered_event_list:
            print '正在向文件中输入第{0}个event的内容……'.format(i)
            for content in self.event_content_dict[event]:
                my_file.write(content)
            my_file.write('\n')
            i += 1
        my_file.close()


if __name__ == '__main__':
    file_generator = ContentFileGenerator()
    file_generator.get_data_from_csv()
    file_generator.generate_content_file()
