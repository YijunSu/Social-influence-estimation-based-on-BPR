import pymysql as mdb


class GenerateFile(object):
    def __init__(self):
        self.event_info = []
        self.useful_event = []

    def get_data_from_sql(self):
        # TODO: get data we need from mysql server
        con = mdb.connect(host='10.109.247.132', port=3306, user='root', passwd='abc123', db='douban')
        cur = con.cursor()
        sql = """SELECT id,category,geo,owner_id
        FROM event_list_new
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
