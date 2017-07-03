# coding=utf-8

import pymysql as mdb


class DataFinder(object):
    def __init__(self):
        self.user_event_pairs = []
        self.event_info = []
        self.friends_info = []

    def get_user_event_pairs_from_sql_sever(self):
        # TODO: get data from mysql server
        con = mdb.connect(host='10.109.247.132', port=3306, user='root', passwd='abc123', db='douban')
        # construct a connection with the database
        cur = con.cursor()
        sql = """SELECT user_id,event_id FROM user_event_new_wish"""
        cur.execute(sql)
        # execute sql with cursor
        self.user_event_pairs = list(cur)
        # item_data is a list of tuples
        # each tuple includes user_id and event_id
        cur.close()
        con.close()

        with open('event_user_pairs.txt', 'w+') as f:

            for item in self.user_event_pairs:
                f.write(item[1] + '\t' + item[0] + '\n')
        f.close()

        print(len(self.user_event_pairs))

    def get_event_info_from_sql_server(self):
        # TODO: get data from mysql server
        con = mdb.connect(host='10.109.247.132', port=3306, user='root', passwd='abc123', db='douban')
        # construct a connection with the database
        cur = con.cursor()
        sql = """SELECT id, category, geo, owner_id FROM event_new"""
        cur.execute(sql)
        # execute sql with cursor
        self.event_info = list(cur)
        # item_data is a list of tuples
        # each tuple includes event_id, event_category, location
        cur.close()
        con.close()

        with open('event_id_category_location_organizer.txt', 'w+') as f:
            for item in self.event_info:
                latitude, longitude = item[2].split()
                if longitude == '0.0' and latitude == '0.0':
                    pass
                else:
                    f.write(str(item[0]) + '\t' + item[1] + '\t' + item[2] + '\t' + str(item[3]) + '\n')
        f.close()

        print(len(self.event_info))

    def get_user_friendship_from_sql_server(self):
        con = mdb.connect(host='10.109.247.132', port=3306, user='root', passwd='abc123', db='douban')
        # construct a connection with the database
        cur = con.cursor()
        sql = """SELECT uid, friends FROM friendship_copy"""
        cur.execute(sql)
        # execute sql with cursor
        self.friends_info = list(cur)
        # item_data is a list of tuples
        # each tuple includes event_id, event_category, location
        cur.close()
        con.close()

        with open('user_and_friends.txt', 'w+') as f:
            for item in self.friends_info:
                print(str(item[0]))
                print('this user\'s friends are:')

                friends = item[1].strip().split(',')
                if len(friends) == 0:
                    print('None')
                else:
                    for friend in friends:
                        print(friend)
                f.write(str(item[0]) + '\t' + item[1] + '\n')
        f.close()


if __name__ == '__main__':
    d = DataFinder()
    d.get_user_event_pairs_from_sql_sever()
    d.get_event_info_from_sql_server()
    d.get_user_friendship_from_sql_server()
