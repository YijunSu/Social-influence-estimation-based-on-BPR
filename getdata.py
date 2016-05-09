import pymysql as mdb
from scipy.sparse import coo_matrix
from numpy import array


class DataMode(object):
    def __init__(self):
        self.data = []
        self.useful_event_id = []
        self.useful_user_id = []
        self.train_row = []
        self.train_column = []
        self.train_value = []
        self.test_row = []
        self.test_column = []
        self.test_value = []

    def generate_file(self):
        # TODO: get data from mysql server and store the data in local files
        con = mdb.connect(host='10.109.247.132', port=3306, user='root', passwd='abc123', db='douban')
        cur = con.cursor()
        sql = """SELECT user_id,event_id,flag
        FROM user_event_bak
        """
        cur.execute(sql)
        self.data = list(cur)
        # item_data is a list of tuples
        # each tuple includes event_id, event_category, location_id and owner_id
        cur.close()
        con.close()

        matrix_file = file('matrix_file.txt', 'w')
        for item in self.data:
            if int(item[0]) in self.useful_user_id and int(item[1]) in self.useful_event_id and item[2] == 1:
                print >>matrix_file, item[0], item[1], item[2]
        matrix_file.close()

    def find_useful_event_id(self):
        with open('event_file.txt') as f:
            for line in f:
                result = line.split()
                self.useful_event_id.append(int(result[0]))

    def find_useful_user_id(self):
        with open('user_file.txt') as f:
            for line in f:
                result = line.split()
                self.useful_user_id.append(int(result[0]))

    def find_data(self):
        user_id_dict = {}
        event_id_dict = {}
        with open('user_file.txt') as uf:
            for u_line in uf:
                u_result = u_line.split()
                user_id_dict[u_result[0]] = int(u_result[1])

        with open('event_file.txt') as ef:
            for e_line in ef:
                e_result = e_line.split()
                event_id_dict[e_result[0]] = int(e_result[1])

        event_user_dict = {}
        # key and value are both string type
        with open('matrix_file.txt') as f:
            for line in f:
                result = line.split()
                if result[1] in event_user_dict:
                    event_user_dict[result[1]].append(result[0])
                else:
                    event_user_dict[result[1]] = []
                    event_user_dict[result[1]].append(result[0])
            # item[0] is event_id
            # item[1] is a list of user_id
        for item in event_user_dict.items():
            if len(item[1]) == 1:
                for i in item[1]:
                    self.train_row.append(event_id_dict[item[0]])
                    self.train_column.append(user_id_dict[i])
                    self.train_value.append(1)
            else:
                beg = 0
                for i in item[1]:
                    if beg == 0:
                        self.test_row.append(event_id_dict[item[0]])
                        self.test_column.append(user_id_dict[i])
                        self.test_value.append(1)
                    else:
                        self.train_row.append(event_id_dict[item[0]])
                        self.train_column.append(user_id_dict[i])
                        self.train_value.append(1)
                    beg += 1

    def find_train_data(self):
        column = array(self.train_column)
        row = array(self.train_row)
        value = array(self.train_value)
        train_data_matrix = coo_matrix((value, (row, column)))
        train_data_csr = train_data_matrix.tocsr()
        return train_data_csr

    def find_test_data(self):
        test_data = []
        for i in xrange(len(self.test_row)):
            test_data.append((self.test_row[i], self.test_column[i]))
        return test_data

if __name__ == '__main__':
    d = DataMode()
    d.find_data()
    test_set = d.find_test_data()
    print "test set is:", test_set
    print 'length of test data is:', len(test_set)
