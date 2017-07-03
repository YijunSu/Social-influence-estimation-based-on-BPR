# coding=utf-8

from scipy.sparse import coo_matrix
from numpy import array
from numpy import random
import tid_vid_dict_generator as dict_generator


class DataMode(object):
    def __init__(self):
        self.data = []
        self.train_row = []
        self.train_column = []
        self.train_value = []
        self.test_row = []
        self.test_column = []

    def find_data(self):

        user_id_dict = dict_generator.generate_user_id_dict()
        event_id_dict = dict_generator.generate_event_id_dict()
        event_users_dict = dict_generator.generate_event_users_dict()

        for key in event_users_dict:
            if len(event_users_dict[key]) == 1:
                self.train_row.append(event_id_dict[key])
                self.test_row.append(event_id_dict[key])
                self.train_column.append(user_id_dict[event_users_dict[key][0]])
                self.test_column.append(user_id_dict[event_users_dict[key][0]])
                self.train_value.append(1)
            else:
                for idx in range(len(event_users_dict[key])):
                    if idx == 0:
                        self.test_row.append(event_id_dict[key])
                        self.test_column.append(user_id_dict[event_users_dict[key][idx]])
                        self.train_row.append(event_id_dict[key])
                        self.train_column.append(user_id_dict[event_users_dict[key][idx]])
                        self.train_value.append(1)
                    else:
                        if random.randint(1, 5) == 1:
                            self.test_row.append(event_id_dict[key])
                            self.test_column.append(user_id_dict[event_users_dict[key][idx]])
                        else:
                            self.train_row.append(event_id_dict[key])
                            self.train_column.append(user_id_dict[event_users_dict[key][idx]])
                            self.train_value.append(1)



        """
        idx = random.randint(0, len(event_users_dict[key]))
                self.test_row.append(event_id_dict[key])
                self.test_column.append(user_id_dict[event_users_dict[key][idx]])

                for i in range(len(event_users_dict[key])):
                    if i != idx:
                        self.train_row.append(event_id_dict[key])
                        self.train_column.append(user_id_dict[event_users_dict[key][i]])
                        self.train_value.append(1)
                    else:
                        if user_id_dict[event_users_dict[key][idx]] == len(user_id_dict) - 1:
                            self.train_row.append(event_id_dict[key])
                            self.train_column.append(user_id_dict[event_users_dict[key][idx]])
                            self.train_value.append(1) 
        

        user_id_dict = dict_generator.generate_user_id_dict()
        self.colomn_length = len(user_id_dict)
        event_id_dict = dict_generator.generate_event_id_dict()
        self.row_lenth = len(event_id_dict)
        event_users_dict = dict_generator.generate_event_users_dict()

        counter = 0
        for key in event_users_dict:
            if len(event_users_dict[key]) == 0:
                counter += 1
        print("num of events which have no user:" + str(counter))
        for key in event_users_dict:
            if len(event_users_dict[key]) == 1:
                self.train_row.append(event_id_dict[key])
                self.test_row.append(event_id_dict[key])
                self.train_column.append(user_id_dict[event_users_dict[key][0]])
                self.test_column.append(user_id_dict[event_users_dict[key][0]])
                self.train_value.append(1)
            else:
                for idx in range(len(event_users_dict[key])):
                    if random.randint(1, 5) == 1:
                        self.test_row.append(event_id_dict[key])
                        self.test_column.append(user_id_dict[event_users_dict[key][idx]])
                    else:
                        self.train_row.append(event_id_dict[key])
                        self.train_column.append(user_id_dict[event_users_dict[key][idx]])
                        self.train_value.append(1)
     """

    def find_train_data(self):
        column = array(self.train_column)
        row = array(self.train_row)
        value = array(self.train_value)
        train_data_matrix = coo_matrix((value, (row, column)))
        train_data_csr = train_data_matrix.tocsr()
        return train_data_csr

    def find_test_data(self):
        test_data = []
        for i in range(len(self.test_row)):
            test_data.append((self.test_row[i], self.test_column[i]))
        return test_data

if __name__ == '__main__':
    d = DataMode()
    d.find_data()
    test_set = d.find_test_data()
    train_data = d.find_train_data()
    num_event, num_user = train_data.shape
    print num_event
    print num_user
