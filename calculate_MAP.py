# coding=utf-8

from getdata import DataMode
import numpy as np


class MapCalculator(object):
    def __init__(self):
        self.bpr_item_factors_list = []
        self.bpr_user_factors_list = []
        self.bpr_bias_list = []
        self.bpr_event_item_factors_list = []
        self.bpr_event_user_factors_list = []
        self.bpr_event_bias_list = []
        self.test_data = []
        self.train_data = None
        self.map_list = []

    def read_bpr_item_factor_file(self, factor_num):
        file_name = "item factor file with {0} factors.txt".format(factor_num)
        temp_list = []
        with open(file_name) as f:
            for line in f:
                result = line.split()
                float_result = []
                for element in result:
                    float_result.append(float(element))
                temp_list.append(float_result)
        self.bpr_item_factors_list.append(temp_list)

    def read_bpr_user_factor_file(self, factor_num):
        file_name = "user factor file with {0} factors.txt".format(factor_num)
        temp_list = []
        with open(file_name) as f:
            for line in f:
                result = line.split()
                float_result = []
                for element in result:
                    float_result.append(float(element))
                temp_list.append(float_result)
        self.bpr_user_factors_list.append(temp_list)

    def read_bpr_bias_file(self, factor_num):
        file_name = "bias file with {0} factors.txt".format(factor_num)
        temp_list = []
        with open(file_name) as f:
            for line in f:
                result = line.split()
                temp_list.append(float(result[0]))
        self.bpr_bias_list.append(temp_list)

    def read_bpr_event_item_factor_file(self, factor_num):
        file_name = "item factor file with {0} factors with event neighbor.txt".format(factor_num)
        temp_list = []
        with open(file_name) as f:
            for line in f:
                result = line.split()
                float_result = []
                for element in result:
                    float_result.append(float(element))
                temp_list.append(float_result)
        self.bpr_event_item_factors_list.append(temp_list)

    def read_bpr_event_user_factor_file(self, factor_num):
        file_name = "user factor file with {0} factors with event neighbor.txt".format(factor_num)
        temp_list = []
        with open(file_name) as f:
            for line in f:
                result = line.split()
                float_result = []
                for element in result:
                    float_result.append(float(element))
                temp_list.append(float_result)
        self.bpr_event_user_factors_list.append(temp_list)

    def read_bpr_event_bias_file(self, factor_num):
        file_name = "bias file with {0} factors with event neighbor.txt".format(factor_num)
        temp_list = []
        with open(file_name) as f:
            for line in f:
                result = line.split()
                temp_list.append(float(result[0]))
        self.bpr_event_bias_list.append(temp_list)

    def read_files(self):
        for num in [10, 20, 30, 40, 50]:
            self.read_bpr_bias_file(num)
            self.read_bpr_item_factor_file(num)
            self.read_bpr_user_factor_file(num)
            self.read_bpr_event_bias_file(num)
            self.read_bpr_event_item_factor_file(num)
            self.read_bpr_event_user_factor_file(num)

    def get_train_and_test_data(self):
        d = DataMode()
        d.find_data()
        self.test_data = d.find_test_data()
        self.train_data = d.find_train_data()

    # when calculate MAP
    # the final rank should include the test tuple and the all negative tuples
    # while the positive tuples should be ignored!
    # bingo!
    def calculate_map(self):
        self.read_files()
        self.get_train_and_test_data()
        for _ in range(10):
            self.map_list.append(0.0)

        for piece in self.test_data:
            event_id = piece[0]
            user_id = piece[1]
            index = self.train_data[event_id].indices
            positive_user_list = list(index)
            positive_user_list.append(user_id)
            negative_user_list = list(set(range(len(self.bpr_bias_list[0]))) - set(positive_user_list))
            # print negative_user_list
            id_influence_dict_list = []
            # id_influence_dict_list has 10 dict, the top 5 belongs to pure bpr model
            # the rest belongs to the bpr_event model

            for i in range(10):
                id_influence_dict_list.append({})
                if i < 5:
                    # bpr model
                    id_influence_dict_list[i][user_id] = np.dot(self.bpr_user_factors_list[i][event_id],
                                                                self.bpr_item_factors_list[i][user_id]) \
                                                         + self.bpr_bias_list[i][user_id]
                    for negative_id in negative_user_list:
                        id_influence_dict_list[i][negative_id] = np.dot(self.bpr_user_factors_list[i][event_id],
                                                                        self.bpr_item_factors_list[i][negative_id]) \
                                                         + self.bpr_bias_list[i][negative_id]
                else:
                    # bpr_event model
                    id_influence_dict_list[i][user_id] = np.dot(self.bpr_event_user_factors_list[i-5][event_id],
                                                                self.bpr_event_item_factors_list[i-5][user_id]) \
                                                         + self.bpr_event_bias_list[i-5][user_id]
                    for negative_id in negative_user_list:
                        id_influence_dict_list[i][negative_id] = np.dot(self.bpr_event_user_factors_list[i-5][event_id],
                                                                        self.bpr_event_item_factors_list[i-5][negative_id]) \
                                                         + self.bpr_event_bias_list[i-5][negative_id]
                sorted_influence_id = sorted(id_influence_dict_list[i].iteritems(), key=lambda d: d[1], reverse=True)

                for x in range(len(sorted_influence_id)):
                    if sorted_influence_id[x][0] == event_id:
                        self.map_list[i] += 1/(1 + float(x))
                        print self.map_list[i]
                    else:
                        pass

        for y in range(10):
            self.map_list[y] /= len(self.test_data)

    def generate_file(self):
        my_file = file('bpr map.txt', 'w')
        for _ in range(5):
            my_file.write(str(self.map_list[_]) + '\n')
        my_file.close()

        my_file = file("bpr map with event neighbor.txt", 'w')
        for _ in range(5):
            my_file.write(str(self.map_list[_ + 5]) + '\n')
        my_file.close()


if __name__ == '__main__':
    calculator = MapCalculator()
    calculator.calculate_map()
    calculator.generate_file()
