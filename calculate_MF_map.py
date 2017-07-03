# coding=utf-8

import numpy as np

event_and_positive_user_dict = {}
user_set = set()
with open("Ceshi_Influence.txt") as f:
    for line in f:
        result = line.split()
        if (int(result[0])) not in user_set:
            user_set.add(int(result[0]))
        if int(result[1]) in event_and_positive_user_dict:
            event_and_positive_user_dict[int(result[1])].append(int(result[0]))
        else:
            event_and_positive_user_dict[int(result[1])] = []
            event_and_positive_user_dict[int(result[1])].append(int(result[0]))
print event_and_positive_user_dict.items()
print len(event_and_positive_user_dict.items())

all_user_list = range(652)
print all_user_list
test_data = []
# test_data is a list of tuples composed of event_id and user_id
for i in event_and_positive_user_dict.iteritems():
    test_data.append((int(i[0]), int(i[1][0])))
print test_data

user_factor_dict = {}
user = 0
with open("MF_p_matrix.txt") as f:
    for line in f:
        result = line.split()
        for i in range(len(result)):
            result[i] = float(result[i])
        user_factor_dict[user] = result
        user += 1

event_factors = []
with open("MF_q_matrix.txt") as f:
    for line in f:
        result = line.split()
        for i in range(len(result)):
            result[i] = float(result[i])
        event_factors.append(result)
print event_factors
print len(event_factors)

event_factor_dict = {}
for i in range(1000):
    event_factor_dict[i] = []
    for j in range(20):
        event_factor_dict[i].append(event_factors[j][i])

AP = 0.0
for piece in test_data:
    event_id = piece[0]
    user_id = piece[1]
    negative_user_list = list(set(all_user_list) - set(event_and_positive_user_dict[event_id]))
    negative_user_list.append(user_id)
    ranking_user_list = negative_user_list

    user_and_score_dict = {}
    for j in ranking_user_list:
        x_u_j = np.dot(event_factor_dict[event_id], user_factor_dict[j])
        user_and_score_dict[j] = x_u_j
    user_and_score_list = sorted(user_and_score_dict.iteritems(), key=lambda d: d[1], reverse=True)
    for i in range(len(user_and_score_list)):
        if user_and_score_list[i][0] == user_id:
            AP += 1.0/(1.0 + i)

MAP = AP/(len(test_data))
print MAP
