# coding=utf-8

"""
    @author:    Jack Ome
    @date:      20160712
"""

from math import *

event_id_info_dict = {}
# event_id_info_dict: key is event tid
# value is a list [category, latitude, longitude, organizer]
with open('event_id_category_location_organizer.txt', 'r') as f:
    for line in f:
        event_tid, category, latitude, longitude, organizer = line.split()
        event_id_info_dict[event_tid] = [category, latitude, longitude, organizer]
        #                                   0          1         2          3

# TODO: read file and generate a dict, key is event_tid value is user set
event_users_dict = {}
with open('event_user_pairs.txt') as f:
    for line in f:
        event, user = line.split()
        if event not in event_users_dict:
            event_users_dict[event] = set()
            event_users_dict[event].add(user)
        else:
            event_users_dict[event].add(user)

# TODO: select some expressive data to verify our model
event_id_info_dict_after_washing = {}
# this is our selected event data
# then select user data according to these events
# this dict is a small subset of event_id_info_dict

key_selected = list(event_id_info_dict.items())[0][0]
# select an event id
# TODO: calculate location similarity using latitude and longitude
m_lat_a = 90.0 - float(event_id_info_dict[key_selected][1])
m_lon_a = float(event_id_info_dict[key_selected][2])

loc_similarity = {}
for key in event_id_info_dict:
    m_lat_b = 90.0 - float(event_id_info_dict[key][1])
    m_lon_b = float(event_id_info_dict[key][2])
    if m_lat_a == m_lat_b and m_lon_a == m_lon_b:
        loc_similarity[key] = 1
    else:
        loc_similarity[key] = exp(-0.5 * pow((pi * 6371.004 * acos(sin(m_lat_a) * sin(m_lat_b) *
                                                                   cos(m_lon_a - m_lon_b) +
                                                                   cos(m_lat_a) * cos(m_lat_b)) / 180), 2.0))
sorted_loc_similarity_list = sorted(loc_similarity.items(), key=lambda d: d[1], reverse=True)

most_similar_on_loc = set()
for i in range(6000):
    # print(sorted_loc_similarity_list[i][0])
    # print(sorted_loc_similarity_list[i][1])
    most_similar_on_loc.add(sorted_loc_similarity_list[i][0])

# event_id_info_dict_after_washing[key_selected] = event_id_info_dict[key_selected]
for key in event_id_info_dict:
    if key != key_selected and event_id_info_dict[key_selected][0] == event_id_info_dict[key][0] \
                and key in most_similar_on_loc and key in event_users_dict:
        event_id_info_dict_after_washing[key] = event_id_info_dict[key]
print(len(event_id_info_dict_after_washing))

with open('event_info_after_washing.txt', 'w') as f:
    for key in event_id_info_dict_after_washing:
        event = key
        category = event_id_info_dict_after_washing[key][0]
        latitude = event_id_info_dict_after_washing[key][1]
        longitude = event_id_info_dict_after_washing[key][2]
        organizer = event_id_info_dict_after_washing[key][3]
        f.write(event + '\t' + category + '\t' + latitude + '\t' + longitude + '\t' + organizer + '\n')

user_set = set()
with open('event_user_pairs_after_washing.txt', 'w+') as f:
    for key in event_id_info_dict_after_washing:
        if key in event_users_dict:
            i = 0
            for user in event_users_dict[key]:
                if i < 10:
                    user_set.add(user)
                    f.write(key + '\t' + user + '\n')
                    i += 1

print(len(user_set))
with open('final_user_id.txt', 'w+') as f:
    for user in user_set:
        f.write(user + '\n')
