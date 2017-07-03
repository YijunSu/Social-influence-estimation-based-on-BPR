# coding=utf-8

"""
    @author:    Jack Ome
    @date:      2016-07-08
    @desc:      find neighbors of each event according to score of category, location, and organizer
"""

from math import *
import tid_vid_dict_generator as dict_generator


class ItemNeighbor(object):
    def __init__(self, n_limit):
        self.items_neighbors = {}
        # item_neighbors is a dict and its key is the id of the event(int)
        # its value is a set of the neighbors of the event(a set of int)
        self.neighbor_limit = n_limit

    def get_neighbors(self):

        # TODO: read info from file and store them in the list event_list
        event_list = []
        # element of event_list includes event id(int), category(str), latitude and latitude(both float)
        # and organizer id(str)
        with open('event_info_after_washing.txt') as f:
            for line in f:
                event_data = line.split()
                event_data[0] = int(event_data[0])
                event_data[2] = float(event_data[2])
                event_data[3] = float(event_data[3])
                event_list.append(event_data)
        f.close()

        event_id_dict = dict_generator.generate_event_id_dict()
        # event_id_dict is a dict and key is string while value is int, both signify the event uniquely

        # TODO: find neighbors and return a dict, key is event id(int) and value is a set( element is event id, int)
        for item in event_list:
            self.items_neighbors[event_id_dict[str(item[0])]] = set()
            loc_similarity = {}
            organizer_similarity = {}
            category_similarity = {}
            similarity = {}
            for i in event_list:
                if i[0] != item[0]:
                    if i[1] == item[1]:
                        category_similarity[i[0]] = 1.0
                    else:
                        category_similarity[i[0]] = 0

                    if i[4] == item[4]:
                        organizer_similarity[i[0]] = 1.0
                    else:
                        organizer_similarity[i[0]] = 0

                    # TODO: calculate the similarity between each pair of events
                    m_lat_a = 90.0 - item[2]
                    m_lat_b = 90.0 - i[2]
                    m_lon_a = item[3]
                    m_lon_b = i[3]
                    if m_lat_a == m_lat_b and m_lon_a == m_lon_b:
                        loc_similarity[i[0]] = 1
                    else:
                        loc_similarity[i[0]] = exp(-0.5 * pow((pi * 6371.004 *
                                                               acos(sin(m_lat_a) * sin(m_lat_b) *
                                                                    cos(m_lon_a - m_lon_b) + cos(m_lat_a) *
                                                                    cos(m_lat_b))/180), 2.0))

                    # TODO: calculate total similarity
                    similarity[i[0]] = category_similarity[i[0]] + loc_similarity[i[0]] + organizer_similarity[i[0]]

            # TODO: sort the dict according to total similarity(score)
            sorted_similarity_list = sorted(similarity.items(), key=lambda d: d[1], reverse=True)

            for i in range(self.neighbor_limit):
                self.items_neighbors[event_id_dict[str(item[0])]].add(event_id_dict[str(sorted_similarity_list[i][0])])

        return self.items_neighbors

if __name__ == '__main__':
    neighbor_num_limit = 10
    n = ItemNeighbor(neighbor_num_limit)
    items_neighbors = n.get_neighbors()
    '''
    none_neighbor_num = 0
    for key in items_neighbors:
        if len(items_neighbors[key]) == 0:
            none_neighbor_num += 1
    print(none_neighbor_num)
    '''
