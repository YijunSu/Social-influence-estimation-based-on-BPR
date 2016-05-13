from math import *


class ItemNeighbor(object):
    def __init__(self, loc_neighbor_num, n_limit):
        self.items_neighbors = {}
        # item_neighbors is a dict and its key is the id of the event
        # its value is a set of the neighbors of the event
        self.top_n = loc_neighbor_num
        self.neighbor_limit = n_limit

    def get_neighbors(self):
        items_neighbors_on_cat = {}
        items_neighbors_on_loc = {}
        items_neighbors_on_org = {}
        event_list = []
        m = 1
        with open('event_info.txt') as f:
            # TODO: find the neighbors of each item and store the info in the dict
            for line in f:
                event_data = line.split()
                event_data[0] = int(event_data[0])
                event_data[2] = float(event_data[2])
                event_data[3] = float(event_data[3])
                event_data[4] = int(event_data[4])
                event_list.append(event_data)

        for item in event_list:
            event_id_dict = {}
            with open('event_file.txt') as ef:
                for e_line in ef:
                    e_result = e_line.split()
                    event_id_dict[int(e_result[0])] = int(e_result[1])
            m += 1
            items_neighbors_on_cat[item[0]] = set()
            items_neighbors_on_org[item[0]] = set()
            items_neighbors_on_loc[item[0]] = set()
            loc_similarity = {}
            for i in event_list:
                if i[0] != item[0]:
                    if i[1] == item[1]:
                        items_neighbors_on_cat[item[0]].add(i[0])
                    if i[4] == item[4]:
                        items_neighbors_on_org[item[0]].add(i[0])
                    """ TODO: calculate the distance through longitude and latitude
                    store the distance data for sort to find the top-N neighbors
                    """
                    m_lat_a = 90.0 - item[2]
                    m_lat_b = 90.0 - i[2]
                    m_lon_a = item[3]
                    m_lon_b = i[3]
                    if m_lat_a == m_lat_b and m_lon_a == m_lon_b:
                        loc_similarity[i[0]] = 1
                    else:
                        loc_similarity[i[0]] = exp(-0.5 * pow((pi * 6371004 *
                                                               acos(sin(m_lat_a) * sin(m_lat_b) *
                                                                    cos(m_lon_a - m_lon_b) + cos(m_lat_a) *
                                                                    cos(m_lat_b))/180), 2.0))
            sorted_loc_similarity_list = sorted(loc_similarity.iteritems(), key=lambda d: d[1], reverse=True)
            for i in range(self.top_n):
                items_neighbors_on_loc[item[0]].add(sorted_loc_similarity_list[i][1])

            item_neighbor_set = (items_neighbors_on_cat[item[0]] & items_neighbors_on_loc[item[0]]) |\
                (items_neighbors_on_loc[item[0]] & items_neighbors_on_org[item[0]]) |\
                (items_neighbors_on_org[item[0]] & items_neighbors_on_cat[item[0]])
            v_id_set = set()
            counter = 0
            for elem in item_neighbor_set:
                if counter < self.neighbor_limit:
                    v_id_set.add(event_id_dict[elem])
            self.items_neighbors[event_id_dict[item[0]]] = v_id_set
        return self.items_neighbors

if __name__ == '__main__':
    top_n = 100
    neighbor_num_limit = 10
    n = ItemNeighbor(top_n, neighbor_num_limit)
    items_neighbors = n.get_neighbors()
    none_neighbor_num = 0
    for key in items_neighbors:
        if len(items_neighbors[key]) == 0:
            none_neighbor_num += 1
    print(none_neighbor_num)
