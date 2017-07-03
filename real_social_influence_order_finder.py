# coding=utf8

import tid_vid_dict_generator as dict_generator
from user_and_friends_dict_generator import FriendshipGenerator


def get_real_social_influence_order():
    g = FriendshipGenerator()
    users_friends_dict = g.generate_friendship()
    event_users_dict = dict_generator.generate_event_users_dict()
    for key in event_users_dict:
        event_users_dict[key] = set(event_users_dict[key])

    event_users_influence_dict = {}
    for key in event_users_dict:
        event_users_influence_dict[key] = {}

    event_sorted_users_dict = {}
    for key in event_users_dict:
        for present in event_users_dict[key]:

            #更改过
            event_users_influence_dict[key][present] = len(event_users_dict[key] & users_friends_dict[present])
        sorted_users_by_influence = sorted(event_users_influence_dict[key].items(), key=lambda d: d[1], reverse=True)
        sorted_user_list = []
        for i in sorted_users_by_influence:
            sorted_user_list.append(i[0])
        event_sorted_users_dict[key] = sorted_user_list

    # change tid into vid
    user_tid_vid_dict = dict_generator.generate_user_id_dict()
    event_tid_vid_dict = dict_generator.generate_event_id_dict()

    event_vid_sorted_user_vid_dict = {}
    for key in event_sorted_users_dict:
        user_vid_list = []
        for user in event_sorted_users_dict[key]:
            user_vid_list.append(user_tid_vid_dict[user])
        event_vid_sorted_user_vid_dict[event_tid_vid_dict[key]] = user_vid_list

    return event_vid_sorted_user_vid_dict

if __name__ == '__main__':
    influence_order = get_real_social_influence_order()
    print influence_order
