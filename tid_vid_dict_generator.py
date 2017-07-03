# coding=utf-8

"""
    @author:   zhaomingxing
    @update:    2016.6.19
"""


def generate_user_id_dict():
    user_id_dict = {}
    with open('final_user_id.txt') as f:
        i = 0
        for u_line in f:
            u_result = u_line.split()
            if u_result[0] not in user_id_dict:
                user_id_dict[u_result[0]] = i
                i += 1
    f.close()
    return user_id_dict


def generate_event_id_dict():
    event_id_dict = {}
    with open('event_info_after_washing.txt') as f:
        i = 0
        for e_line in f:
            e_result = e_line.split()
            if e_result[0] not in event_id_dict:
                event_id_dict[e_result[0]] = i
                i += 1
    f.close()
    return event_id_dict

if __name__ == '__main__':
    user_dict = generate_user_id_dict()
    event_dict = generate_event_id_dict()
    print(user_dict)
    print(event_dict)


def generate_event_users_dict():
    event_users_dict = {}
    with open('event_user_pairs_after_washing.txt') as f:
        for line in f:
            event, user = line.split()
            if event not in event_users_dict:
                event_users_dict[event] = []
                event_users_dict[event].append(user)
            else:
                event_users_dict[event].append(user)
    f.close()
    return event_users_dict
