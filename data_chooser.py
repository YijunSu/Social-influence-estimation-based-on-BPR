# coding=utf-8

event_users_dict = {}
with open('event_user_pairs.txt') as f:
    for line in f:
        event, user = line.split()
        if event not in event_users_dict:
            event_users_dict[event] = set()
            event_users_dict[event].add(user)
        else:
            event_users_dict[event].add(user)
print(len(event_users_dict))
f.close()

event_users_dict_after_washing = {}
for key in event_users_dict:
    if 25 <= len(event_users_dict[key]) <= 30:
        event_users_dict_after_washing[key] = event_users_dict[key]
print(len(event_users_dict_after_washing))

final_event_list = []
with open('event_id_category_location_organizer.txt') as f:
    with open('event_info_after_washing.txt', 'w+') as f_w:
        for line in f:
            event, category, latitude, longitude, organizer = line.split()
            if event in event_users_dict_after_washing:
                final_event_list.append(event)
                f_w.write(event + '\t' + category + '\t' + latitude + '\t' + longitude + '\t' + organizer + '\n')
    f_w.close()
f.close()

with open('event_user_pairs_after_washing.txt', 'w+') as f:
    for key in event_users_dict_after_washing:
        if key in final_event_list:
            for user in event_users_dict_after_washing[key]:
                f.write(key + '\t' + user + '\n')
f.close()
