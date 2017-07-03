# coding=utf-8


class FriendshipGenerator(object):
    def __init__(self):
        self.users_and_friends = {}
        self.final_users = set()

    def generate_friendship(self):
        self.get_final_users()

        # TODO:initialize the users_and_friends
        for user in self.final_users:
            self.users_and_friends[user] = set()

        with open('user_and_friends.txt', 'r') as f:
            for line in f:
                user_id, friends = line.split('\t')
                if user_id in self.final_users:
                    friend_set = set()
                    if friends != '\n':
                        friends = friends.replace(',', '\t')
                        for u in friends.strip().split('\t'):
                            friend_set.add(u)
                    self.users_and_friends[user_id] = friend_set

        return self.users_and_friends

    def get_final_users(self):
        with open('final_user_id.txt', 'r') as f:
            for line in f:
                self.final_users.add(line.strip())

if __name__ == '__main__':
    g = FriendshipGenerator()
    friendship = g.generate_friendship()
