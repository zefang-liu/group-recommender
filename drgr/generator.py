"""
Processing Datasets
"""
import os
import shutil
import zipfile

import numpy as np
import scipy.sparse as sp
from collections import Counter


class GroupGenerator(object):
    """
    Group Data Generator
    """
    def __init__(self, data_path, output_path, rating_threshold, num_groups,
                 group_sizes, min_num_ratings, train_ratio, val_ratio,
                 negative_sample_size, verbose=False):
        self.rating_threshold = rating_threshold
        self.negative_sample_size = negative_sample_size
        users_path = os.path.join(data_path, 'users.dat')
        items_path = os.path.join(data_path, 'movies.dat')
        ratings_path = os.path.join(data_path, 'ratings.dat')

        users = self.load_users_file(users_path)
        items = self.load_items_file(items_path)
        rating_mat, timestamp_mat = \
            self.load_ratings_file(ratings_path, max(users), max(items))

        groups, group_ratings, groups_rated_items_dict, groups_rated_items_set = \
            self.generate_group_ratings(users, rating_mat, timestamp_mat,
                                        num_groups=num_groups,
                                        group_sizes=group_sizes,
                                        min_num_ratings=min_num_ratings)
        members, group_ratings_train, group_ratings_val, group_ratings_test, \
            group_negative_items_val, group_negative_items_test, \
            user_ratings_train, user_ratings_val, user_ratings_test, \
            user_negative_items_val, user_negative_items_test = \
            self.split_ratings(group_ratings, rating_mat, timestamp_mat,
                               groups, groups_rated_items_dict, groups_rated_items_set,
                               train_ratio=train_ratio, val_ratio=val_ratio)

        groups_path = os.path.join(output_path, 'groupMember.dat')
        group_ratings_train_path = os.path.join(output_path, 'groupRatingTrain.dat')
        group_ratings_val_path = os.path.join(output_path, 'groupRatingVal.dat')
        group_ratings_test_path = os.path.join(output_path, 'groupRatingTest.dat')
        group_negative_items_val_path = os.path.join(output_path, 'groupRatingValNegative.dat')
        group_negative_items_test_path = os.path.join(output_path, 'groupRatingTestNegative.dat')
        user_ratings_train_path = os.path.join(output_path, 'userRatingTrain.dat')
        user_ratings_val_path = os.path.join(output_path, 'userRatingVal.dat')
        user_ratings_test_path = os.path.join(output_path, 'userRatingTest.dat')
        user_negative_items_val_path = os.path.join(output_path, 'userRatingValNegative.dat')
        user_negative_items_test_path = os.path.join(output_path, 'userRatingTestNegative.dat')

        self.save_groups(groups_path, groups)
        self.save_ratings(group_ratings_train, group_ratings_train_path)
        self.save_ratings(group_ratings_val, group_ratings_val_path)
        self.save_ratings(group_ratings_test, group_ratings_test_path)
        self.save_negative_samples(group_negative_items_val, group_negative_items_val_path)
        self.save_negative_samples(group_negative_items_test, group_negative_items_test_path)
        self.save_ratings(user_ratings_train, user_ratings_train_path)
        self.save_ratings(user_ratings_val, user_ratings_val_path)
        self.save_ratings(user_ratings_test, user_ratings_test_path)
        self.save_negative_samples(user_negative_items_val, user_negative_items_val_path)
        self.save_negative_samples(user_negative_items_test, user_negative_items_test_path)
        shutil.copyfile(src=os.path.join(data_path, 'movies.dat'), dst=os.path.join(output_path, 'movies.dat'))
        shutil.copyfile(src=os.path.join(data_path, 'users.dat'), dst=os.path.join(output_path, 'users.dat'))

        if verbose:
            num_group_ratings = len(group_ratings)
            num_user_ratings = len(user_ratings_train) + len(user_ratings_val) + len(user_ratings_test)
            num_rated_items = len(groups_rated_items_set)

            print('Save data: ' + output_path)
            print('# Users: ' + str(len(members)))
            print('# Items: ' + str(num_rated_items))
            print('# Groups: ' + str(len(groups)))
            print('# U-I ratings: ' + str(num_user_ratings))
            print('# G-I ratings: ' + str(num_group_ratings))
            print('Avg. # ratings / user: {:.2f}'.format(num_user_ratings / len(members)))
            print('Avg. # ratings / group: {:.2f}'.format(num_group_ratings / len(groups)))
            print('Avg. group size: {:.2f}'.format(np.mean(list(map(len, groups)))))

    def load_users_file(self, users_path):
        users = []

        with open(users_path, 'r') as file:
            for line in file.readlines():
                users.append(int(line.split('::')[0]))

        return users

    def load_items_file(self, items_path):
        items = []

        with open(items_path, 'r', encoding='iso-8859-1') as file:
            for line in file.readlines():
                items.append(int(line.split('::')[0]))

        return items

    def load_ratings_file(self, ratings_path, max_num_users, max_num_items):
        rating_mat = sp.dok_matrix((max_num_users + 1, max_num_items + 1),
                                   dtype=np.int)
        timestamp_mat = rating_mat.copy()

        with open(ratings_path, 'r') as file:
            for line in file.readlines():
                arr = line.replace('\n', '').split('::')
                user, item, rating, timestamp = \
                    int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])
                rating_mat[user, item] = rating
                timestamp_mat[user, item] = timestamp

        return rating_mat, timestamp_mat

    def generate_group_ratings(self, users, rating_mat, timestamp_mat,
                               num_groups, group_sizes, min_num_ratings):
        np.random.seed(0)
        groups = set()
        groups_ratings = []
        groups_rated_items_dict = {}
        groups_rated_items_set = set()

        while len(groups) < num_groups:
            group_id = len(groups) + 1

            while True:
                group = tuple(np.sort(
                    np.random.choice(users, np.random.choice(group_sizes),
                                     replace=False)))
                if group not in groups:
                    break

            pos_group_rating_counter = Counter()
            neg_group_rating_counter = Counter()
            group_rating_list = []
            group_rated_items = set()

            for member in group:
                _, items = rating_mat[member, :].nonzero()
                pos_items = [item for item in items
                             if rating_mat[member, item] >= self.rating_threshold]
                neg_items = [item for item in items
                             if rating_mat[member, item] < self.rating_threshold]
                pos_group_rating_counter.update(pos_items)
                neg_group_rating_counter.update(neg_items)

            for item, num_ratings in pos_group_rating_counter.items():
                if num_ratings == len(group):
                    timestamp = max([timestamp_mat[member, item]
                                     for member in group])
                    group_rated_items.add(item)
                    group_rating_list.append((group_id, item, 1, timestamp))

            for item, num_ratings in neg_group_rating_counter.items():
                if (num_ratings == len(group)) \
                        or (num_ratings + pos_group_rating_counter[item] == len(group)):
                    timestamp = max([timestamp_mat[member, item]
                                     for member in group])
                    group_rated_items.add(item)
                    group_rating_list.append((group_id, item, 0, timestamp))

            if len(group_rating_list) >= min_num_ratings:
                groups.add(group)
                groups_rated_items_dict[group_id] = group_rated_items
                groups_rated_items_set.update(group_rated_items)
                for group_rating in group_rating_list:
                    groups_ratings.append(group_rating)

        return list(groups), groups_ratings, groups_rated_items_dict, groups_rated_items_set

    def split_ratings(self, group_ratings, rating_mat, timestamp_mat,
                      groups, groups_rated_items_dict, groups_rated_items_set, train_ratio, val_ratio):
        num_group_ratings = len(group_ratings)
        num_train = int(num_group_ratings * train_ratio)
        num_test = int(num_group_ratings * (1 - train_ratio - val_ratio))

        group_ratings = \
            sorted(group_ratings, key=lambda group_rating: group_rating[-1])
        group_ratings_train = group_ratings[:num_train]
        group_ratings_val = group_ratings[num_train:-num_test]
        group_ratings_test = group_ratings[-num_test:]

        timestamp_split_train = group_ratings_train[-1][-1]
        timestamp_split_val = group_ratings_val[-1][-1]

        user_ratings_train = []
        user_ratings_val = []
        user_ratings_test = []

        members = set()
        users_rated_items_dict = {}

        for group in groups:
            for member in group:
                if member in members:
                    continue
                members.add(member)
                user_rated_items = set()
                _, items = rating_mat[member, :].nonzero()
                for item in items:
                    if item not in groups_rated_items_set:
                        continue
                    user_rated_items.add(item)
                    if rating_mat[member, item] >= self.rating_threshold:
                        rating_tuple = (member, item, 1,
                                        timestamp_mat[member, item])
                    else:
                        rating_tuple = (member, item, 0,
                                        timestamp_mat[member, item])
                    if timestamp_mat[member, item] <= timestamp_split_train:
                        user_ratings_train.append(rating_tuple)
                    elif timestamp_split_train < timestamp_mat[member, item] <= timestamp_split_val:
                        user_ratings_val.append(rating_tuple)
                    else:
                        user_ratings_test.append(rating_tuple)

                users_rated_items_dict[member] = user_rated_items

        np.random.seed(0)

        user_negative_items_val = self.get_negative_samples(
            user_ratings_val, groups_rated_items_set, users_rated_items_dict)
        user_negative_items_test = self.get_negative_samples(
            user_ratings_test, groups_rated_items_set, users_rated_items_dict)
        group_negative_items_val = self.get_negative_samples(
            group_ratings_val, groups_rated_items_set, groups_rated_items_dict)
        group_negative_items_test = self.get_negative_samples(
            group_ratings_test, groups_rated_items_set, groups_rated_items_dict)

        return members, group_ratings_train, group_ratings_val, group_ratings_test, \
            group_negative_items_val, group_negative_items_test, \
            user_ratings_train, user_ratings_val, user_ratings_test, \
            user_negative_items_val, user_negative_items_test

    def get_negative_samples(self, ratings, groups_rated_items_set, rated_items_dict):
        negative_items_list = []
        for sample in ratings:
            sample_id, item, _, _ = sample
            missed_items = groups_rated_items_set - rated_items_dict[sample_id]
            negative_items = \
                np.random.choice(list(missed_items), self.negative_sample_size,
                                 replace=(len(missed_items) < self.negative_sample_size))
            negative_items_list.append((sample_id, item, negative_items))
        return negative_items_list

    def save_groups(self, groups_path, groups):
        with open(groups_path, 'w') as file:
            for i, group in enumerate(groups):
                file.write(str(i + 1) + ' '
                           + ','.join(map(str, list(group))) + '\n')

    def save_ratings(self, ratings, ratings_path):
        with open(ratings_path, 'w') as file:
            for rating in ratings:
                file.write(' '.join(map(str, list(rating))) + '\n')

    def save_negative_samples(self, negative_items, negative_items_path):
        with open(negative_items_path, 'w') as file:
            for samples in negative_items:
                user, item, negative_items = samples
                file.write('({},{}) '.format(user, item)
                           + ' '.join(map(str, list(negative_items))) + '\n')


if __name__ == '__main__':
    data_folder_path = os.path.join('..', 'data')
    data_path = os.path.join(data_folder_path, 'MovieLens-1M')
    data_zip_path = os.path.join(data_folder_path, 'MovieLens-1M.zip')
    output_path = os.path.join(data_folder_path, 'MovieLens-Rand')

    if not os.path.exists(data_path):
        with zipfile.ZipFile(data_zip_path, 'r') as data_zip:
            data_zip.extractall(data_folder_path)
            print('Unzip file: ' + data_zip_path)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    group_generator = GroupGenerator(data_path, output_path,
                                     rating_threshold=4,
                                     num_groups=1000,
                                     group_sizes=[2, 3, 4, 5],
                                     min_num_ratings=20,
                                     train_ratio=0.7,
                                     val_ratio=0.1,
                                     negative_sample_size=100,
                                     verbose=True)
