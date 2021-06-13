"""
Data
"""
import os
from collections import deque, defaultdict
from typing import Dict

import pandas as pd
from scipy.sparse import coo_matrix

from config import Config


class DataLoader(object):
    """
    Data Loader
    """

    def __init__(self, config: Config):
        """
        Initialize DataLoader

        :param config: configurations
        """
        self.config = config
        self.history_length = config.history_length
        self.item_num = self.get_item_num()
        self.user_num = self.get_user_num()
        self.group_num, self.total_group_num, self.group2members_dict, self.user2group_dict = self.get_groups()

        if not os.path.exists(self.config.saves_folder_path):
            os.mkdir(self.config.saves_folder_path)

    def get_item_num(self) -> int:
        """
        Get number of items

        :return: number of items
        """
        df_item = pd.read_csv(self.config.item_path, sep='::', index_col=0, engine='python')
        self.config.item_num = df_item.index.max()
        return self.config.item_num

    def get_user_num(self) -> int:
        """
        Get number of users

        :return: number of users
        """
        df_user = pd.read_csv(self.config.user_path, sep='::', index_col=0, engine='python')
        self.config.user_num = df_user.index.max()
        return self.config.user_num

    def get_groups(self):
        """
        Get number of groups and group members

        :return: group_num, total_group_num, group2members_dict, user2group_dict
        """
        df_group = pd.read_csv(self.config.group_path, sep=' ', header=None, index_col=None,
                               names=['GroupID', 'Members'])
        df_group['Members'] = df_group['Members']. \
            apply(lambda group_members: tuple(map(int, group_members.split(','))))
        group_num = df_group['GroupID'].max()

        users = set()
        for members in df_group['Members']:
            users.update(members)
        users = sorted(users)
        total_group_num = group_num + len(users)

        df_user_group = pd.DataFrame()
        df_user_group['GroupID'] = list(range(group_num + 1, total_group_num + 1))
        df_user_group['Members'] = [(user,) for user in users]
        df_group = df_group.append(df_user_group, ignore_index=True)
        group2members_dict = {row['GroupID']: row['Members'] for _, row in df_group.iterrows()}
        user2group_dict = {user: group_num + user_index + 1 for user_index, user in enumerate(users)}

        self.config.group_num = group_num
        self.config.total_group_num = total_group_num
        return group_num, total_group_num, group2members_dict, user2group_dict

    def load_rating_data(self, mode: str, dataset_name: str, is_appended=True) -> pd.DataFrame():
        """
        Load rating data

        :param mode: in ['user', 'group']
        :param dataset_name: name of the dataset in ['train', 'val', 'test']
        :param is_appended: True to append all datasets before this dataset
        :return: df_rating
        """
        assert (mode in ['user', 'group']) and (dataset_name in ['train', 'val', 'test'])
        rating_path = os.path.join(self.config.data_folder_path, mode + 'Rating' + dataset_name.capitalize() + '.dat')
        df_rating_append = pd.read_csv(rating_path, sep=' ', header=None, index_col=None,
                                       names=['GroupID', 'MovieID', 'Rating', 'Timestamp'])
        print('Read data:', rating_path)

        if is_appended:
            if dataset_name == 'train':
                df_rating = df_rating_append
            elif dataset_name == 'val':
                df_rating = self.load_rating_data(mode=mode, dataset_name='train')
                df_rating = df_rating.append(df_rating_append, ignore_index=True)
            else:
                df_rating = self.load_rating_data(mode=mode, dataset_name='val')
                df_rating = df_rating.append(df_rating_append, ignore_index=True)
        else:
            df_rating = df_rating_append

        return df_rating

    def _load_rating_matrix(self, df_rating: pd.DataFrame()):
        """
        Load rating matrix

        :param df_rating: rating data
        :return: rating_matrix
        """
        group_ids = df_rating['GroupID']
        item_ids = df_rating['MovieID']
        ratings = df_rating['Rating']
        rating_matrix = coo_matrix((ratings, (group_ids, item_ids)),
                                   shape=(self.total_group_num + 1, self.config.item_num + 1)).tocsr()
        return rating_matrix

    def load_rating_matrix(self, dataset_name: str):
        """
        Load group rating matrix

        :param dataset_name: name of the dataset in ['train', 'val', 'test']
        :return: rating_matrix
        """
        assert dataset_name in ['train', 'val', 'test']

        df_user_rating = self.user2group(self.load_rating_data(mode='user', dataset_name=dataset_name))
        df_group_rating = self.load_rating_data(mode='group', dataset_name=dataset_name)
        df_group_rating = df_group_rating.append(df_user_rating, ignore_index=True)
        rating_matrix = self._load_rating_matrix(df_group_rating)

        return rating_matrix

    def user2group(self, df_user_rating):
        """
        Change user ids to group ids

        :param df_user_rating: user rating
        :return: df_user_rating
        """
        df_user_rating['GroupID'] = df_user_rating['GroupID'].apply(lambda user_id: self.user2group_dict[user_id])
        return df_user_rating

    def _load_eval_data(self, df_data_train: pd.DataFrame(), df_data_eval: pd.DataFrame(),
                        negative_samples_dict: Dict[tuple, list]) -> pd.DataFrame():
        """
        Write evaluation data

        :param df_data_train: train data
        :param df_data_eval: evaluation data
        :param negative_samples_dict: one dictionary mapping (group_id, item_id) to negative samples
        :return: data for evaluation
        """
        df_eval = pd.DataFrame()
        last_state_dict = defaultdict(list)
        groups = []
        histories = []
        actions = []
        negative_samples = []

        for group_id, rating_group in df_data_train.groupby(['GroupID']):
            rating_group.sort_values(by=['Timestamp'], ascending=True, ignore_index=True, inplace=True)
            state = rating_group[rating_group['Rating'] == 1]['MovieID'].values.tolist()
            last_state_dict[group_id] = state[-self.config.history_length:]

        for group_id, rating_group in df_data_eval.groupby(['GroupID']):
            rating_group.sort_values(by=['Timestamp'], ascending=True, ignore_index=True, inplace=True)
            action = rating_group[rating_group['Rating'] == 1]['MovieID'].values.tolist()
            state = deque(maxlen=self.history_length)
            state.extend(last_state_dict[group_id])
            for item_id in action:
                if len(state) == self.config.history_length:
                    groups.append(group_id)
                    histories.append(list(state))
                    actions.append(item_id)
                    negative_samples.append(negative_samples_dict[(group_id, item_id)])
                state.append(item_id)

        df_eval['group'] = groups
        df_eval['history'] = histories
        df_eval['action'] = actions
        df_eval['negative samples'] = negative_samples

        return df_eval

    def load_negative_samples(self, mode: str, dataset_name: str):
        """
        Load negative samples

        :param mode: in ['user', 'group']
        :param dataset_name: name of the dataset in ['val', 'test']
        :return: negative_samples_dict
        """
        assert (mode in ['user', 'group']) and (dataset_name in ['val', 'test'])
        negative_samples_path = os.path.join(self.config.data_folder_path, mode + 'Rating'
                                             + dataset_name.capitalize() + 'Negative.dat')
        negative_samples_dict = {}

        with open(negative_samples_path, 'r') as negative_samples_file:
            for line in negative_samples_file.readlines():
                negative_samples = line.split()
                ids = negative_samples[0][1:-1].split(',')
                group_id = int(ids[0])
                if mode == 'user':
                    group_id = self.user2group_dict[group_id]
                item_id = int(ids[1])
                negative_samples = list(map(int, negative_samples[1:]))
                negative_samples_dict[(group_id, item_id)] = negative_samples

        return negative_samples_dict

    def load_eval_data(self, mode: str, dataset_name: str, reload=False):
        """
        Load evaluation data

        :param mode: in ['user', 'group']
        :param dataset_name: in ['val', 'test']
        :param reload: True to reload the dataset file
        :return: data for evaluation
        """
        assert (mode in ['user', 'group']) and (dataset_name in ['val', 'test'])
        exp_eval_path = os.path.join(self.config.saves_folder_path, 'eval_' + mode + '_' + dataset_name + '_'
                                     + str(self.config.history_length) + '.pkl')

        if reload or not os.path.exists(exp_eval_path):
            if dataset_name == 'val':
                df_rating_train = self.load_rating_data(mode=mode, dataset_name='train')
            else:
                df_rating_train = self.load_rating_data(mode=mode, dataset_name='val')
            df_rating_eval = self.load_rating_data(mode=mode, dataset_name=dataset_name, is_appended=False)

            if mode == 'user':
                df_rating_train = self.user2group(df_rating_train)
                df_rating_eval = self.user2group(df_rating_eval)

            negative_samples_dict = self.load_negative_samples(mode=mode, dataset_name=dataset_name)
            df_eval = self._load_eval_data(df_rating_train, df_rating_eval, negative_samples_dict)
            df_eval.to_pickle(exp_eval_path)
            print('Save data:', exp_eval_path)
        else:
            df_eval = pd.read_pickle(exp_eval_path)
            print('Load data:', exp_eval_path)

        return df_eval


if __name__ == '__main__':
    data_loader = DataLoader(Config())
