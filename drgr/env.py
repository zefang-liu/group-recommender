"""
Environment
"""
import os

import gym
import numpy as np
from scipy.sparse.csr import csr_matrix
from sklearn.decomposition import NMF

from config import Config
from data import DataLoader


class Env(gym.Env):
    """
    Environment for the recommender system
    https://github.com/openai/gym/blob/master/gym/core.py
    """
    metadata = {'render.modes': ['human']}
    reward_range = (0, 1)

    def __init__(self, config: Config, rating_matrix: csr_matrix, dataset_name: str):
        """
        Initialize Env

        :param config: configurations
        :param rating_matrix: rating matrix
        :param dataset_name: dataset name
        """
        assert dataset_name in ['train', 'val', 'test']
        self.config = config
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(config.action_size,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(config.state_size,))

        self.rating_matrix = rating_matrix
        rating_matrix_coo = rating_matrix.tocoo()
        rating_matrix_rows = rating_matrix_coo.row
        rating_matrix_columns = rating_matrix_coo.col
        self.rating_matrix_index_set = set(zip(*(rating_matrix_rows, rating_matrix_columns)))
        self.env_name = 'env_' + dataset_name + '_' + str(self.config.env_n_components) + '.npy'
        self.env_path = os.path.join(config.saves_folder_path, self.env_name)

        self.rating_matrix_pred = None
        self.load_env()

        self.state = None
        self.reset()

    def load_env(self):
        """
        Load environment
        """
        if not os.path.exists(self.env_path):
            env_model = NMF(n_components=self.config.env_n_components, init='random', tol=self.config.env_tol,
                            max_iter=self.config.env_max_iter, alpha=self.config.env_alpha, verbose=True,
                            random_state=0)
            print('-' * 50)
            print('Train environment:')
            W = env_model.fit_transform(X=self.rating_matrix)
            H = env_model.components_
            self.rating_matrix_pred = W @ H
            print('-' * 50)
            np.save(self.env_path, self.rating_matrix_pred)
            print('Save environment:', self.env_path)
        else:
            self.rating_matrix_pred = np.load(self.env_path)
            print('Load environment:', self.env_path)

    def reset(self):
        """
        Reset the environment

        :return: state
        """
        while True:
            group_id = np.random.choice(range(1, self.config.total_group_num + 1))
            nonzero_row, nonzero_col = self.rating_matrix[group_id, :].nonzero()
            if len(nonzero_col) >= self.config.history_length:
                break
        history = np.random.choice(nonzero_col, size=self.config.history_length, replace=False).tolist()
        self.state = [group_id] + history
        return self.state

    def step(self, action: int):
        """
        Take one action to the environment

        :param action: action
        :return: new_state, reward, done, info
        """
        group_id = self.state[0]
        history = self.state[1:]

        if (group_id, action) in self.rating_matrix_index_set:
            reward = self.rating_matrix[group_id, action]
        else:
            reward_probability = self.rating_matrix_pred[group_id, action]
            reward = np.random.choice(self.config.rewards, p=[1 - reward_probability, reward_probability])

        if reward > 0:
            history = history[1:] + [action]

        new_state = [group_id] + history
        self.state = new_state
        done = False
        info = {}

        return new_state, reward, done, info

    def render(self, mode='human'):
        """
        Render the environment

        :param mode: mode
        """
        pass


if __name__ == '__main__':
    np.random.seed(0)
    config = Config()
    dataloader = DataLoader(config)
    env = Env(config=config, rating_matrix=dataloader.load_rating_matrix(dataset_name='train'), dataset_name='train')
