"""
Evaluation
"""
import numpy as np
import pandas as pd

from agent import DDPGAgent
from config import Config
from data import DataLoader
from env import Env
from utils import OUNoise


class Evaluator(object):
    """
    Evaluator
    """

    def __init__(self, config: Config):
        """
        Initialize Evaluator

        :param config: configurations
        """
        self.config = config

    def evaluate(self, agent: DDPGAgent, df_eval: pd.DataFrame(), mode: str, top_K=5):
        """
        Evaluate the agent

        :param agent: agent
        :param df_eval: evaluation data
        :param mode: in ['user', 'group']
        :param top_K: length of the recommendation list
        :return: avg_recall_score, avg_ndcg_score
        """
        recall_scores = []
        ndcg_scores = []

        for _, row in df_eval.iterrows():
            group = row['group']
            history = row['history']
            item_true = row['action']
            item_candidates = row['negative samples'] + [item_true]
            np.random.shuffle(item_candidates)

            state = [group] + history
            items_pred = agent.get_action(state=state, item_candidates=item_candidates, top_K=top_K)

            recall_score = 0
            ndcg_score = 0

            for k, item in enumerate(items_pred):
                if item == item_true:
                    recall_score = 1
                    ndcg_score = np.log2(2) / np.log2(k + 2)
                    break

            recall_scores.append(recall_score)
            ndcg_scores.append(ndcg_score)

        avg_recall_score = float(np.mean(recall_scores))
        avg_ndcg_score = float(np.mean(ndcg_scores))
        print('%s: Recall@%d = %.4f, NDCG@%d = %.4f' % (mode.capitalize(), top_K, avg_recall_score,
                                                        top_K, avg_ndcg_score))
        return avg_recall_score, avg_ndcg_score


if __name__ == '__main__':
    config = Config()
    dataloader = DataLoader(config)
    rating_matrix_train = dataloader.load_rating_matrix(dataset_name='train')
    df_eval_user_val = dataloader.load_eval_data(dataset_name='val', mode='user')
    df_eval_group_val = dataloader.load_eval_data(dataset_name='val', mode='group')
    env = Env(config=config, rating_matrix=rating_matrix_train, dataset_name='train')
    noise = OUNoise(config=config)
    agent = DDPGAgent(config=config, noise=noise, group2members_dict=dataloader.group2members_dict, verbose=True)
    evaluator = Evaluator(config=config)
    evaluator.evaluate(agent, df_eval_user_val, mode='user', top_K=5)
    evaluator.evaluate(agent, df_eval_group_val, mode='group', top_K=5)
