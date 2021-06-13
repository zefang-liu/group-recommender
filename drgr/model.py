"""
Models
"""
from typing import Tuple

import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Actor Network
    """

    def __init__(self, embedded_state_size: int, action_weight_size: int, hidden_sizes: Tuple[int]):
        """
        Initialize Actor

        :param embedded_state_size: embedded state size
        :param action_weight_size: embedded action size
        :param hidden_sizes: hidden sizes
        """
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedded_state_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_weight_size),
        )

    def forward(self, embedded_state):
        """
        Forward

        :param embedded_state: embedded state
        :return: action weight
        """
        return self.net(embedded_state)


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, embedded_state_size: int, embedded_action_size: int, hidden_sizes: Tuple[int]):
        """
        Initialize Critic

        :param embedded_state_size: embedded state size
        :param embedded_action_size: embedded action size
        :param hidden_sizes: hidden sizes
        """
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedded_state_size + embedded_action_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, embedded_state, embedded_action):
        """
        Forward

        :param embedded_state: embedded state
        :param embedded_action: embedded action
        :return: Q value
        """
        return self.net(torch.cat([embedded_state, embedded_action], dim=-1))


class Embedding(nn.Module):
    """
    Embedding Network
    """

    def __init__(self, embedding_size: int, user_num: int, item_num: int):
        """
        Initialize Embedding

        :param embedding_size: embedding size
        :param user_num: number of users
        :param item_num: number of items
        """
        super(Embedding, self).__init__()
        self.user_embedding = nn.Embedding(user_num + 1, embedding_size)
        self.item_embedding = nn.Embedding(item_num + 1, embedding_size)
        self.user_attention = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )
        self.user_softmax = nn.Softmax(dim=-1)

    def forward(self, group_members, history):
        """
        Forward

        :param group_members: group members
        :param history: browsing history of items
        :return: embedded state
        """
        embedded_group_members = self.user_embedding(group_members)
        group_member_attentions = self.user_softmax(self.user_attention(embedded_group_members))
        embedded_group = torch.squeeze(torch.inner(group_member_attentions.T, embedded_group_members.T))
        embedded_history = torch.flatten(self.item_embedding(history), start_dim=-2)
        embedded_state = torch.cat([embedded_group, embedded_history], dim=-1)
        return embedded_state
