# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from adversarial.BaseAttacker import BaseAttacker


class Attacker(BaseAttacker):

    def __init__(self, model, optimizer, epsilon=0.01, K=4, **kwargs):
        super(Attacker, self).__init__(model)
        embedding = self.get_embedding()
        self.delta = torch.zeros_like(embedding.data)
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.K = K

    def train(self, trains, labels):
        # 初始化梯度
        embedding = self.get_embedding()
        embedding_data_backup = embedding.data.clone()

        for _ in range(self.K):
            # embedding.data = embedding_data_backup.clone() + self.delta
            embedding.data.add_(self.delta)
            outputs_adv = self.model(trains)
            loss_adv = F.cross_entropy(outputs_adv, labels)
            loss_adv.backward()

            self.delta += self.epsilon * torch.sign(embedding.grad)
            self.delta = torch.clamp(self.delta, -self.epsilon, self.epsilon)

            embedding.data = embedding_data_backup.clone()

            # 更新参数
            self.optimizer.step()
            self.model.zero_grad()

        return outputs_adv, loss_adv
