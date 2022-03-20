# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from adversarial.BaseAttacker import BaseAttacker


class Attacker(BaseAttacker):

    def __init__(self, model, optimizer, epsilon=0.01, **kwargs):
        super(Attacker, self).__init__(model)
        self.optimizer = optimizer
        self.epsilon = epsilon

    def train(self, trains, labels):
        outputs = self.model(trains)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # 对抗训练
        embedding = self.get_embedding()
        embedding_data_backup = embedding.data.clone()
        r_at = self.epsilon * torch.sign(embedding.grad)
        embedding.data.add_(r_at)

        outputs_adv = self.model(trains)
        loss_adv = F.cross_entropy(outputs_adv, labels)
        loss_adv.backward()
        embedding.data = embedding_data_backup

        # 更新模型
        self.optimizer.step()
        self.model.zero_grad()
        return outputs, loss
