# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from adversarial.BaseAttacker import BaseAttacker


class Attacker(BaseAttacker):
    def __init__(self, model, optimizer, epsilon=0.01, alpha=0.1, K=3, **kwargs):
        super(Attacker, self).__init__(model)
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.alpha = alpha
        self.K = K

    def train(self, trains, labels):
        outputs = self.model(trains)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # 对抗训练
        embedding = self.get_embedding()
        grad_backup = embedding.grad.clone()
        emb_backup = embedding.data.clone()

        for k in range(self.K):
            norm = torch.norm(embedding.grad)
            if norm != 0:
                r_at = self.alpha * embedding.grad / norm
                embedding.data.add_(r_at)

                # 映射回球面
                r = embedding.data - emb_backup
                norm = torch.norm(r)
                if norm > self.epsilon:
                    r = self.epsilon * r / norm
                embedding.data = emb_backup + r

            if k != self.K - 1:
                self.model.zero_grad()
            else:
                embedding.grad = grad_backup

            outputs_adv = self.model(trains)
            loss_adv = F.cross_entropy(outputs_adv, labels)
            loss_adv.backward()

        embedding.data = emb_backup

        # 更新模型
        self.optimizer.step()
        self.model.zero_grad()
        return outputs, loss
