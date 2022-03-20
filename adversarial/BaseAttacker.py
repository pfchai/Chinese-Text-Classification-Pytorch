# -*- coding: utf-8 -*-


class BaseAttacker():

    def __init__(self, model):
        self.model = model

    def get_embedding(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'embedding' in name:
                return param
        raise

    def attack_train(self, trains, labels, **kwargs):
        raise NotImplemented
