# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--adversarial', type=str, help='对抗训练方法')
parser.add_argument('--adv_epsilon', type=float, default=0.01, help='对抗训练参数 epsilon')
parser.add_argument('--adv_alpha', type=float, default=0.05, help='对抗训练参数 alpha')
parser.add_argument('--adv_k', type=int, default=4, help='对抗训练参数 K')
parser.add_argument('--no_dropout', type=bool, default=False, help='是否去掉dropout操作')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    config.no_dropout = args.no_dropout
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)

    # adversarial train
    adversarial = {}
    if args.adversarial:
        if args.adversarial.lower() == 'fgsm':
            adversarial = {
                'name': 'FGSM',
                'epsilon': args.adv_epsilon,
            }
        if args.adversarial.lower() == 'pgd':
            adversarial = {
                'name': 'PGD',
                'epsilon': args.adv_epsilon,
                'alpha': args.adv_alpha,
                'K': args.adv_k,
            }
        if args.adversarial.lower() == 'freeat':
            adversarial = {
                'name': 'FreeAT',
                'epsilon': args.adv_epsilon,
                'K': args.adv_k,
            }
    train(config, model, train_iter, dev_iter, test_iter, adversarial)
