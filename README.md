# Chinese-Text-Classification-Pytorch
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer, 基于pytorch，开箱即用。

## 介绍

项目代码介绍，请移步[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)

## 环境
```
python 3.7
pytorch 1.1
tqdm
sklearn
tensorboardX
```

## 使用说明

本项目主要是增加了一部分对抗训练相关的代码，当前只支持TextCNN模型的对抗训练

```
# 训练并测试：
# TextCNN 无对抗训练
python run.py --model TextCNN

# TextCNN 移除dropout
python run.py --model TextCNN --no_dropout 1

# TextCNN 使用FGSM方法对抗训练
python run.py --model TextCNN --adversarial fgsm

# TextCNN 使用PGD方法对抗训练
python run.py --model TextCNN --adversarial pgd

# TextCNN 使用FreeAT方法对抗训练
python run.py --model TextCNN --adversarial freeat

```

调参相关的运行命令见文件 `adversarial_experiment.sh`
