# Evaluation

## For evaluation: 

**Inception Score, FID**: 

We use the evaluation script from: https://github.com/v-iashin/SpecVQGAN

We transform ours spectrogram (mel_num: 128) to the standard spectrogram setting (mel_num: 80) in [SpecVQGAN](https://github.com/v-iashin/SpecVQGAN).

Using the **transform_spec.py** script under this folder.


## Align ACC:

Download our Eval_Classifier on HuggingFace. 

> Put it here: inference/diff_foley_ckpt/eval_classifier.ckpt

**Your can refer to the implementation of align_acc.py script and modify it to satisfy your need.**



Write down your generated audio dataset path in config/eval_classifier.yaml


> python align_acc.py