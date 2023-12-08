# Evaluation

## For evaluation: 

**Inception Score, FID**: 

We use the evaluation script from: https://github.com/v-iashin/SpecVQGAN

We transform ours spectrogram (mel_num: 128) to the standard spectrogram setting (mel_num: 80) in [SpecVQGAN](https://github.com/v-iashin/SpecVQGAN).

Using the **transform_spec.py** script under this folder.
