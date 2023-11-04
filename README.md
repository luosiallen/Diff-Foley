# [NeurIPS 2023] Diff-Foley: Synchronized Video-to-Audio Synthesis with Latent Diffusion Models

Offical implementation of the NeurIPS 2023 paper: *[Diff-Foley: Synchronized Video-to-Audio Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2306.17203v1)*.


Project Page: https://diff-foley.github.io


## News
- (🔥New) 2023/11/5 **Diff-Foley Inference Pipeline** is released! See the 'Inference Usages'.
- (🔥New) 2023/11/5 **Diff-Foley Pretrained Model** is released! Download from Hugging Face 🤗 [here](https://huggingface.co/SimianLuo/Diff-Foley).
- Including: Stage1-CAVP, Stage2-LDM, Double-Guidance Classifier !!


## Inference Usages:
1. Open the `diff_foley_inference.ipynb` in `inference` folder.
2. Download the pretrained model foler `diff_foley_ckpt` from Hugging Face 🤗 [here](https://huggingface.co/SimianLuo/Diff-Foley) and place it under `inference` folder.
3. Run the `diff_foley_inference.ipynb`.


## Diff-Foley
<p align="center">
    <img src="teaser.png">
</p>

## BibTeX

```bibtex
@misc{luo2023difffoley, 
title={Diff-Foley: Synchronized Video-to-Audio Synthesis with Latent Diffusion Models}, 
author={Simian Luo and Chuanhao Yan and Chenxu Hu and Hang Zhao}, 
year={2023}, 
eprint={2306.17203}, 
archivePrefix={arXiv}, 
primaryClass={cs.SD} 
}
```

