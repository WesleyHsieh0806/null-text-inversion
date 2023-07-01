# Null-Text Inversion for Editing Real Images

### [Project Page](https://null-text-inversion.github.io/)&ensp;&ensp;&ensp;[Paper](https://arxiv.org/abs/2211.09794)



Null-text inversion enables intuitive text-based editing of **real images** with the Stable Diffusion model. We use an initial DDIM inversion as an anchor for our optimization which only tunes the null-text embedding used in classifier-free guidance.


![teaser](docs/null_text_teaser.png)

## Setup

This code was tested with Python 3.8, [Pytorch](https://pytorch.org/) 1.11 using pre-trained models through [huggingface / diffusers](https://github.com/huggingface/diffusers#readme).
Specifically, we implemented our method over  [Latent Diffusion](https://huggingface.co/CompVis/ldm-text2im-large-256) and  [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4).
Additional required packages are listed in the requirements file.
The code was tested on a Tesla V100 16GB but should work on other cards with at least **12GB** VRAM.
```
conda env create -f environment.yaml
conda activate null-text-inversion
pip install -r requirements.txt
```

## Null-Text Inversion
```bash
python null_text_inversion.py --image /data3/chengyeh/DragDiffusion-Experiment/TEdBench/originals/bird/bird.png \
    --prompt "a photo of a bird" --device cuda:3 --output_dir /data3/chengyeh/DragDiffusion-Experiment/TEdBench/originals/bird/Null-Text-Inversion
```

This outputs the following files in `--output_dir`
  1. a `results.png` containing original image, image passed through AutoEncoder, and image inverted by Null-Text inversion.
  2. a `uncond_embeddings.pth` checkpoint containing optimized unconditional embeddings for all time steps.
  3.  a `args.json` with all arguments used in this run.

## Editing Real Images

Prompt-to-Prompt editing of real images by first using Null-text inversion is provided in this [**Notebooke**][null_text].

``` bibtex
@article{mokady2022null,
  title={Null-text Inversion for Editing Real Images using Guided Diffusion Models},
  author={Mokady, Ron and Hertz, Amir and Aberman, Kfir and Pritch, Yael and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2211.09794},
  year={2022}
}
```


## Disclaimer

This is not an officially supported Google product.

[p2p-ldm]: prompt-to-prompt_ldm.ipynb
[p2p-stable]: prompt-to-prompt_stable.ipynb
[null_text]: null_text_w_ptp.ipynb
