This subdirectory hosts codebase we used for image density modeling experiments of NanoFlow, with Glow as the base architecture.

The repo is based on a hard-fork of [Glow-PyTorch] (courtesy of [@y0ast]) and provides the parameterization scheme of NanoFlow.



## Setup and run

The code has minimal dependencies. You need python 3.6+ and up to date versions of:

```
pytorch (tested on 1.1.0)
torchvision
pytorch-ignite
tqdm
```

To install in a local conda:

```
conda install pytorch torchvision pytorch-ignite tqdm -c pytorch
```

**To train your own model:**

```
python train.py --download
```

Will download the CIFAR10 dataset for you, and start training. The defaults are tested on a `1080Ti`, Glow is a memory hungry model and it might be necessary to tune down the model size for your specific GPU. The output files will be send to `output/`.

Everything is configurable through command line arguments, see

```
python train.py --help
```

for what is possible.


## Example commands 
Glow (256 channels)

`python train.py --seed 1234 --output_dir output/glow 
--model_type glow --hidden_channels 256 --epochs 3000`

Glow-large

`python train.py --seed 1234 --output_dir output/glow-large 
--model_type glow_large --hidden_channels 256 --epochs 3000`

NanoFlow-Naive

`python train.py --seed 1234 --output_dir output/nanoflow-naive 
--model_type nanoflow_naive --hidden_channels 256 --epochs 3000`

NanoFlow-Decomp

`python train.py --seed 1234 --output_dir output/nanoflow-decomp 
--model_type nanoflow_decomp --hidden_channels 256 --epochs 3000`

NanoFlow

`python train.py --seed 1234 --output_dir output/nanoflow 
--model_type nanoflow --hidden_channels 256 --flow_embed_dim 1 --epochs 3000`

NanoFlow (K=48)

`python train.py --seed 1234 --output_dir output/nanoflow-k48
--model_type nanoflow --hidden_channels 256 --flow_embed_dim 3 --K 48 --epochs 3000`

[Glow-PyTorch]:  https://github.com/y0ast/Glow-PyTorch

[@y0ast]: https://github.com/y0ast

