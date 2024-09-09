# Sigmoid Attention

This repo contains the code associated with [Theory, Analysis, and Best Practices for
Sigmoid Self-Attention](https://arxiv.org/abs/2409.04431)

## Components

The three components of this release are:

  - [FlashSigmoid](./flash_sigmoid): A hardware aware implementation of Sigmoid Attention.
  - [Optorch](./optorch): PyTorch-based functional implementation of standard optimizers.
  - [Attention Simulator](./attention_simulator): A research friendly codebase for diagnosing and debugging attention.

## Installation

See the `README.md` in the corresponding component for installation and usage instructions.  

We provide a convenience installation helper for all three packages:
```bash
# Create an environment for sigmoid attention, if not done already.
conda create -n sigmoid-attn-py310 python=3.10
conda activate sigmoid-attn-py310

# Setup Flashsigmoid -> Optorch -> Attention Simulator.
bash setup.bash
```

## Performance

|                                      Forward pass kernels on H100.                                       |                                Backward pass kernels on H100.                                |
|:--------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
| ![Sigmoid vs. Softmax Forward Kernels](./figures/H100_noalibi_FWD_Full_17.39_0.07_Causal_18.76_0.06.png) | ![Sigmoid vs. Softmax Backward Kernels](./figures/H100_noalibi_BWD_Full_2.7_0.06_Causal_6.19_0.06.png) |


|                   Train losses comparing SigmoidAttn with SoftmaxAttn.                   |
|:----------------------------------------------------------------------------------------:|
| ![SigmoidAttn vs. SoftmaxAttn Train Losses](./figures/train_nll_softmax_vs_sigmoid.png) |



## Citation
If you find this work useful in your research, please cite:
```
@misc{ramapuram2024theoryanalysisbestpractices,
      title={Theory, Analysis, and Best Practices for Sigmoid Self-Attention},
      author={Jason Ramapuram and Federico Danieli and Eeshan Dhekane and Floris Weers and Dan Busbridge and Pierre Ablin and Tatiana Likhomanenko and Jagrit Digani and Zijin Gu and Amitis Shidani and Russ Webb},
      year={2024},
      eprint={2409.04431},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.04431},
}
```
