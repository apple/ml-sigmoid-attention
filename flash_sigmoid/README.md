# FlashSigmoid

This repository contains the code for FlashSigmoid approach from the paper: [Theory, Analysis, and Best Practices for
Sigmoid Self-Attention](https://arxiv.org/abs/2409.04431).

```python3
# (Softmax) Attention
out = softmax(q @ k.T / sqrt(d)) @ v

# Sigmoid Attention 
out = sigmoid(q @ k.T / sqrt(d) + b) @ v  # b: scalar
```

  - FlashSigmoid is motivated by the efficient hardware aware implementation of [FlashAttention2](https://github.com/Dao-AILab/flash-attention). 
  - We compute `sigmoid(x)` as `sigmoid(x) = 0.5*(1 + tanh(0.5*x))` and leverage fast tanh primitives.
  - We remove allocation and computation of unnecessary variables (e.g., row-sum, row-max), which are not needed for sigmoid attention.

## Installation

Our FlashSigmoid implementation builds on [FlashAttention2](https://github.com/Dao-AILab/flash-attention) at commit `6c9e60de566800538fedad2ad5e6b7b55ca7f0c5` (version 2.5.6).

Subsequently, we inherit the same requirements as FlashAttention2:
  - CUDA `11.6` and above.
  - PyTorch `1.12` and above.
  - Linux operating system. 

Before installation, make sure that:
  - PyTorch is installed.
  - `packaging` package is installed. If not, run `pip install packaging`.
  - Make sure that `ninja` package is installed and that it works correctly. 
    - This can be done by checking if `ninja --version` followed by `echo $?` should return exit code `0`. 
    - Otherwise, reinstall the package as `pip uninstall -y ninja && pip install ninja`. 
    - Without `ninja`, compiling can take a very long time. 

From the `ml-sigmoid-attention` directory run the following commands to install FlashSigmoid:
```bash
# Create an environment for sigmoid attention, if not done already.
conda create -n sigmoid-attn-py310 python=3.10
conda activate sigmoid-attn-py310

# Remove pre-existing implementation, if any, and install.
cd flash_sigmoid
pip uninstall -y flash_sigmoid
rm -rf build dist flash_sigmoid.egg_info
# Note that if build fails with no apparent cause, try decreasing MAX_JOBS.
# On the other hand, you might want to try a higher value, should your setup support that, to speed-up install process.
MAX_JOBS=8 python3 setup.py install

# You can also run unit tests as follows.
# pytest -k test_flash_attn_output tests/test_flash_attn.py 
```

You can collocate softmax FlashAttention2 at the above commit as well:
```bash
# Create an environment for sigmoid attention, if not done already.
conda create -n sigmoid-attn-py310 python=3.10
conda activate sigmoid-attn-py310 

git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
git checkout 6c9e60de566800538fedad2ad5e6b7b55ca7f0c5
# Note that if build fails with no apparent cause, try decreasing MAX_JOBS.
# On the other hand, you might want to try a higher value, should your setup support that, to speed-up install process.
MAX_JOBS=8 python3 setup.py install 
cd .. && rm -rf flash-attention 
```

## Difference from Softmax FlashAttention2
```bash
# Open the github repo in browser and augment the URL with the following:
# The difference below shows <FlashAttention2> .. <FlashSigmoid>. 
https://<github-url-name>/compare/6c9e60de566800538fedad2ad5e6b7b55ca7f0c5..533c2691e05e05899eeaa546e8909f510e9cf657
```

## Example Usage

The usage and signature of flash functions of FlashSigmoid are the same as that of FlashAttention2 except:
  - We can pass an optional additional argument `sigmoid_bias: float` to the functions.
  This argument represents the `b` scalar in the defining equation of FlashSigmoid above.
  If not passed, `sigmoid_bias` gets assigned the default value of `0`. 
  - We do **NOT** support `varlen` and `kvcache` variants of flash functions. 
  - We do **NOT** support `dropout_p` and thus, `dropout_p` will always be `0`.

```python3
from flash_sigmoid import flash_attn_func as flash_sigmoid_func

# Batch size: B
# Sequence length: T
# Query heads: H_q
# Feature dimension per head: D
# Key/value heads: H_kv

# q: torch.Tensor with dtype bf16/fp16 and shape: [B, T, H_q, D]
# k: torch.Tensor with dtype bf16/fp16 and shape: [B, T, H_kv, D]
# v: torch.Tensor with dtype bf16/fp16 and shape: [B, T, H_kv, D]
# softmax_scale: Optional[float] that defaults to 1/sqrt(D) if None
# dropout_p: Attention dropout, which is NOT yet supported and is 0 for now.
# window_size: tuple[int, int] showing left and right extremes of windowed attention.
#    If we don't want windowed attention, set to (-1, -1).
# alibi_slopes: torch.Tensor with dtype fp32 and shape: [H_q] or [B, H_q].
# causal: bool to indicate whether we want to carry out causal attention.
# sigmoid_bias: float (not trainable) to be added to q @ k.T / sqrt(D).
# out: torch.Tensor with dtype and shape of q: [B, T, H_q, D]

out = flash_sigmoid_func(
    q,
    k,
    v,
    softmax_scale,
    dropout_p,
    window_size,
    alibi_slopes,
    causal,
    sigmoid_bias,
)
```

- A more detailed single file usage implementation of FlashSigmoid can be [found here](../attention_simulator/src/attention_simulator/layers/flash_sigmoid_attention.py).
- A more detailed single file usage implementation of FlashAttention2 can be [found here](../attention_simulator/src/attention_simulator/layers/flash_softmax_attention.py).

## Performance

|                                       Forward pass kernels on H100.                                       |                                     Backward pass kernels on H100.                                      |
|:---------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|
| ![Sigmoid vs. Softmax Forward Kernels](../figures/H100_noalibi_FWD_Full_17.39_0.07_Causal_18.76_0.06.png) | ![Sigmoid vs. Softmax Backward Kernels](../figures/H100_noalibi_BWD_Full_2.7_0.06_Causal_6.19_0.06.png) |


|                   Train losses comparing SigmoidAttn with SoftmaxAttn.                   |
|:----------------------------------------------------------------------------------------:|
| ![SigmoidAttn vs. SoftmaxAttn Train Losses](../figures/train_nll_softmax_vs_sigmoid.png) |

