# Attention simulator

Categorize and quantify all parts of attention in a transformer block.

  - Track step-wise parameter stats.
  - Track step-wise gradient stats.
  - Track step-wise attention matrix stats.
  - Track step-wise attention matrix @ value stats.
  - Track step-wise residual stats.
  - Track step-wise MLP output stats.
  - Track step-wise optimizer state stats.
  - Whatever else you desire.


`attention_simulator` helps drive research from a "track everything" perspective.

It uses a fully functional approach to model inference, gradient tracking, and optimization.

## Installation

While being in the directory `ml-sigmoid-attention`, run the following commands:

```bash
# Create an environment for sigmoid attention, if not done already.
conda create -n sigmoid-attn-py310 python=3.10
conda activate sigmoid-attn-py310

# Follow directions to install optorch.
# (Optional) Follow directions to install flash_sigmoid. 

cd attention_simulator
pip install .

```

## Example

```bash
# Test on macOS. 
> python examples/language_modeling/train_autoregressive_language_model.py train_datasets.0.num_workers=0

# Test on Linux. 
> python examples/language_modeling/train_autoregressive_language_model.py
```

This will create a model:

```bash
CausalBlockContainer(
  (input_embedding): Embedding(32100, 384)
  (output_embedding): PreNormLinear(
    (norm): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
    (linear): Linear(in_features=384, out_features=32100, bias=False)
  )
  (pe_layer): Identity()
  (blocks): ModuleList(
    (0): TransformerBlock(
      (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (attn): Attention(                                                                                                                                                                                                                                attn_temp=tensor([0.]),
        attn_bias=tensor([0.]),
        num_heads=12,
        head_dim=32,
        masking_fn=<function softmax_masking_fn.<locals>.masking_fn at 0x7911a5148820>,
        attn_activ_fn=functools.partial(<function softmax_attention at 0x7911a51b70a0>),
        qk_transform=<function apply_rotary_emb at 0x7911a51b56c0>,
        attn_logits_transform=None,
        (qkv): Linear(in_features=384, out_features=1152, bias=False)
        (q_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (k_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (proj): Linear(in_features=384, out_features=384, bias=False)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (ls1): Identity()
      (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=384, out_features=1536, bias=False)
        (norm1): Identity()
        (act): GELU(approximate='none')
        (drop1): Dropout(p=0.0, inplace=False)
        (fc2): Linear(in_features=1536, out_features=384, bias=False)
        (drop2): Dropout(p=0.0, inplace=False)
      )
      (ls2): Identity()
    )
  )
)
```

This run will result in a `pd.DataFrame`, figures, and logs saved to `outputs/`. The dataframe contains step level values for everything. An example is below:

```bash
   output_mean_pnorm_attn_matrix  output_mean_pnorm_attn_times_v  output_mean_pnorm_attn_output  ...  optimizer_state_min_pnorm_exp_avg_sq_ls2.gamma      loss  step
0                       6.080858                       27.965849                      16.199158  ...                                    0.000000e+00  4.257858     1
1                       6.088493                       28.282953                      16.425102  ...                                    6.599810e-08  4.797568     2
2                       6.074718                       28.227150                      16.403595  ...                                    4.476077e-07  4.331578     3
3                       6.089534                       28.531183                      16.720461  ...                                    2.031736e-06  4.323077     4
4                       6.083996                       27.993509                      16.205957  ...                                    5.433111e-06  4.613260     5
5                       6.089570                       28.326208                      16.584709  ...                                    6.383375e-06  4.767944     6
6                       6.096973                       28.635052                      16.687046  ...                                    8.674428e-06  4.549541     7
7                       6.103353                       28.690317                      16.879353  ...                                    9.642086e-06  4.761167     8
8                       6.095811                       28.583351                      16.706120  ...                                    1.698104e-05  4.643979     9
9                       6.122532                       28.401485                      16.773514  ...                                    2.700629e-05  4.605836    10
```

## Changing Hyper-Parameters

- The hyper-parameters are defined using [Hydra](https://hydra.cc/) configuration files in YAML format.
- The default configuration file is `examples/language_modeling/experiment_tiny_shakespeare_rope.yaml`.
- You can easily override any of the hyper-parameters defined in the YAML file from the command line. For example, to use `ALiBiAttention` instead of `RopeAttention`:

```bash
python examples/language_modeling/train_autoregressive_language_model.py model.block_fn.attn_fn._target_=attention_simulator.layers.attention.ALiBiAttention
# Add train_datasets.0.num_workers=0 if on macOS.
```

This will override the `model.block_fn.attn_fn._target_` setting to use ALiBi.

Some key hyper-parameters defined in the YAML:

  - `seed`: Random seed.
  - `training.max_steps`: Maximum number of training steps.
  - `train_datasets`: List of training datasets and their configurations.
  - `test_datasets`: List of evaluation datasets and their configurations.
  - `tokenizer`: Tokenizer configuration.
  - `model`: Transformer model architecture configuration.
  - `model.dim`: Embedding dimension.
  - `model.num_blocks`: Number of transformer blocks. You can also [parameterize each block individually](examples/language_modeling/configs/experiment_realnews_transformer_bf16_flash_sigmoid_multilayer_rope.yaml).
  - `model.block_fn.attn_fn`: Attention module configuration.
  - `model.block_fn.mlp_fn`: MLP module configuration
  - `optimizer`: Optimizer configuration.
  - `lr_scheduler`: Learning rate scheduler configuration (in [0, 1]).

Refer to the YAML files for the full set of configurable hyper-parameters.

## FlashSigmoid vs. FlashAttenion2 Configurations

You can test both forms of attention in one installation. See the slightly more complicated configs at:

 - **FlashSigmoid**: [experiment_realnews_transformer_bf16_flash_sigmoid_multilayer_rope.yaml](examples/language_modeling/configs/experiment_realnews_transformer_bf16_flash_sigmoid_multilayer_rope.yaml)
 - **FlashSoftmax**: [experiment_realnews_transformer_bf16_flash_softmax_multilayer_rope.yaml](examples/language_modeling/configs/experiment_realnews_transformer_bf16_flash_softmax_multilayer_rope.yaml)
