# Optorch

A preliminary version of `optorch` -- a purely functional pytorch optimizer framework.

## Installation

From the `ml-sigmoid-attention` directory run the following commands:

```bash
# Create an environment for sigmoid attention, if not done already.
conda create -n sigmoid-attn-py310 python=3.10
conda activate sigmoid-attn-py310

cd optorch
pip install .
```

## Example

Adding a new optimizer with Optorch is very easy. For example, Stochastic gradient descent (SGD) is fully summarized by: 

``` python3
@torch.no_grad()
def sgd(
    params: PARAMS_GRADS_TYPE,
    grads: PARAMS_GRADS_TYPE,
    momentum: PARAMS_GRADS_TYPE,
    config: SGDConfig,
) -> t.Tuple[PARAMS_GRADS_TYPE, PARAMS_GRADS_TYPE]:
    """Functional implementation of the SGD optimizer.

    :param params: List of model parameters.
    :param grads: List of gradients of the model parameters.
    :param momentum: List of momentum values for each parameter.
    :param config: Configuration for the SGD optimizer.
    :return: Updated model parameters, momentum values, and state steps.

    """

    def _sgd(param, grad, momentum):
        """Inplace SGD update rule."""
        if config.weight_decay > 0:
            update = grad.add(param, alpha=config.weight_decay)

        if config.momentum > 0:
            momentum.mul_(config.momentum).add_(grad)
            update = momentum

        # Update for the current scheduled LR.
        current_lr = config.max_lr
        if config.lr_schedule_value is not None:
            current_lr *= config.lr_schedule_value

        param.sub_(current_lr * update)

    tree.map_structure(_sgd, params, grads, momentum)
    return params, momentum
```

## Extra Details About Optorch

A functional optimizer consists of three things:

  1. **Config dict**: parameterizes LR, WD, etc -- see `DEFAULT_OPTIMIZER_CONFIGS` in [optorch/builder.py](./src/optorch/builder.py).
  2. **State dict**: contains stateful objects needed for optimization (eg: momentum) -- see `DEFAULT_OPTIMIZER_STATES` in [optorch/builder.py](./src/optorch/builder.py).
  3. **Stepper function**: a function that takes in `{params, grads, state dict, config}` and does one step of optimization -- see `sgd_step` in [optorch/sgd.py](./src/optorch/sgd.py) for example.

The [implemented Adam optimizer](./src/optorch/adam.py) for example supports Adam, AdamW. See `src/optorch` for further details.

## Tests

- The optimizers (barring LARS and fully decoupled Adam) are tested against their pytorch counterparts.
- To run the tests, run the following commands while sitting inside the `optorch` directory. 

``` bash
> pytest src/tests/test_sgd.py
> pytest src/tests/test_adam.py
> pytest src/tests/test_lars.py  # doesn't test equivalence just yet.
```

**NB:** the functional [automatic mixed precision scaler](./src/optorch/amp.py) is early beta. Preliminary tests indicate that it works and matches the baseline full precision runs, but more testing is needed. All experiments in the manuscript with attention simulator are evaluated using full precision. Large scale experiments are handled in a separate codebase.
