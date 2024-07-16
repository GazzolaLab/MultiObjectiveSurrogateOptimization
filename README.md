# MultiObjectiveSurrogateOptimization

## Getting started

Create a virtualenv and `pip install -r requirements.txt`.

This project uses [machinable](https://machinable.org/); check out the [interface](./interface/) module to learn about available options. 

### Using the Python API

Check out the [example_script.py](./example_script.py).

### Using the CLI

Generally, your command lines will likely look like the following:
```sh
PYTHONPATH=.:$PYTHONPATH machinable get machinable.index directory=$STORAGE <interfaces here>
```
This specifies to save and load results in the `$STORAGE` directory and it's useful to add an alias for this to your `.bashrc`:
```sh
function ma { PYTHONPATH=.:$PYTHONPATH machinable get machinable.index directory=$STORAGE "$@"; }
```
so you can type
```sh
ma <interfaces here>
```

Examples for the various benchmarks are provided below.

Note that `.<path>` is a shorthand for `interface.<path>`, e.g. typing `interface.dmosopt` is equivalent to `.dmosopt`.

## Interface

```sh
# General structure
ma <index/storage module> <execution module> <interface> --launch
```

### Execution

#### Frontera

```sh
.execution.frontera "**kwargs={'resources': {'-p': 'normal', '-t': '4:00:00'}}"
```

#### Status

```sh
.execution.status  # provides a summary of the execution status
```

#### Globus

```sh
.execution.upload  # uploads the experiment to globus
.execution.download  # download the experiment from globus
```

### Benchmarks

#### Single neuron modeling

```sh
.sopt_modeling "~from_protocol('benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_SCA.yaml')" dopt_params.n_initial=200
```

### Surrogates

#### Joint model

```python
"dopt_params": {
    "surrogate_custom_training": "models.ops.mlp",
    "surrogate_custom_training_kwargs": {
        "scope": ['objective', 'feasibility', 'sensitivity'],
        # Things to predict with this model
        #  Note that if a scope is disabled, it will fall back on the usual
        #  dmosopt options. For example, if you specify `"surrogate_method_name": 'gpr'`
        #  and scope `['feasibility']`, the MLP model will be used for the constraints
        #  but to predict the objective the usual `gpr` surrogate will be used.
        "joint": True,
        # Whether to model objective and constraints jointly. If False, the model will
        #  only learn and predict the constraints
        "feasibility_solving": False,
        # If True, the gradient information of the model will be used to push samples
        #  towards feasibility. This can be activated conditionally using a string,
        #  e.g. `'f1>0.4'` to only solve if the models F1 score is greater than 0.4
        "feasibility_max_iterations": 50,
        # Only applies if feasibility_solving is True; number of iterations
        "feasibility_use_joint_loss": True,
        # Only applies if feasibility_solving is True; whether to use joint loss
        "feasibility_max_steps_filter": True,
        # Only applies if feasibility_solving is True; optional early stopping
    },
}
```

#### Dynamic sampling

```python
"dopt_params": {
    "dynamic_initial_sampling": "models.ops.dynamic_sampling",
    "dynamic_initial_sampling_kwargs": {
        "samples_per_iteration": 25,
        # Samples to evaluate per iteration
        "max_samples": 500,
        # Number of maximum samples
        "stop_condition": "convergence_condition and feasible_ratio > 0.1",
        # Stop condition, e.g. `'f1>0.4'` to only solve if the models F1 score is greater than 0.4
        "convergence_condition": "iteration > 3 and max(recent('ecov', 3)) < 0.1",
        # Condition to determine model reliability
        "optimizer_sampling": 0.2,
        # Fraction of samples to draw using optimizer cross-over
        "feasibility_solving": True,
        # If True, the gradient information of the model will be used to push samples
        #  towards feasibility. 
        "feasibility_max_iterations": 50,
        # Only applies if feasibility_solving is True; number of iterations
        "feasibility_use_joint_loss": True,
        # Only applies if feasibility_solving is True; whether to use joint loss
        "feasibility_max_steps_filter": True,
        # Only applies if feasibility_solving is True; optional early stopping
    },
}
```


