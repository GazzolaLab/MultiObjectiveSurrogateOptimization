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