# MultiObjectiveSurrogateOptimization

## Getting started

Install [uv](https://docs.astral.sh/uv) and `uv sync`.

To use the surrogate models, you also need to install:

```
tensorflow==2.16.2
tensorflow-probability==0.24
```
and
```
torch
gpytorch
```

This project uses [machinable](https://machinable.org/); check out the [interface](./interface/) module to learn about available [CLI](https://machinable.org/guide/cli.html) options. 

## Experiments and results

All experiments and plots are in the notebooks at the top-level of this repository.

## Data availability

Data is available on [globus.org](https://www.globus.org/) and can be retrieved automatically using the [machinable globus interface](https://machinable.org/examples/globus-storage/).

### Authenticating the Globus SDK client

The globus client must be authenticated once using the following steps:

#### 1. Make sure the globus SDK is available

`pip install globus_sdk`

#### 2. Create the client configuration 

Create `~/.globus-config.json`

```json
{
    "client_id": "af2d8f73-c999-43d9-8cae-ec841055f6ca",   # machinable client
    "local_endpoint_id": "<your-globus-client-id-here>",   # UPDATE with your globus client ID that is running on your machine
    "local_endpoint_directory": "$STORAGE",
    "remote_endpoint_id": "<remote storage globus id>",
    "remote_endpoint_directory": "<remote storage location>"
}
```

#### 3. Initiate the authentication

```
$ python -c "from machinable import get; get('interface.storage.globus').authorizer"
```

This will generate a URL to obtain a client token that you need to copy back into the terminal.

