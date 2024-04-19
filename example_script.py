import os
from machinable import get

# set storage directory
get("machinable.index", os.environ["STORAGE"]).__enter__()

with get('interface.execution.frontera', resources={"-p": "normal"}):
# with get('interface.execution.status'):
# with get("interface.execution.mpi", {"resume_failed": True}, resources={"-n": 8}):
    if experiment := get(
        "interface.sopt_modeling",
        [
            f"""~from_protocol(
                "benchmarks/motoneuron_modeling/config/motoneuron.yaml"
            )""",
            {
                "dopt_params": {
                    "surrogate_method_name": "gpr",
                    "n_initial": 10,
                },
                "nodes": "10",
            },
        ],
    ).future():
        print(experiment.get_best())
