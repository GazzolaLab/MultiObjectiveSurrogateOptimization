import os
from machinable import get
from utils.protocol import from_yaml

with get(
    "machinable.index",
    os.path.expandvars(os.path.join("$SCRATCH", "MultiObjectiveSurrogateOptimization")),
):
    with get("interface.execution.frontera", {"nodes": 32, "ranks": 56}):
        experiment = get(
            "interface.motoneurons",
            {
                "protocol": from_yaml("motoneuron"),
                "num_epochs": 2,
                "num_initial": 2000,
                "population_size": 1000,
            },
        ).launch()

    if not experiment.execution.is_finished():
        experiment.execution.stream_output()
    else:
        experiment.plot()
