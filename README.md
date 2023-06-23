# MultiObjectiveSurrogateOptimization

## Motoneuron model

```python
from machinable import get
from utils.protocol import from_yaml

with get("interface.execution.frontera", {"nodes": 2, "ranks": 2}):
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
```

