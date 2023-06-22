# MultiObjectiveSurrogateOptimization

## Motoneuron model

```python
from machinable import get

with get("interface.execution.frontera", {"nodes": 32, "ranks": 56}):
    experiment = get("motoneurons", {
        "num_epochs": 2, "num_initial": 2000, "population_size": 1000
    }).launch()

if not experiment.execution.is_finished():
    experiment.execution.stream_output()
else:
    experiment.plot()
```
