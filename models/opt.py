from dmosopt.MOEA import MOEA, remove_worst
from typing import Optional, Any, Dict, Tuple
import numpy as np


class Opt(MOEA):
    def __init__(
        self,
        popsize: int,
        nInput: int,
        nOutput: int,
        model: Optional[Any],
        **kwargs,
    ):
        super().__init__(
            name="ModelOpt",
            popsize=popsize,
            nInput=nInput,
            nOutput=nOutput,
            **kwargs,
        )
        self.model = model
        self.objectives = None
        self.parameters = None
        self.x = None
        self.y = None

    def get_population_strategy(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.parameters.copy(), self.objectives.copy()

    def initialize_state(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bounds: np.ndarray,
        local_random: Optional[np.random.Generator] = None,
        **params,
    ):
        self.x = x
        self.y = y
        self.parameters, self.objectives, _ = remove_worst(x, y, self.popsize)

    def generate_strategy(self, **params):
        split = len(self.parameters) // 2
        positive, _ = self.model.objective.make_feasible(
            self.parameters[:split, :],
            targets="objective",
            verbose=1,
        )

        negative, _ = self.model.objective.make_feasible(
            self.parameters[split:, :],
            targets="objective inverse",
            verbose=1,
        )

        samples = np.concatenate([positive, negative], axis=0)

        assert len(samples) == len(self.parameters)

        return samples, None

    def update_strategy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        state: Dict[Any, Any],
        **params,
    ):
        self.parameters, self.objectives, _ = remove_worst(x, y, self.popsize)
