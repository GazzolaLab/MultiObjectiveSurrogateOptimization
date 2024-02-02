import numpy as np
from dmosopt.model import Model as ModelWrapper
from dmosopt.NSGA2 import NSGA2
from models.mlp import MLP


class Model(MLP):
    def __init__(self, *args, xlb, xub, **kwargs):
        super().__init__(*args, **kwargs)
        self.xlb = xlb
        self.xub = xub

    def rank(self, x):
        # return dummy; will be reset in Optimizer
        return None
        
        

def initialize(Xinit, Yinit, C, xlb, xub, file_path, options):
    x = Xinit.copy()
    y = Yinit.copy()
    yC = (C > 0).astype(int)
    
    model = Model(
        num_parameters=Xinit.shape[1],
        num_constraints=C.shape[1],
        num_objectives=Yinit.shape[1],
        xlb=xlb,
        xub=xub,
    )
    
    model.fit(
        x, yC, epochs=1000, batch_size=2048, validation_split=0.2, verbose=1
    )
    
    return ModelWrapper(feasibility=model)



class Optimizer(NSGA2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_distance_metrics = None
        
    def feasibility_transform(self, x, cutoff=50):
        if self.model.feasibility is None:
            return x
        
        xlb = self.model.feasibility.xlb
        xub = self.model.feasibility.xub
        
        x_transformed, steps = self.model.feasibility.make_feasible(
            x, learning_rate=0.1, transform=[(l,u) for l, u in zip(xlb, xub)], max_iterations=cutoff+2, verbose=0
        )
        
        if cutoff is None:
            return x_transformed
        
        # only use the samples where the steps where below cutoff
        x_transformed = np.where(np.tile(np.expand_dims(steps < cutoff, 1), reps=x.shape[1]), x_transformed, x)
        
        return x_transformed
        
        
    def generate_initial(self, bounds, local_random):
        x = super().generate_initial(bounds, local_random)
        
        x = self.feasibility_transform(x)
        
        return x
        
    def generate_strategy(self, **params):
        x_gen, state_gen = super().generate_strategy(**params)
        
        x_gen = self.feasibility_transform(x_gen)
        
        return x_gen, state_gen
    
    