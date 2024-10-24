from sklearn.linear_model import LinearRegression
import numpy as np

class LR:
    def __init__(
        self,
        xin,
        yin,
        nInput,
        nOutput,
        xlb,
        xub,
        seed=None,
        logger=None,
    ):
        self.xin = xin
        self.yin = yin
        self.nInput = nInput
        self.nOutput = nOutput
        self.xlb = xlb
        self.xub = xub
        self.seed = seed
        self.logger = logger
        
        # filter NaNs
        valid_indices = ~np.isnan(yin).any(axis=1)
        self.xin = xin[valid_indices]
        self.yin = yin[valid_indices]

        self.model = LinearRegression()
        self.model.fit(self.xin, self.yin)

    def predict(self, xin):
        y = self.model.predict(xin)
        y_var = np.var(y, axis=0)
        return y, y_var

    def evaluate(self, x):
        mean, var = self.predict(x)
        return mean
    