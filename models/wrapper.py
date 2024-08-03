import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error

class Wrapper:
    def __init__(self, name, xlb, xub, **model_options) -> None:
        self.xlb = np.array(xlb)
        self.xub = np.array(xub)
        self.model_options = model_options
        self.model = None
        if name == "gpr":
            from dmosopt.model import GPR_Matern

            self.model_cls = GPR_Matern
        elif name == "megp":
            from dmosopt.model import MEGP_Matern

            self.model_cls = MEGP_Matern

    def preprocess(self, x, y, yC):
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)
        yC = np.nan_to_num(yC)

        # remove outliers
        ylog = np.log(y + 1)
        ylmean = np.mean(ylog, axis=0)
        ylstd = np.std(ylog, axis=0)
        zscores = (ylog - ylmean) / ylstd
        outlier = np.any(np.abs(zscores) > 2, axis=1)

        if yC is None:
            return x[~outlier], y[~outlier], yC
    
        return x[~outlier], y[~outlier], yC[~outlier]

    def autofit(self, x, y, yC, *args, **kwargs):
        x, y, yC = self.preprocess(x, y, yC)

        if yC is not None:
            feasible = np.argwhere(np.all(yC > 0.0, axis=1))
            if len(feasible) > 0:
                feasible = feasible.ravel()
                x = x[feasible, :]
                y = y[feasible, :]
                yC = yC[feasible, :]
        
        from dmosopt import MOEA

        x, y = MOEA.remove_duplicates(x, y)

        self.model = self.model_cls(
            xin=x,
            yin=y,
            nInput=x.shape[1],
            nOutput=y.shape[1],
            xlb=self.xlb,
            xub=self.xub,
            **self.model_options,
        )

    def autoeval(
        self,
        x,
        y,
        yC,
        verbose=2,
    ):
        x, y, yC = self.preprocess(x, y, yC)

        y_pred = self.model.evaluate(x)

        return {
            "mdae": float(
                median_absolute_error(
                    y,
                    y_pred,
                )
            ),
            "mae": float(
                mean_absolute_error(
                    y,
                    y_pred,
                )
            ),
        }
        
    def predict(self, x):
        return self.model.evaluate(x)