import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from models.utils import preprocess


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
        return preprocess(x, y, yC, nan="remove")

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

        try:
            mean_yR = []
            std_yR = []
            for m in self.model.smlist:
                mean_yR.append(m._y_train_mean)
                std_yR.append(m._y_train_std)
            mean_yR = np.array(mean_yR)
            std_yR = np.array(std_yR)

            def normed(metric):
                def _w(y_true, y_pred, *args, **kwargs):
                    return metric(
                        (y_true - mean_yR) / std_yR,
                        (y_pred - mean_yR) / std_yR,
                        *args,
                        **kwargs,
                    )

                return _w

        except:

            def normed(metric):
                return metric

        y_pred = self.model.evaluate(x)

        return {
            "mdae": normed(median_absolute_error)(
                y, y_pred, multioutput="raw_values"
            ).tolist(),
            "r2": normed(r2_score)(y, y_pred, multioutput="raw_values").tolist(),
            "mae": normed(mean_absolute_error)(
                y, y_pred, multioutput="raw_values"
            ).tolist(),
        }

    def predict(self, x):
        return self.model.evaluate(x)
