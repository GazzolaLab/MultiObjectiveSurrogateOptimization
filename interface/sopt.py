from interface.dmosopt import Dmosopt
import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error


class Sopt(Dmosopt):

    @property
    def custom_training(self) -> bool:
        return (
            self.config.dopt_params.get("surrogate_custom_training", None) is not None
        )

    @property
    def custom_training_kwargs(self) -> dict:
        if not self.custom_training:
            return {}

        return self.config.dopt_params.get("surrogate_custom_training_kwargs", {})

    @property
    def custom_surrogate_name(self) -> str:
        return "joint-" + self.custom_training_kwargs.get("mode", "c+o")

    @property
    def surrogate_method_name(self) -> str:
        if self.custom_training and self.custom_training_kwargs.get("objectives", True):
            return self.custom_surrogate_name

        return super().surrogate_method_name

    @property
    def mO(self) -> str:
        return self.surrogate_method_name or "-"

    @property
    def mC(self) -> str:
        if self.custom_training_kwargs.get("constraints", False):
            return self.custom_surrogate_name

        if self.config.dopt_params.get("feasiblity", None) is True:
            return "logR"

        return "-"

    @property
    def mS(self):
        if s := self.custom_training_kwargs.get("sensitivity", False):
            return self.custom_surrogate_name + (f"-{s}" if isinstance(s, str) else "")

        if (
            name := self.config.dopt_params.get("sensitivity_method_name", None)
        ) is not None:
            return name

        return "-"

    @property
    def m(self):
        return f"O:{self.mO}/C:{self.mC}/S:{self.mS}"

    def get_model(self, name, **model_options):
        if name == "mlp" or "joint" in name:
            from models.mlp import MLP

            return MLP(
                self.num_parameters,
                self.num_constraints,
                self.num_objectives,
                # joint=self.config.dopt_params.get(
                #     "surrogate_custom_training_kwargs", {}
                # ).get("joint", True),
                # xlb=self.xlb,
                # xub=self.xub,
                **model_options,
            )

        class _Wrapper:
            def __init__(self, name, xlb, xub) -> None:
                self.xlb = np.array(xlb)
                self.xub = np.array(xub)
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
                    **model_options,
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

        return _Wrapper(name, self.xlb, self.xub)

    def label(self):
        m = self.config.dopt_params.opt_id.replace("dmosopt_", "") + "::" + self.m

        if self.config.dopt_params.get("surrogate_custom_training", None) is None:
            return m

        fs = self.custom_training_kwargs.get("feasibility_solving", False)
        fs = "fs" if fs else "-"

        return m + "[" + fs + "]"

    def version_joint_model(self, **kwargs):
        if kwargs.get("mode", "c+o") not in ["c+o", "c", "o"]:
            raise ValueError("Invalid mode")
        return {
            "dopt_params": {
                "surrogate_custom_training": "models.ops.mlp",
                "surrogate_custom_training_kwargs": kwargs,
            }
        }

    def version_dynamic_sampling(self, **kwargs):
        return {
            "dopt_params": {
                "dynamic_initial_sampling": "models.ops.dynamic_sampling",
                "dynamic_initial_sampling_kwargs": kwargs,
            }
        }
