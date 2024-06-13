from interface.dmosopt import Dmosopt
import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error


class Sopt(Dmosopt):

    @property
    def scope(self) -> list[str]:
        if self.config.dopt_params.get("surrogate_custom_training", None) is None:
            return []

        if (
            k := self.config.dopt_params.get("surrogate_custom_training_kwargs", None)
        ) is not None:
            return k.get("scope", [])

        return []

    @property
    def custom_surrogate_name(self) -> str:
        return "joint-" + (
            "c+o"
            if self.config.dopt_params.get("surrogate_custom_training_kwargs", {}).get(
                "joint", True
            )
            else "c"
        )

    @property
    def surrogate_method_name(self) -> str:
        if "objective" in self.scope:
            return self.custom_surrogate_name

        return super().surrogate_method_name

    @property
    def mO(self) -> str:
        return self.surrogate_method_name or "-"

    @property
    def mC(self) -> str:
        if "feasiblity" in self.scope:
            return self.custom_surrogate_name

        if self.config.dopt_params.get("feasiblity", None) is True:
            return "logR"

        return "-"

    @property
    def mS(self):
        if "sensitivity" in self.scope:
            return self.custom_surrogate_name

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
                joint=self.config.dopt_params.get(
                    "surrogate_custom_training_kwargs", {}
                ).get("joint", True),
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
                outlier = np.any(np.abs(zscores) > 3, axis=1)

                return x[~outlier], y[~outlier], yC[~outlier]

            def autofit(self, x, y, yC, *args, **kwargs):
                x, y, yC = self.preprocess(x, y, yC)

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
                    "epochs": 1.0,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
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

        return _Wrapper(name, self.xlb, self.xub)

    def label(self):
        return self.m

    def version_joint_model(
        self,
        scope=None,
        joint=True,
        feasibility_solving=False,
        feasibility_max_iterations=50,
        feasibility_use_joint_loss=True,
        feasibility_max_steps_filter=True,
    ):
        if scope is None:
            scope = [
                "objective",
                "feasiblity",
            ]
        return {
            "dopt_params": {
                "surrogate_custom_training": "models.ops.mlp",
                "surrogate_custom_training_kwargs": {
                    "scope": scope,
                    "joint": joint,
                    "feasibility_solving": feasibility_solving,
                    "feasibility_max_iterations": feasibility_max_iterations,
                    "feasibility_use_joint_loss": feasibility_use_joint_loss,
                    "feasibility_max_steps_filter": feasibility_max_steps_filter,
                },
            }
        }

    def version_dynamic_sampling(
        self,
        max_iterations=10,
        stop_condition="f1>0.4",
        feasibility_solving=False,
        feasibility_max_iterations=50,
        feasibility_use_joint_loss=True,
        feasibility_max_steps_filter=True,
    ):
        return {
            "dopt_params": {
                "dynamic_initial_sampling": "models.ops.dynamic_sampling",
                "dynamic_initial_sampling_kwargs": {
                    "max_iterations": max_iterations,
                    "stop_condition": stop_condition,
                    "feasibility_solving": feasibility_solving,
                    "feasibility_max_iterations": feasibility_max_iterations,
                    "feasibility_use_joint_loss": feasibility_use_joint_loss,
                    "feasibility_max_steps_filter": feasibility_max_steps_filter,
                },
            }
        }
