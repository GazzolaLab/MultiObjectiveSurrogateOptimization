from interface.dmosopt import Dmosopt


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
        if "joint" in name:
            if 'mlp' in name:
                from models.mlp import MLP as Model
            elif 'resnet' in name:
                from models.resnet import Resnet as Model
            elif 'fttransformer' in name:
                from models.fttransformer import FTTransformer as Model
            else:
                from models.transformer import Transformer as Model

            return Model(
                self.num_parameters,
                self.num_constraints,
                self.num_objectives,
                xlb=self.xlb,
                xub=self.xub,
                **model_options,
            )

        from models.wrapper import Wrapper

        return Wrapper(name, self.xlb, self.xub)

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
                "surrogate_custom_training": "models.ops.joint",
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
