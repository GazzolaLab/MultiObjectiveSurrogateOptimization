from interface.dmosopt import Dmosopt


class Sopt(Dmosopt):
    @property
    def dynamic_sampling(self) -> bool:
        return self.config.dopt_params.get("dynamic_initial_sampling", None) is not None

    @property
    def dynamic_sampling_kwargs(self) -> dict:
        if not self.dynamic_sampling:
            return {}

        return self.config.dopt_params.get("dynamic_initial_sampling_kwargs", {})

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

    @property
    def backbone(self):
        if not self.custom_training_kwargs:
            return None

        return self.custom_training_kwargs.get("backbone", "resnet")

    @property
    def trial(self):
        return self.context.predicate.get("trial", 0)

    def get_model(self, name=None, **model_options):
        if name is None:
            name = self.mO

        if name == "-":
            raise ValueError("No default model, please provide a model name.")

        if "joint" in name:
            if "mlp" in self.backbone:
                from models.mlp import MLP as Model
            elif "resnet" in self.backbone:
                from models.resnet import Resnet as Model
            elif "fttransformer" in self.backbone:
                from models.fttransformer import FTTransformer as Model
            else:
                from models.transformer import Transformer as Model

            if self.custom_training_kwargs is not None:
                model_options.setdefault(
                    "mode", self.custom_training_kwargs.get("mode", "c+o")
                )

            model_options.setdefault("xlb", self.xlb)
            model_options.setdefault("xub", self.xub)

            return Model(
                self.num_parameters,
                self.num_constraints,
                self.num_objectives,
                **model_options,
            )

        from models.wrapper import Wrapper

        return Wrapper(name, self.xlb, self.xub, **model_options)

    def weights(self, epoch=-1):
        if epoch < 0:
            epoch = self.n_epochs + epoch

        return self.local_directory(f"{epoch}.weights.h5")

    def get_sa(self, name: str = "dgsm"):
        from dmosopt import sa

        return getattr(sa, "SA_" + name.upper())(
            lo_bounds=self.xlb,
            hi_bounds=self.xub,
            param_names=self.parameter_names,
            output_names=self.objective_names,
        )

    def label(self):
        m = self.config.dopt_params.opt_id.replace("dmosopt_", "") + "::" + self.m

        if self.config.dopt_params.get("surrogate_custom_training", None) is None:
            return m

        fs = self.custom_training_kwargs.get("feasibility_solving", False)
        fs = "fs" if fs else "-"

        return m + "[" + fs + "]"

    def version_opt(self):
        return {
            "dopt_params": {
                "optimizer_name": "models.opt.Opt",
                "optimizer_kwargs": {},
                "num_generations": 1,
            }
        }

    def version_joint_model(self, **kwargs):
        if kwargs.get("mode", "c+o") not in ["c+o", "c", "o"]:
            raise ValueError("Invalid mode")
        params = {}
        if kwargs.get("sgrad", False):
            params = {"num_generations": 1}
            if kwargs.get("targets", False):
                params["optimizer_kwargs"] = dict(targets=kwargs.pop("targets"))
        return {
            "dopt_params": {
                "surrogate_custom_training": "models.ops.joint",
                "surrogate_custom_training_kwargs": kwargs,
                **params,
            }
        }

    def version_dynamic_sampling(self, **kwargs):
        return {
            "dopt_params": {
                "dynamic_initial_sampling": "models.ops.dynamic_sampling",
                "dynamic_initial_sampling_kwargs": kwargs,
            }
        }

    def version_predefined(self, source):
        return {
            "dopt_params": {
                "dynamic_initial_sampling": "models.ops.predefined_sampling",
                "dynamic_initial_sampling_kwargs": {"source": source},
                "n_epochs": 0,
                "n_initial": 0,
            }
        }
