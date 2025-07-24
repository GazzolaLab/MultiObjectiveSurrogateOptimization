from machinable import Interface, get, Index
import os


class _Ca1Inference(Interface):
    class Config:
        index: str = ""

    def populations(self):
        return [
            "SCA",
            "IVY",
            "PVBC",
            "CCKBC",
            "AAC",
            "BS",
            "OLM",
            "NGFC",
            "IS",
        ]

    def samplers(self):
        return ["slh", "lh", "mc", "sobol"]

    def launch(self):
        sampler = "mc"
        trial = 0

        index = self.config.index
        if index == "":
            index = Index.get().config.directory

        for nm in self.populations():
            for model in [
                "resnet-o",
                "resnet-c+o",
                "fttransformer-o",
                "fttransformer-c+o",
                "megp",
            ]:
                with get("machinable.index", {"directory": index}):
                    with get("machinable.scope", {"trial": trial}):
                        experiment = get(
                            "interface.sopt_modeling",
                            [
                                f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                                {
                                    "dopt_params.surrogate_method_name": "gpr",
                                    "dopt_params.initial_method": sampler,
                                },
                            ],
                        )

                        version = {
                            "run": experiment.uuid,
                            "index": os.path.abspath(index),
                            "model": model,
                        }

                get(
                    "interface.sopt_inference",
                    version,
                ).launch().save_attribute("model_name", f"{nm}-{model}")

    def models(self):
        models = []
        for c in self.components:
            models.append(c.load_attribute("model_name"))
        return models

    def export(self):
        for c in self.components:
            print(
                "cp "
                + c.local_directory("results.p")
                + " ./export/"
                + c.load_attribute("model_name")
                + ".p"
            )
