import os
from machinable import Interface, get
import shutil


class _Ca1SoptBaseline(Interface):
    class Config:
        selection: tuple = (-4, None)
        ranges: bool = False

    def populations(self):
        return [
            "IVY",
            "PVBC",
            "CCKBC",
            "AAC",
            "IS",
            "SCA",
            "BS",
            "OLM",
            "NGFC",
        ][slice(*self.config.selection)]

    def launch(self):
        wr = "_wide_range" if self.config.ranges else ""

        for nm in self.populations():
            for trial in range(3):
                with get("machinable.scope", {"trial": trial}):
                    initial = get(
                        "interface.sopt_modeling",
                        [
                            f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config{wr}/CA1_{nm}.yaml")""",
                            {
                                "dopt_params.surrogate_method_name": "gpr",
                                "dopt_params.n_epochs": 0,
                            },
                        ],
                    ).launch()
                    initial.save_attribute("preflight", True)
                    if initial.cached():
                        for version in [
                            {},
                            {
                                "dopt_params.surrogate_method_name": "gpr",
                            },
                            {
                                "dopt_params.surrogate_method_name": "megp",
                            },
                            "~joint_model(mode='o', backbone='resnet')",
                            "~joint_model(mode='c+o', backbone='resnet')",
                            "~joint_model(mode='o', backbone='fttransformer')",
                            "~joint_model(mode='c+o', backbone='fttransformer')",
                        ]:
                            with get("machinable.scope", {"parent": initial.hash}):
                                e = get(
                                    [
                                        "interface.sopt_modeling",
                                        f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config{wr}/CA1_{nm}.yaml")""",
                                    ],
                                    [version]
                                    + [
                                        {
                                            "dopt_params.n_epochs": 25,
                                        }
                                    ],
                                ).launch()

                                if os.environ.get("LAUNCH", 0) and not os.path.isfile(
                                    e.output_filepath
                                ):
                                    e.commit()
                                    shutil.copyfile(
                                        initial.output_filepath, e.output_filepath
                                    )
