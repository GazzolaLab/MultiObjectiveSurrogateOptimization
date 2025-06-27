from machinable import Interface, get


class _E_I_WC(Interface):

    def launch(self):
        for trial in range(1):
            with get("machinable.scope", {"trial": trial}):
                for version in [
                    {},
                    {
                        "dopt_params.surrogate_method_name": "gpr",
                    },
                    {
                        "dopt_params.surrogate_method_name": "megp",
                    },
                    "~joint_model(mode='o', backbone='resnet')",
                    # "~joint_model(mode='o', backbone='fttransformer')",
                ]:
                    get("interface.sopt_wc", version).launch()

    def inspect(self):
        print(self.components[0].get_best()["y"])
