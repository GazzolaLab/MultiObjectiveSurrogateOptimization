from machinable import Interface, get


class _E_I_WC(Interface):

    def launch(self):
        get(
            "interface.sopt_wc",
        ).launch()

    def inspect(self):
        print(self.components[0].get_best()["y"])
