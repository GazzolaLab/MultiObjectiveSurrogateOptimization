from machinable import Interface, get, Execution


class Baseline(Interface):

    def launch(self):
        for trial in range(1):
            with get("machinable.scope", {"trial": trial, 'cohort': 'baseline1'}):
                for nm in [
                    "SCA",
                    "IVY",
                    "PVBC",
                    "CCKBC",
                    "AAC",
                    "BS",
                    "OLM", 
                    "NGFC",
                    "IS"
                ]:
                    protocol = [
                        "interface.sopt_modeling",
                        f"""~from_protocol("benchmarks/ca1_pinsky_rinzel_modeling/config/CA1_{nm}.yaml")""",
                    ]

                    # no surrogate
                    get(protocol, {
                        'nodes': "128",
                    }).launch()

    def progress(self):
        for c in self.components:
            print(c.label())
            done = len(c.load_h5()['epochs'])
            print(done, ' / ', c.num_evals_total, ' (', round(done / c.num_evals_total * 100), '%)')
            print('')