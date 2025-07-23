from machinable import Component, get
from pydantic import BaseModel, Field
from models.wrapper import Wrapper
from models.model import Model
import numpy as np
import time


class SoptInference(Component):

    class Config(BaseModel):
        run: str = Field("???")
        index: str = "./index"
        model: str = "fttransformer-c+o"

    def __call__(self):
        results = []

        with get("machinable.index", {"directory": self.config.index}):
            run = get.by_id(self.config.run)

            x, y, yC = run.load_h5_arrays()

            epoch_ranges = run.epoch_ranges()

            for e, (train, test) in enumerate(epoch_ranges[1:]):
                x_train = x[:train]
                y_train = y[:train]
                yC_train = yC[:train]

                x_test = x[train:test]
                y_test = y[train:test]
                yC_test = yC[train:test]

                epoch = {
                    "epoch": e + 1,
                    "start": train,
                    "end": test,
                    "x_train": x_train,
                    "y_train": y_train,
                    "yC_train": yC_train,
                    "x_test": x_test,
                    "y_test": y_test,
                    "yC_test": yC_test,
                    "t_autoepoch": 0.0,
                }

                epochs = None
                if "-" in self.config.model:
                    backbone, mode = self.config.model.split("-")
                    model = Model.unserialize(
                        dict(
                            cls=backbone,
                            mode=mode,
                            xlb=run.xlb,
                            xub=run.xub,
                            num_parameters=run.num_parameters,
                            num_constraints=run.num_constraints,
                            num_objectives=run.num_objectives,
                        )
                    )
                    t = time.time()
                    m = model.autoepoch(x_train, y_train, yC_train)
                    epoch["t_autoepoch"] = time.time() - t
                    epochs = np.mean(m)
                else:
                    model = Wrapper(self.config.model, run.xlb, run.xub)

                t = time.time()
                model.autofit(x_train, y_train, yC_train, epochs=epochs)
                epoch["t_autofit"] = time.time() - t

                t = time.time()
                epoch["y_pred_train"] = model.predict(x_train)
                epoch["t_y_pred_train"] = time.time() - t

                t = time.time()
                epoch["y_pred_test"] = model.predict(x_test)
                epoch["t_y_pred_test"] = time.time() - t

                results.append(epoch)

        self.save_file("results.p", results)
