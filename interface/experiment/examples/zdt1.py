from machinable import Component
import numpy as np
from pydantic import Field, BaseModel


class ZDT1(Component):
    class Config(BaseModel):
        p: dict = Field("???")

    def __call__(self) -> None:
        """This is the Zitzler-Deb-Thiele Function - type A
        Bound: XUB = [1,1,...]; XLB = [0,0,...]
        dim = 30
        """
        x = np.asarray([self.config.p[k] for k in sorted(self.config.p)])
        num_variables = len(x)
        f = np.zeros(2)
        f[0] = x[0]
        g = 1.0 + 9.0 / float(num_variables - 1) * np.sum(x[1:])
        h = 1.0 - np.sqrt(f[0] / g)
        f[1] = g * h
        return f

    @classmethod
    def objective(cls, parameters):
        return cls({"p": {k: float(v) for k, v in parameters.items()}})()

    @staticmethod
    def pareto(n_points=100):
        f = np.zeros([n_points, 2])
        f[:, 0] = np.linspace(0, 1, n_points)
        f[:, 1] = 1.0 - np.sqrt(f[:, 0])
        return f
