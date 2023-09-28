import numpy as np
from machinable import get, Component, Scope
from miv_simulator import coding
from pydantic import BaseModel
from matplotlib import pyplot as plt
from collections import defaultdict

np.set_printoptions(suppress=True)


def generate_poisson_spike_train(rate_hz, duration_ms) -> coding.SpikeTimes:
    duration_s = duration_ms / 1000.0
    ISIs = np.random.exponential(1.0 / rate_hz, int(rate_hz * duration_s * 1.5))
    spike_times = np.cumsum(ISIs)
    spike_times = spike_times[spike_times < duration_s]
    return coding.cast_spike_times(spike_times * 1000)


def gaussian_conv(spike_times, duration, tau=5, dt=0.1):
    t = np.arange(0, duration + dt, dt)
    spike_function = np.zeros_like(t)
    for spike in spike_times:
        spike_function[int(spike / dt)] = 1
    w = 3
    x = np.arange(-w * tau, w * tau + dt, dt)
    kernel = np.exp(-((x / tau) ** 2))
    return np.convolve(spike_function, kernel, mode="same")


def spike_train_distance(u_times, v_times, duration, tau=5):
    """d(u, v) in Maass+2002"""
    u_continuous = gaussian_conv(u_times, duration, tau)
    v_continuous = gaussian_conv(v_times, duration, tau)
    distance = np.linalg.norm(u_continuous - v_continuous)
    return distance / duration


class SpikeTrainPairs(Component):
    class Config:
        N: int = 200
        distance: float = 0.1
        splits: int = 10
        duration: float = 500

    def __call__(self):
        distances = [self.config.distance]

        g = lambda r: generate_poisson_spike_train(r, self.config.duration)
        d = lambda a, b: spike_train_distance(
            a, b, duration=self.config.duration
        )

        # figure out frequencies for which distances are likely
        m = {t: (0, None) for t in distances}
        for ff in range(
            1, 100, 1
        ):  # 5Hz spontanous, so around 10Hz might be reasonable
            f = ff * self.config.splits
            dd = np.mean([d(g(f), g(f)) for _ in range(50)])
            print(f, dd)
            for k, v in m.items():
                if k - v[0] > dd - v[0]:
                    m[k] = (dd, f)
        print(f"Frequency-distance map: {m}")

        # plt.hist([d(g(), g()) for _ in range(200)], bins=20)
        # plt.show()

        eps = 0.01
        data = defaultdict(list)
        for target in distances:
            f = m[target][1]
            while len(data[target]) < self.config.N:
                u, v = g(f), g(f)
                dd = d(u, v)
                if np.abs(dd - target) < eps:
                    data[target].append((u, v))

        self.save_file("data.p", data[self.config.distance])

    def data(self):
        return self.load_file("data.p", None)


data = get(SpikeTrainPairs, {"distance": 0.1}).launch().data()


# simulation

from simulation import microcircuit

# distances: list[float] = [0.1, 0.2, 0.4]
trials = {}
with get("interface.execution.local"):
    for trial in range(1):
        with get('machinable.scope', {"trial": trial}):
            trials[trial] = Component.make(
                "interface.experiment.rc",
                [
                    microcircuit.graph.files(),
                    {
                        "t_end": 500,
                        "cell_types": "from_file('simulation/config/cell_types.yml')",
                        "synapses": "from_file('simulation/config/synapses.yml')",
                        "stimulus": data[trial][0].tolist(),
                    },
                ],
            ).launch()


# x = np.arange(0, 5000, 100)
# y = x

fig = plt.figure()

for trial, experiment in trials.items():
    r = experiment.readout()
    plt.plot(r[:, 0], r[:, 1], label=f"Trial {trial}")

plt.legend()
plt.xlabel("Time [ms]")
plt.ylabel("State distance")

plt.savefig("plot.png")
