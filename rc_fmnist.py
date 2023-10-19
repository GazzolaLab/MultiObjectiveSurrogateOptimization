import numpy as np
from machinable import get, Component, Scope
from miv_simulator import coding
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

np.set_printoptions(suppress=True)

from fashion_mnist.utils import mnist_reader

X_train, y_train = mnist_reader.load_mnist("fashion_mnist/data/fashion", kind="train")
X_test, y_test = mnist_reader.load_mnist("fashion_mnist/data/fashion", kind="t10k")


def binary_problem(X_train, y_train, classes=(0, 1)):
    combined_mask = (y_train == classes[0]) | (y_train == classes[1])

    return X_train[combined_mask], y_train[combined_mask]


def encode(image, duration=100, max_rate=200.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    image = image.reshape([28, 28])
    h, w = image.shape
    image = zoom(image, np.sqrt(duration) / h)

    image = image / 255.0 * max_rate  # normalize

    image = image.ravel()

    return coding.binary_sparse_spike_train_2_spike_times(
        rng.random(len(image)) < image * 0.001, temporal_resolution=1.0
    )


def encode_batch(images, duration=100, max_rate=200.0, seed=None):
    rng = np.random.default_rng(seed)
    sequence = []
    for i, image in enumerate(images):
        sequence.append(i * duration + encode(image, duration, max_rate, rng))
    return np.concatenate(sequence)


binary = False
if binary:
    X_train01, y_train01 = binary_problem(X_train, y_train)
else:
    X_train01, y_train01 = X_train, y_train


from MultiObjectiveSurrogateOptimization.simulation import culture

fig = plt.figure()


# with get("interface.execution.local"):
with get(
    "interface.execution.frontera",
    {"partition": "normal"},
    resources={"-t": "2:00:00"},
):
    N = 25  # 100   or 25
    max_rate = 500
    total = finished = 0
    for duration in [100.0]:
        stimulus = encode_batch(
            X_train01[:N], duration=duration, max_rate=max_rate, seed=42
        )
        for trial in range(1):
            with Scope(
                {
                    "trial": trial,
                    "duration": duration,
                    "N": N,
                    "max_rate": max_rate,
                }
            ):
                e = get(
                    "interface.experiment.rc",
                    [
                        culture.graph.files(),
                        {
                            "t_end": duration * N,
                            "cell_types": "from_file('simulation/config/cell_types.yml')",
                            "synapses": "from_file('simulation/config/synapses.yml')",
                            "stimulus": stimulus.tolist(),
                            "nodes_": 32,
                            "ranks_": 16,
                        },
                    ],
                ).launch()

                total += 1
                if e.cached():
                    finished += 1
                readout = e.readout()
                print(e.local_directory())
                if readout is None:
                    continue

                # group
                readout = readout[: int(N * duration)].reshape([N, -1])

                # decode using root mean square
                readout_rms = np.sqrt(np.mean(readout**2, axis=1))

                # tune on first 80%, test on 20%
                training = int(0.8 * N)

                def to_cat(x):
                    return np.rint(np.clip(x, -0.01, 9.01))

                baseline = LinearRegression()
                baseline.fit(X_train01[:training], y_train01[:training])
                y_pred_baseline = baseline.predict(X_train01[training + 1 : N])
                acc_baseline = accuracy_score(
                    y_true=y_train01[training + 1 : N], y_pred=to_cat(y_pred_baseline)
                )
                print(f"Baseline accuracy {acc_baseline*100}%")

                lr = LinearRegression()
                lr.fit(readout[:training], y_train01[:training])
                y_pred01 = lr.predict(readout[training + 1 :])
                y_pred01[y_pred01 < 0.5] = 0
                y_pred01[y_pred01 >= 0.5] = 1
                acc = accuracy_score(
                    y_true=y_train01[training + 1 : N], y_pred=to_cat(y_pred01)
                )
                print(
                    f"Accuracy on {N - training} samples: {acc*100}% (trained on {training})"
                )

    print(f"Found {finished}/{total} cached experiments")
    # if finished != total:
    #     continue

    # plt.plot(x, y, label=f"d(u,v)={distance} (mean={round(reduced, 4)})")
    # # plt.fill_between(x, y - error, y + error)

    # plt.legend()
    # plt.xlabel("Time [ms]")
    # plt.ylabel("State distance")
    # plt.savefig("plot.png")
