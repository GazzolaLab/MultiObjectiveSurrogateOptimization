import json
import os
import networkx as nx
import numpy as np
from functools import partial
import diffrax
import jax.numpy as jnp
from benchmarks.cortical_culture.data_preprocessing import compute_band_power


def obj_fun(pp, env, targets, t_end):
    env.set_params(pp)

    dt = 0.005  # twice as high as highest frequency (200 Hz)
    t, y = env.run(t_end, dt)

    data = y[::2, :].T

    scale = 327.29  # scale to match experimental units
    offset = 0

    q = compute_band_power(data * scale + offset, fs=1 / dt)

    means = q.mean().to_dict()
    stds = q.std().to_dict()

    objectives = []
    for k, target in targets.items():
        if k.endswith("_mean"):
            observed = means[k.replace("_mean", "")]
        elif k.endswith("_std"):
            observed = stds[k.replace("_std", "")]

        objectives.append((observed - target) ** 2)

    obj_values = np.asarray(objectives)

    return obj_values


def obj_fun_init(worker=None):
    env = Env()

    # load experimental data
    t_end = 60.0
    targets = {
        "Delta_mean": 23.491851806640625,
        "Theta_mean": 3.3645029067993164,
        "Alpha_mean": 0.62232905626297,
        "Beta_mean": 14.114240646362305,
        "Gamma_mean": 0.12134022265672684,
        "Delta_std": 29.584436416625977,
        "Theta_std": 3.922821044921875,
        "Alpha_std": 1.4222825765609741,
        "Beta_std": 99.20442962646484,
        "Gamma_std": 0.2624305188655853,
    }

    return partial(obj_fun, env=env, targets=targets, t_end=t_end)


def sigmoid(x, theta):
    return 1 / (1 + jnp.exp(-x + theta))


class Env:
    def __init__(self):
        self.system = json.load(
            open(os.path.join(os.path.dirname(__file__), "system.json"))
        )
        self.params = {}
        self.G = None
        self.W = None
        self.dx = 1.0
        self.laplacians = {}
        self.nodes = None
        self.node_to_idx = {}
        self.populations = ["E", "I"]
        self.diffusion_populations = ["E"]
        self.pop_indices = {}

    def set_params(self, params):
        self.params = params.copy()
        self._build_network()
        self._setup_populations()
        self._compute_laplacians()

    def _build_network(self):
        G = nx.DiGraph()
        for i, (x, y) in enumerate(self.system["electrodes"]):
            G.add_node(
                f"E{i}",
                population="E",
                pos=np.array([x, y]),
                electrode_id=i,
                population_type="excitatory",
            )

            G.add_node(
                f"I{i}",
                population="I",
                pos=np.array([x, y]),
                electrode_id=i,
                population_type="inhibitory",
            )

        distance_scale = self.params.get("distance_scale", 200.0)
        for i, pos_i in enumerate(self.system["electrodes"]):
            for j, pos_j in enumerate(self.system["electrodes"]):
                if i == j:
                    continue
                distance = np.sqrt(
                    (pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2
                )
                weight = np.exp(-distance / distance_scale)
                G.add_edge(
                    f"E{i}", f"E{j}", weight=self.params.get("E_E_weight", 1.0) * weight
                )
                G.add_edge(
                    f"E{i}", f"I{j}", weight=self.params.get("E_I_weight", 1.0) * weight
                )
                G.add_edge(
                    f"I{i}", f"E{j}", weight=self.params.get("I_E_weight", 1.0) * weight
                )
                G.add_edge(
                    f"I{i}", f"I{j}", weight=self.params.get("I_I_weight", 1.0) * weight
                )

        self.G = G
        self.W = jnp.array(nx.adjacency_matrix(G).toarray())

    def _setup_populations(self):
        self.nodes = list(self.G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        self.pop_indices = {}
        for pop in self.populations:
            self.pop_indices[pop] = np.array(
                [
                    i
                    for i, node in enumerate(self.nodes)
                    if self.G.nodes[node]["population"] == pop
                ]
            )

    def _compute_laplacians(self):
        self.laplacians = {}
        for pop in self.diffusion_populations:
            nodes = [n for n in self.G.nodes() if self.G.nodes[n]["population"] == pop]
            subgraph = self.G.subgraph(
                nodes
            ).to_undirected()  # Convert to undirected graph

            L = nx.laplacian_matrix(subgraph).toarray()

            full_L = np.zeros((len(self.nodes), len(self.nodes)))
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    full_L[self.node_to_idx[node1], self.node_to_idx[node2]] = L[i, j]

            self.laplacians[pop] = full_L

    def dynamics(self, t, state, args=None):
        # vectorized dynamics with diffusion
        E = state[::2]
        I = state[1::2]

        network_input = self.W @ state
        network_input_e = network_input[::2]
        network_input_i = network_input[1::2]

        local_e = self.params["E_E_c"] * E - self.params["E_I_c"] * I
        local_i = self.params["I_E_c"] * E - self.params["I_I_c"] * I

        diffusion_input_e = jnp.zeros_like(E)
        diffusion_input_i = jnp.zeros_like(I)

        if "E" in self.laplacians:
            pop_indices = self.pop_indices["E"]
            laplacian = self.laplacians["E"]
            diffusive_term = (self.params["E_diffusion_strength"] / (self.dx**2)) * (
                laplacian[pop_indices][:, pop_indices] @ E
            )
            diffusion_input_e = diffusive_term

        if "I" in self.laplacians:
            pop_indices = self.pop_indices["I"]
            laplacian = self.laplacians["I"]
            diffusive_term = (self.params["I_diffusion_strength"] / (self.dx**2)) * (
                laplacian[pop_indices][:, pop_indices] @ I
            )
            diffusion_input_i = diffusive_term

        total_e = local_e + network_input_e + diffusion_input_e
        total_i = local_i + network_input_i + diffusion_input_i

        dEdt = (-E + sigmoid(total_e, self.params["E_theta"])) / self.params["E_tau"]
        dIdt = (-I + sigmoid(total_i, self.params["I_theta"])) / self.params["I_tau"]

        dydt = jnp.zeros(len(E) + len(I))
        dydt = dydt.at[::2].set(dEdt)
        dydt = dydt.at[1::2].set(dIdt)

        return dydt

    def run(self, T, dt=0.05, initial_conditions=None):
        t0 = 0.0
        t1 = T
        steps = int(jnp.ceil((t1 - t0) / dt))
        ts = jnp.linspace(t0, t1, steps)

        if initial_conditions is None:
            initial_state = np.random.rand(len(self.nodes))
            initial_state = jnp.array(initial_state)
        else:
            initial_state = jnp.array([initial_conditions[node] for node in self.nodes])

        term = diffrax.ODETerm(self.dynamics)
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=initial_state,
            saveat=saveat,
            max_steps=100000,
        )
        return sol.ts, sol.ys.T


if __name__ == "__main__":
    env = Env()
    env.set_params(
        {
            "distance_scale": 100.0,
            "E_E_weight": 23.0,
            "E_I_weight": 1.0,
            "I_E_weight": 133.0,
            "I_I_weight": 1.0,
            "E_E_radius": 400.0,
            "E_I_radius": 200.0,
            "I_E_radius": 300.0,
            "I_I_radius": 400.0,
            "E_E_c": 10.0,
            "E_I_c": 10.0,
            "I_E_c": 10.0,
            "I_I_c": 0.0,
            "E_diffusion_strength": 0.5,
            "I_diffusion_strength": 0.0,
            "E_theta": 2.0,
            "I_theta": 3.5,
            "E_tau": 1.0,
            "I_tau": 2.0,
        }
    )

    t, y = env.run(60, 0.05)

    E = y[::2]
    I = y[1::2]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 20))
    n_channels = E.shape[0]
    offset = 0.5

    for i in range(n_channels):
        ax.plot(t, E[i] + i * offset, label=f"E {i}", color="blue", alpha=0.7)
        ax.plot(
            t, I[i] + i * offset, label=f"I {i}", color="red", alpha=0.7, linestyle="--"
        )

    ax.set_title("Neural Activity Over Time (All Channels)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity (offset per channel)")
    # ax.set_yticks([i * offset for i in range(n_channels)])
    # ax.set_yticklabels([f"Ch {i}" for i in range(n_channels)])
    # ax.grid(True)
    # ax.legend(loc="upper right", ncol=2, fontsize="small")
    fig.tight_layout()
    fig.savefig("neural_activity.png")
