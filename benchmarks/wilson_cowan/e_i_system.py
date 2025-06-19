import json
import os
import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
from machinable.utils import load_file


def obj_fun(pp, env, targets, t_end):
    env.set_params(pp)

    t, y = env.run(t_end, 0.5)

    E_activity = y[::2, :]

    spike_threshold = 0.5
    E_diff = np.diff(E_activity > spike_threshold, axis=1)
    spikes = [[] for _ in range(E_activity.shape[0])]
    for channel_id in range(E_activity.shape[0]):
        spike_indices = np.where(E_diff[channel_id, :] > 0)[0]
        spike_times = t[spike_indices]
        spikes[channel_id] = spike_times

    if len(spikes) == 0:
        return np.asarray([999.0])

    bin_size = 1.0  # seconds
    num_bins = int(np.ceil(t[-1] / bin_size))
    model_binned_rates = np.zeros((len(spikes), num_bins))

    for channel_id, neuron_spikes in enumerate(spikes):
        if len(neuron_spikes) > 0:
            bin_indices = np.floor(neuron_spikes / bin_size).astype(int)
            for bin_idx in bin_indices:
                if bin_idx < num_bins:
                    model_binned_rates[channel_id, bin_idx] += 1

    target_binned_rates = np.zeros((len(targets), num_bins))
    for channel_id, neuron_spikes in enumerate(targets):
        if len(neuron_spikes) > 0:
            bin_indices = np.floor(neuron_spikes / bin_size).astype(int)
            for bin_idx in bin_indices:
                if bin_idx < num_bins:
                    target_binned_rates[channel_id, bin_idx] += 1

    if np.max(model_binned_rates) > 0:
        model_binned_rates = model_binned_rates / np.max(model_binned_rates)
    if np.max(target_binned_rates) > 0:
        target_binned_rates = target_binned_rates / np.max(target_binned_rates)

    firing_mismatch = np.mean((model_binned_rates - target_binned_rates) ** 2)

    obj_values = np.asarray([firing_mismatch])

    return obj_values


def obj_fun_init(
    experimental_data,
    worker=None,
):
    env = Env()

    t_end = 0.5
    targets = load_file(experimental_data)
    targets = [
        np.array(channel_data)[np.array(channel_data) <= t_end]
        for channel_data in targets
    ]

    return partial(obj_fun, env=env, targets=targets, t_end=t_end)


def sigmoid(x, theta):
    return 1 / (1 + np.exp(-x + theta))


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

        connection_radius = self.params.get("connection_radius", 400.0)  # um

        for i, pos_i in enumerate(self.system["electrodes"]):
            for j, pos_j in enumerate(self.system["electrodes"]):
                if i == j:
                    continue

                distance = np.sqrt(
                    (pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2
                )

                if distance <= connection_radius:
                    conn_prob = np.exp(-distance / (connection_radius / 2))

                    seed_value = hash(f"{i}_{j}") % (2**32)
                    rng = np.random.RandomState(seed_value)

                    if rng.random() < conn_prob:
                        weight_ee = self.params.get("E_E_weight", 1.0) * np.exp(
                            -distance / self.params.get("E_E_radius", connection_radius)
                        )
                        G.add_edge(f"E{i}", f"E{j}", weight=weight_ee)

                        weight_ei = self.params.get("E_I_weight", 1.0) * np.exp(
                            -distance
                            / (self.params.get("E_I_radius", connection_radius))
                        )
                        G.add_edge(f"E{i}", f"I{j}", weight=weight_ei)

                        weight_ie = (
                            -1
                            * self.params.get("I_E_weight", 1.0)
                            * np.exp(
                                -distance
                                / (self.params.get("I_E_radius", connection_radius))
                            )
                        )
                        G.add_edge(f"I{i}", f"E{j}", weight=weight_ie)

                        weight_ii = (
                            -1
                            * self.params.get("I_I_weight", 1.0)
                            * np.exp(
                                -distance
                                / (self.params.get("I_I_radius", connection_radius))
                            )
                        )
                        G.add_edge(f"I{i}", f"I{j}", weight=weight_ii)

        self.G = G
        self.W = nx.adjacency_matrix(G).toarray()

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

    def dynamics(self, t, state):
        # vectorized dynamics with diffusion
        E = state[::2]
        I = state[1::2]

        network_input = self.W.dot(state)
        network_input_e = network_input[::2]
        network_input_i = network_input[1::2]

        local_e = self.params["E_E_c"] * E - self.params["E_I_c"] * I
        local_i = self.params["I_E_c"] * E - self.params["I_I_c"] * I

        diffusion_input_e = np.zeros_like(E)
        diffusion_input_i = np.zeros_like(I)

        if "E" in self.laplacians:
            pop_indices = self.pop_indices["E"]
            laplacian = self.laplacians["E"]
            diffusive_term = (self.params["E_diffusion_strength"] / (self.dx**2)) * (
                laplacian[pop_indices] @ state
            )
            diffusion_input_e = diffusive_term

        if "I" in self.laplacians:
            pop_indices = self.pop_indices["I"]
            laplacian = self.laplacians["I"]
            diffusive_term = (self.params["I_diffusion_strength"] / (self.dx**2)) * (
                laplacian[pop_indices] @ state
            )
            diffusion_input_i = diffusive_term

        e_noise = 0
        if self.params.get("E_noise", 0) > 0:
            e_noise = np.random.normal(0, self.params["E_noise"], len(E))

        i_noise = 0
        if self.params.get("I_noise", 0) > 0:
            i_noise = np.random.normal(0, self.params["I_noise"], len(I))

        total_e = local_e + network_input_e + diffusion_input_e + e_noise
        total_i = local_i + network_input_i + diffusion_input_i + i_noise

        dEdt = (-E + sigmoid(total_e, self.params["E_theta"])) / self.params["E_tau"]
        dIdt = (-I + sigmoid(total_i, self.params["I_theta"])) / self.params["I_tau"]

        dydt = np.zeros(len(E) + len(I))
        dydt[::2] = dEdt
        dydt[1::2] = dIdt

        return dydt

    def run(self, T, dt, initial_conditions=None):
        t_span = (0, T)
        t_eval = np.arange(0, T, dt)

        if initial_conditions is None:
            initial_state = 0.1 * np.random.rand(len(self.nodes))
        else:
            initial_state = np.array([initial_conditions[node] for node in self.nodes])

        solution = solve_ivp(
            self.dynamics,
            t_span,
            initial_state,
            method="DOP853",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-6,
        )

        return solution.t, solution.y


if __name__ == "__main__":
    env = Env()
    env.set_params(
        {
            "connection_radius": 400.0,
            "E_E_weight": 1.0,
            "E_I_weight": 1.0,
            "I_E_weight": 1.0,
            "I_I_weight": 1.0,
            "E_E_radius": 400.0,
            "E_I_radius": 400.0,
            "I_E_radius": 400.0,
            "I_I_radius": 400.0,
            "E_E_c": 10.0,
            "E_I_c": 10.0,
            "I_E_c": 10.0,
            "I_I_c": 0.0,
            "E_diffusion_strength": 0.5,
            "I_diffusion_strength": 0.0,
            "E_noise": 0.1,
            "I_noise": 0.1,
            "E_theta": 2.0,
            "I_theta": 3.5,
            "E_tau": 1.0,
            "I_tau": 2.0,
        }
    )

    t, y = env.run(10, 1)

    E = y[::2]
    I = y[1::2]

    print(E)
