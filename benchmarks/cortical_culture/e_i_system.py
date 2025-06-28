import json
import os
import networkx as nx
import numpy as np
from functools import partial
from benchmarks.cortical_culture.data_preprocessing import compute_band_power
import matplotlib.pyplot as plt
import warnings
from scipy.integrate import solve_ivp


def feature_dtypes(component):
    return [
        (feature_name, np.float32)
        for feature_name in component.config.dopt_params.objective_names
    ]


def obj_fun(pp, env, targets, t_end):
    env.set_params(pp)

    dt = 0.005  # twice as high as highest frequency (200 Hz)
    t, y = env.run(t_end, dt)

    data = np.array(y).T

    scale = 327.29  # scale to match experimental units
    offset = 0

    try:
        q = compute_band_power(data * scale + offset, fs=1 / dt)

        means = q.mean().to_dict()
        stds = q.std().to_dict()
    except:
        means = {}
        stds = {}

    objectives = []
    features = []
    for k, target in targets.items():
        if k.endswith("_mean"):
            observed = means.get(k.replace("_mean", ""), -1)
        elif k.endswith("_std"):
            observed = stds.get(k.replace("_std", ""), -1)

        objectives.append((observed - target) ** 2)
        features.append(observed)

    obj_values = np.asarray(objectives)
    features = np.asarray(
        [tuple(rf for rf in features)],
        dtype=np.dtype(
            [
                (k, np.float32)
                for k in [
                    "Delta_mean",
                    "Theta_mean",
                    "Alpha_mean",
                    "Beta_mean",
                    "Gamma_mean",
                    "Delta_std",
                    "Theta_std",
                    "Alpha_std",
                    "Beta_std",
                    "Gamma_std",
                ]
            ]
        ),
    )

    return obj_values, features


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


class Env:
    def __init__(self):
        self.system = json.load(
            open(os.path.join(os.path.dirname(__file__), "system.json"))
        )
        self.params = {}

    def set_params(
        self,
        params,
        spatial_kernel="gaussian",
        dx=0.5,
        diffusion_populations=None,
        saturation=None,
    ):
        self.params = params.copy()
        G = self._build_network()

        self.G = G.copy()
        self.spatial_kernel = spatial_kernel
        self.spatial_scale = self.params.get("distance_scale", 200.0)
        self.dx = dx
        self.diffusion_strength = self.params.get("diffusion_strength", 0.0)
        self.diffusion_populations = (
            [] if diffusion_populations is None else diffusion_populations
        )

        # Initialize graph with positions if missing
        self._initialize_positions()

        # Get population information
        self.populations = sorted(set(nx.get_node_attributes(G, "population").values()))
        self.n_populations = len(self.populations)

        # Create mappings
        self.pop_to_idx = {pop: idx for idx, pop in enumerate(self.populations)}
        self.idx_to_pop = {idx: pop for pop, idx in self.pop_to_idx.items()}
        self.nodes = list(G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.n_nodes = len(self.nodes)

        # Set default parameters
        self.tau = {"E": self.params["E_tau"], "I": self.params["I_tau"]}
        self.theta = {"E": self.params["E_theta"], "I": self.params["I_theta"]}
        self.gain = {"E": self.params["E_gain"], "I": self.params["I_gain"]}
        self.saturation = (
            saturation
            if saturation is not None
            else {pop: 1.0 for pop in self.populations}
        )

        # Create parameter arrays for vectorized operations
        self._create_parameter_arrays()

        # Pre-compute coupling and diffusion matrices
        self.coupling_matrices = self._compute_coupling_matrices()
        self.laplacians = self._compute_laplacians()

        # Create combined matrices for efficient computation
        self._create_combined_matrices()

        # Initialize external input
        self.external_input = np.zeros(self.n_nodes)

    def _initialize_positions(self):
        """Add default positions to nodes if missing"""
        pos = 0.0
        for node in self.G.nodes():
            if "pos" not in self.G.nodes[node]:
                nx.set_node_attributes(self.G, {node: {"pos": pos}})
                pos += self.dx

    def _create_parameter_arrays(self):
        """Create vectorized parameter arrays"""
        self.tau_array = np.array(
            [self.tau[self.G.nodes[node]["population"]] for node in self.nodes]
        )
        self.theta_array = np.array(
            [self.theta[self.G.nodes[node]["population"]] for node in self.nodes]
        )
        self.gain_array = np.array(
            [self.gain[self.G.nodes[node]["population"]] for node in self.nodes]
        )
        self.saturation_array = np.array(
            [self.saturation[self.G.nodes[node]["population"]] for node in self.nodes]
        )

        self.pop_indices = {}
        for pop in self.populations:
            self.pop_indices[pop] = np.array(
                [
                    i
                    for i, node in enumerate(self.nodes)
                    if self.G.nodes[node]["population"] == pop
                ]
            )

    def _spatial_weight(self, pos1, pos2):
        """Compute spatial weighting between two positions"""
        if self.spatial_kernel is None:
            return 1.0

        if np.isscalar(pos1):
            distance = abs(pos1 - pos2)
        else:
            distance = np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))

        if self.spatial_kernel == "exponential":
            return np.exp(-distance / self.spatial_scale)
        elif self.spatial_kernel == "gaussian":
            return np.exp(-((distance / self.spatial_scale) ** 2))
        else:
            raise ValueError(f"Unknown spatial kernel: {self.spatial_kernel}")

    def _compute_coupling_matrices(self):
        """Compute coupling matrices for all population interactions"""
        matrices = {}

        # Initialize matrices for all population pairs
        for pop1 in self.populations:
            for pop2 in self.populations:
                matrices[(pop1, pop2)] = np.zeros((self.n_nodes, self.n_nodes))

        # Fill matrices based on graph edges and spatial positions
        for edge in self.G.edges(data=True):
            source, target = edge[0], edge[1]
            base_weight = edge[2].get("weight", 1.0)

            source_pop = self.G.nodes[source]["population"]
            target_pop = self.G.nodes[target]["population"]

            source_idx = self.node_to_idx[source]
            target_idx = self.node_to_idx[target]

            # Get positions
            source_pos = np.array(self.G.nodes[source]["pos"])
            target_pos = np.array(self.G.nodes[target]["pos"])

            # Compute spatial weight
            spatial_factor = self._spatial_weight(source_pos, target_pos)

            matrices[(source_pop, target_pop)][target_idx, source_idx] = (
                base_weight * spatial_factor
            )

        return matrices

    def _compute_laplacians(self):
        """Compute Laplacian matrices for diffusive coupling"""
        laplacians = {}

        for pop in self.populations:
            if pop not in self.diffusion_populations:
                continue

            # Create subgraph for this population
            nodes = [n for n in self.G.nodes() if self.G.nodes[n]["population"] == pop]
            subgraph = self.G.subgraph(nodes)

            # Compute Laplacian matrix
            L = nx.laplacian_matrix(subgraph).toarray()

            # Convert to full node ordering
            full_L = np.zeros((self.n_nodes, self.n_nodes))
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    full_L[self.node_to_idx[node1], self.node_to_idx[node2]] = L[i, j]

            laplacians[pop] = full_L

        return laplacians

    def _create_combined_matrices(self):
        """Create combined coupling and diffusion matrices"""

        # Combine all coupling matrices into one large matrix
        self.W_total = np.zeros((self.n_nodes, self.n_nodes))

        for target_pop in self.populations:
            target_indices = self.pop_indices[target_pop]
            for source_pop in self.populations:
                source_indices = self.pop_indices[source_pop]
                coupling = self.coupling_matrices[(source_pop, target_pop)]
                self.W_total[np.ix_(target_indices, source_indices)] = coupling[
                    np.ix_(target_indices, source_indices)
                ]

        # Create combined diffusion matrix
        self.L_total = np.zeros((self.n_nodes, self.n_nodes))
        for pop in self.populations:
            if pop in self.laplacians:
                pop_indices = self.pop_indices[pop]
                laplacian = self.laplacians[pop]
                self.L_total[np.ix_(pop_indices, pop_indices)] = laplacian[
                    np.ix_(pop_indices, pop_indices)
                ]

        # Scale diffusion matrix
        self.L_total *= self.diffusion_strength / (self.dx**2)

    def sigmoid(self, x, theta, gain, saturation):
        """
        Sigmoid activation function

        Parameters:
        x: input
        theta: threshold
        gain: slope/gain parameter (higher = steeper)
        saturation: saturation level (max output)
        """
        # Clip input to prevent overflow
        x_clipped = np.clip(gain * (x - theta), -500, 500)
        return saturation / (1 + np.exp(-x_clipped))

    def sigmoid_derivative(self, x, theta, gain, saturation):
        """Derivative of sigmoid function"""
        x_clipped = np.clip(gain * (x - theta), -500, 500)
        exp_term = np.exp(-x_clipped)
        return (saturation * gain * exp_term) / ((1 + exp_term) ** 2)

    def compute_input(self, state):
        """Compute total input to each node"""
        # Synaptic coupling
        synaptic_input = self.W_total @ state

        # Diffusive coupling
        diffusive_input = self.L_total @ state

        # Total input
        total_input = synaptic_input + diffusive_input + self.external_input

        return total_input

    def dynamics(self, t, state):
        """Compute the right-hand side of the Wilson-Cowan equations"""
        # Compute total input
        total_input = self.compute_input(state)

        # Compute sigmoid activation
        activation = self.sigmoid(
            total_input, self.theta_array, self.gain_array, self.saturation_array
        )

        # Compute derivatives
        derivatives = (-state + activation) / self.tau_array

        return derivatives

    def jacobian(self, t, state):
        """
        Compute the analytical Jacobian matrix

        The Jacobian J[i,j] = \frac{\partial (dx_i/dt)} {\partial x_j}

        For Wilson-Cowan: dx_i/dt = (-x_i + sigmoid(input_i)) / tau_i

        Therefore: J[i,j] = \frac{\partial}{\partial x_j} [(-x_i + sigmoid(input_i)) / tau_i]
                          = (1/tau_i) * [\partial sigmoid(input_i) / \partial x_j - s_ij]
                          = (1/tau_i) * [sigmoid'(input_i) * \partial input_i / \partial x_j - s_ij]

        Where \frac{\partial input_i} {\partial x_j} = W_ij + L_ij (coupling + diffusion terms)
        """
        # Compute total input
        total_input = self.compute_input(state)

        # Compute sigmoid derivative
        sigmoid_deriv = self.sigmoid_derivative(
            total_input, self.theta_array, self.gain_array, self.saturation_array
        )

        # Create Jacobian matrix
        J = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            tau_i = self.tau_array[i]
            sigmoid_deriv_i = sigmoid_deriv[i]

            for j in range(self.n_nodes):
                if i == j:
                    # Diagonal terms: -1/tau_i + (1/tau_i) * sigmoid'_i * (W_ii + L_ii)
                    J[i, j] = (
                        -1 + sigmoid_deriv_i * (self.W_total[i, j] + self.L_total[i, j])
                    ) / tau_i
                else:
                    # Off-diagonal terms: (1/tau_i) * sigmoid'_i * (W_ij + L_ij)
                    J[i, j] = (
                        sigmoid_deriv_i
                        * (self.W_total[i, j] + self.L_total[i, j])
                        / tau_i
                    )

        return J

    def set_external_input(self, input_dict=None, input_array=None):
        """Set external input to nodes"""
        if input_dict is not None:
            for node, value in input_dict.items():
                if node in self.node_to_idx:
                    self.external_input[self.node_to_idx[node]] = value
        elif input_array is not None:
            if len(input_array) == self.n_nodes:
                self.external_input = np.array(input_array)
            else:
                raise ValueError("Input array length must match number of nodes")

    def simulate(
        self,
        T,
        dt=None,
        initial_conditions=None,
        method="DOP853",
        rtol=1e-6,
        atol=1e-9,
        max_steps=10000,
    ):
        """
        Simulate the model

        Parameters:
        T (float): Total simulation time
        dt (float): Output time step (for scipy) or output interval (for CVODE)
        initial_conditions (array or dict): Initial conditions
        method (str): Integration method ('CVODE', 'DOP853', 'RK45', 'Radau', 'BDF', 'LSODA')
        rtol, atol (float): Relative and absolute tolerances
        max_steps (int): Maximum number of internal steps

        Returns:
        tuple: Time points and solution arrays
        """
        # Set default dt if not provided
        if dt is None:
            dt = T / 1000

        # Prepare initial conditions
        if initial_conditions is None:
            y0 = np.random.rand(self.n_nodes)
        elif isinstance(initial_conditions, dict):
            y0 = np.array([initial_conditions.get(node, 0.1) for node in self.nodes])
        else:
            y0 = np.array(initial_conditions)

        return self._simulate_scipy(T, dt, y0, method, rtol, atol, max_steps)

    def _simulate_scipy(self, T, dt, y0, method, rtol, atol, max_steps):
        """Simulate using scipy.integrate.solve_ivp"""
        t_span = (0, T)
        t_eval = np.arange(0, T + dt, dt)

        # Prepare Jacobian for scipy
        def jac_scipy(t, y):
            return self.jacobian(t, y)

        # Solve the system
        if method in ["Radau", "BDF"]:
            # These methods can use Jacobian
            solution = solve_ivp(
                self.dynamics,
                t_span,
                y0,
                method=method,
                t_eval=t_eval,
                jac=jac_scipy,
                rtol=rtol,
                atol=atol,
                max_step=dt * 10,
            )
        else:
            # Other methods don't use Jacobian
            solution = solve_ivp(
                self.dynamics,
                t_span,
                y0,
                method=method,
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
                max_step=dt * 10,
            )

        if not solution.success:
            warnings.warn(f"Integration failed: {solution.message}")

        return solution.t, solution.y

    def _build_network(self):
        G = nx.DiGraph()
        # Add nodes
        for i, (x, y) in enumerate(self.system["electrodes"]):
            G.add_node(f"E{i}", population="E", pos=np.array([x, y]) / 1000)
            G.add_node(f"I{i}", population="I", pos=np.array([x, y]) / 1000)

        # Add connections
        N = len(self.system["electrodes"])
        for i in range(N):
            # E-E connections (recurrent excitation)
            G.add_edge(
                f"E{i}", f"E{(i + 1) % N}", weight=self.params.get("E_E_weight", 0.0)
            )

            # I-I connections (recurrent inhibition)
            G.add_edge(
                f"I{i}", f"I{(i + 1) % N}", weight=-self.params.get("I_I_weight", 0.0)
            )
            G.add_edge(
                f"I{i}", f"I{(i - 1) % N}", weight=-self.params.get("I_I_weight", 0.0)
            )

            # E->I connections
            G.add_edge(f"E{i}", f"I{i}", weight=self.params.get("E_I_weight", 1.0))
            G.add_edge(
                f"E{i}", f"I{(i + 1) % N}", weight=self.params.get("E_I_weight", 1.0)
            )
            G.add_edge(
                f"E{i}", f"I{(i - 1) % N}", weight=self.params.get("E_I_weight", 1.0)
            )

            # I->E connections, wider inhibition
            for j in range(-3, 4):
                G.add_edge(
                    f"I{i}",
                    f"E{(i + j) % N}",
                    weight=-self.params.get("I_E_weight", 1.0),
                )

        return G

    def run(self, T, dt=0.05, initial_conditions=None):
        t, y = self.simulate(
            T, dt, initial_conditions=initial_conditions, method="DOP853"
        )
        return t, y


if __name__ == "__main__":
    env = Env()
    env.set_params(
        {
            "distance_scale": 1.5,
            "E_E_weight": 10.0,
            "E_I_weight": 4.0,
            "I_E_weight": 3.0,
            "I_I_weight": 1.0,
            "diffusion_strength": 1.0,
            "E_theta": 4.0,
            "I_theta": 3.0,
            "E_tau": 0.02,
            "I_tau": 0.01,
            "E_gain": 2.0,
            "I_gain": 3.0,
        }
    )

    t, y = env.run(60, 0.05)

    E = y[::2]
    I = y[1::2]

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
