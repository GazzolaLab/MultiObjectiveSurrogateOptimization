import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import networkx as nx
from collections import defaultdict
import time
import warnings
from scipy.integrate import solve_ivp
try:
    from scikits.odes import ode
    has_sundials = True
except ImportError:
    has_sundials = False

class WilsonCowanGraph:
    def __init__(self, G, spatial_kernel='exponential', spatial_scale=1.0, 
                 dx=1.0, diffusion_strength=1.0, diffusion_populations=None,
                 tau=None, theta=None, use_cvode=True):
        """
        Initialize Wilson-Cowan model on a graph with spatial and
        diffusive coupling.
        
        Parameters:
        G (networkx.DiGraph): Graph where:
            - nodes have 'population' attribute indicating population type
            - edges have 'weight' attribute for coupling strength
        spatial_kernel (str): Type of spatial kernel ('exponential', 'gaussian', None)
        spatial_scale (float): Spatial scale parameter
        dx (float): Spatial discretization
        diffusion_strength (float): Strength of diffusive coupling
        diffusion_populations (list): Populations that have diffusive coupling
        tau (dict): Time constants for each population type
        theta (dict): Threshold values for each population type
        use_cvode (bool): Whether to use CVODE (requires scikits.odes)
        """
        self.G = G.copy()
        self.spatial_kernel = spatial_kernel
        self.spatial_scale = spatial_scale
        self.dx = dx
        self.diffusion_strength = diffusion_strength
        self.diffusion_populations = [] if diffusion_populations is None else diffusion_populations
        self.use_cvode = use_cvode
        if use_cvode and not has_sundials:
            warnings.warn("CVODE requested but scikits.odes not available.")
            self.use_cvode = False
        
        # Initialize graph with positions if missing
        self._initialize_positions()
        
        # Get population information
        self.populations = sorted(set(nx.get_node_attributes(G, 'population').values()))
        self.n_populations = len(self.populations)
        
        # Create mappings
        self.pop_to_idx = {pop: idx for idx, pop in enumerate(self.populations)}
        self.idx_to_pop = {idx: pop for pop, idx in self.pop_to_idx.items()}
        self.nodes = list(G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.n_nodes = len(self.nodes)
        
        # Set default parameters
        self.tau = tau if tau is not None else {pop: 1.0 for pop in self.populations}
        self.theta = theta if theta is not None else {pop: 4.0 for pop in self.populations}
        
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
            if 'pos' not in self.G.nodes[node]:
                nx.set_node_attributes(self.G, {node: {'pos': pos}})
                pos += self.dx
                
    def _create_parameter_arrays(self):
        """Create vectorized parameter arrays"""
        self.tau_array = np.array([
            self.tau[self.G.nodes[node]['population']]
            for node in self.nodes
        ])
        self.theta_array = np.array([
            self.theta[self.G.nodes[node]['population']]
            for node in self.nodes
        ])
        
        self.pop_indices = {}
        for pop in self.populations:
            self.pop_indices[pop] = np.array([
                i for i, node in enumerate(self.nodes)
                if self.G.nodes[node]['population'] == pop
            ])
    
    def _spatial_weight(self, pos1, pos2):
        """Compute spatial weighting between two positions"""
        if self.spatial_kernel is None:
            return 1.0
            
        if np.isscalar(pos1):
            distance = abs(pos1 - pos2)
        else:
            distance = np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))
        
        if self.spatial_kernel == 'exponential':
            return np.exp(-distance / self.spatial_scale)
        elif self.spatial_kernel == 'gaussian':
            return np.exp(-(distance / self.spatial_scale) ** 2)
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
            base_weight = edge[2].get('weight', 1.0)
            
            source_pop = self.G.nodes[source]['population']
            target_pop = self.G.nodes[target]['population']
            
            source_idx = self.node_to_idx[source]
            target_idx = self.node_to_idx[target]

            # Get positions
            source_pos = np.array(self.G.nodes[source]['pos'])
            target_pos = np.array(self.G.nodes[target]['pos'])
            
            # Compute spatial weight
            spatial_factor = self._spatial_weight(source_pos, target_pos)
            
            matrices[(source_pop, target_pop)][target_idx, source_idx] = base_weight * spatial_factor
        
        return matrices
    
    def _compute_laplacians(self):
        """Compute Laplacian matrices for diffusive coupling"""
        laplacians = {}
        
        for pop in self.populations:
            if pop not in self.diffusion_populations:
                continue
            
            # Create subgraph for this population
            nodes = [n for n in self.G.nodes() if self.G.nodes[n]['population'] == pop]
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
                self.W_total[np.ix_(target_indices, source_indices)] = coupling[np.ix_(target_indices, source_indices)]
        
        # Create combined diffusion matrix
        self.L_total = np.zeros((self.n_nodes, self.n_nodes))
        for pop in self.populations:
            if pop in self.laplacians:
                pop_indices = self.pop_indices[pop]
                laplacian = self.laplacians[pop]
                self.L_total[np.ix_(pop_indices, pop_indices)] = laplacian[np.ix_(pop_indices, pop_indices)]
        
        # Scale diffusion matrix
        self.L_total *= (self.diffusion_strength / (self.dx ** 2))
    
    def sigmoid(self, x, theta):
        """Sigmoid activation function"""
        # Clip input to prevent overflow
        x_clipped = np.clip(x - theta, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def sigmoid_derivative(self, x, theta):
        """Derivative of sigmoid function"""
        s = self.sigmoid(x, theta)
        return s * (1 - s)
    
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
        activation = self.sigmoid(total_input, self.theta_array)
        
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
        sigmoid_deriv = self.sigmoid_derivative(total_input, self.theta_array)
        
        # Create Jacobian matrix
        J = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            tau_i = self.tau_array[i]
            sigmoid_deriv_i = sigmoid_deriv[i]
            
            for j in range(self.n_nodes):
                if i == j:
                    # Diagonal terms: -1/tau_i + (1/tau_i) * sigmoid'_i * (W_ii + L_ii)
                    J[i, j] = (-1 + sigmoid_deriv_i * (self.W_total[i, j] + self.L_total[i, j])) / tau_i
                else:
                    # Off-diagonal terms: (1/tau_i) * sigmoid'_i * (W_ij + L_ij)
                    J[i, j] = sigmoid_deriv_i * (self.W_total[i, j] + self.L_total[i, j]) / tau_i
        
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
    
    def simulate(self, T, dt=None, initial_conditions=None, method='CVODE', 
                 rtol=1e-6, atol=1e-9, max_steps=10000):
        """
        Simulate the model using CVODE or scipy
        
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
            y0 = 0.1 * np.random.rand(self.n_nodes)
        elif isinstance(initial_conditions, dict):
            y0 = np.array([initial_conditions.get(node, 0.1) for node in self.nodes])
        else:
            y0 = np.array(initial_conditions)
        
        # Use CVODE if requested
        if self.use_cvode and method == 'CVODE':
            return self._simulate_cvode(T, dt, y0, rtol, atol, max_steps)
        else:
            return self._simulate_scipy(T, dt, y0, method, rtol, atol, max_steps)
    
    def _simulate_cvode(self, T, dt, y0, rtol, atol, max_steps):
        """Simulate using CVODE from scikits.odes"""
        # Create time points
        t_eval = np.arange(0, T + dt, dt)
        
        # Set up CVODE solver
        solver = ode('cvode', self.dynamics, jacfn=self.jacobian)
        
        # Set solver options
        solver.set_options(
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            linsolver='dense',  # Use dense linear solver for Jacobian
            first_step=dt/100,  # Initial step size
            max_step=dt*10      # Maximum step size
        )
        
        # Initialize solver
        solver.init_step(0.0, y0)
        
        # Solve
        solution, info = solver.solve(t_eval, y0)
        
        if info.errors.t:
            warnings.warn(f"CVODE integration errors: {info.errors}")
        
        return t_eval, solution.T  # Transpose to match scipy format
    
    def _simulate_scipy(self, T, dt, y0, method, rtol, atol, max_steps):
        """Simulate using scipy.integrate.solve_ivp"""
        t_span = (0, T)
        t_eval = np.arange(0, T + dt, dt)
        
        # Prepare Jacobian for scipy
        def jac_scipy(t, y):
            return self.jacobian(t, y)
        
        # Solve the system
        if method in ['Radau', 'BDF']:
            # These methods can use Jacobian
            solution = solve_ivp(
                self.dynamics, t_span, y0,
                method=method, t_eval=t_eval,
                jac=jac_scipy,
                rtol=rtol, atol=atol,
                max_step=dt*10
            )
        else:
            # Other methods don't use Jacobian
            solution = solve_ivp(
                self.dynamics, t_span, y0,
                method=method, t_eval=t_eval,
                rtol=rtol, atol=atol,
                max_step=dt*10
            )
        
        if not solution.success:
            warnings.warn(f"Integration failed: {solution.message}")
        
        return solution.t, solution.y

    # Torch version of dynamics function
    def dynamics_torch(self, x):
        """PyTorch version of the Wilson-Cowan dynamics function."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for dynamics_torch. ")
        # Convert matrices to torch tensors
        W_total_torch = torch.tensor(self.W_total, dtype=torch.float64)
        L_total_torch = torch.tensor(self.L_total, dtype=torch.float64)
        tau_array_torch = torch.tensor(self.tau_array, dtype=torch.float64)
        theta_array_torch = torch.tensor(self.theta_array, dtype=torch.float64)
        external_torch = torch.tensor(self.external_input, dtype=torch.float64)
        
        synaptic_input = W_total_torch @ x
        diffusive_input = L_total_torch @ x
        total_input = synaptic_input + diffusive_input + external_torch
        
        activation = torch.sigmoid(total_input - theta_array_torch)
        
        derivatives = (-x + activation) / tau_array_torch
        
        return derivatives

    def jacobian_autodiff(self, t, state):
        """
        Compute the Jacobian matrix using PyTorch automatic differentiation.

        Parameters:
        t (float): Time (not used but kept for interface compatibility)
        state (array): Current state of all nodes

        Returns:
        ndarray: Jacobian matrix (n_nodes, n_nodes)
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for automatic differentiation. ")

        # Convert state to torch tensor
        x_torch = torch.tensor(state, dtype=torch.float64, requires_grad=True)
        
        # Compute Jacobian using functional interface
        J_torch = torch.autograd.functional.jacobian(self.dynamics_torch, x_torch)
        
        # Convert back to numpy
        return J_torch.detach().numpy()

    def verify_jacobian(self, t, state, rtol=1e-6, atol=1e-8):
        """
        Compares analytical Jacobian with automatic differentiation version.

        Parameters:
        t (float): Time
        state (array): Current state
        rtol, atol (float): Relative and absolute tolerances for comparison

        Returns:
        dict: Verification results including max error
        """
        try:
            # Compute both Jacobians
            J_analytical = self.jacobian(t, state)
            J_autodiff = self.jacobian_autodiff(t, state)

            # Compare
            diff = np.abs(J_analytical - J_autodiff)
            max_error = np.max(diff)
            mean_error = np.mean(diff)

            # Check agreement within tolerance
            agreement = np.allclose(J_analytical, J_autodiff, rtol=rtol, atol=atol)

            results = {
                'analytical_jacobian': J_analytical,
                'autodiff_jacobian': J_autodiff,
                'max_error': max_error,
                'mean_error': mean_error,
                'agreement': agreement,
                'rtol': rtol,
                'atol': atol
            }

            # Print summary
            if agreement:
                print(f"Jacobian verification PASSED")
                print(f"   Max error: {max_error:.2e}")
                print(f"   Mean error: {mean_error:.2e}")
            else:
                print(f"Jacobian verification FAILED")
                print(f"   Max error: {max_error:.2e} (tolerance: {atol:.2e})")
                print(f"   Mean error: {mean_error:.2e}")

                max_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"   Max error at position [{max_idx[0]}, {max_idx[1]}]:")
                print(f"     Analytical: {J_analytical[max_idx]:.6e}")
                print(f"     Autodiff:   {J_autodiff[max_idx]:.6e}")

            return results

        except Exception as e:
            print(f"Jacobian verification failed with error: {e}")
            return {'error': str(e), 'agreement': False}
    
    def analyze_stability(self, equilibrium=None):
        """
        Analyze linear stability around an equilibrium point
        
        Parameters:
        equilibrium (array): Equilibrium state to analyze (if None, find numerically)
        
        Returns:
        dict: Stability analysis results
        """
        if equilibrium is None:
            # Find equilibrium numerically
            from scipy.optimize import fsolve
            
            def equilibrium_condition(state):
                return self.dynamics(0.0, state)
            
            # Start from random initial guess
            x0 = 0.1 * np.random.rand(self.n_nodes)
            equilibrium = fsolve(equilibrium_condition, x0)
            
            if np.max(np.abs(self.dynamics(0.0, equilibrium))) > 1e-4:
                warnings.warn("Could not find accurate equilibrium point")
        
        # Compute Jacobian at equilibrium
        J = self.jacobian(0, equilibrium)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(J)
        
        max_real_part = np.max(np.real(eigenvalues))
        is_stable = max_real_part < 0
        
        # Find oscillatory modes
        oscillatory_freqs = np.imag(eigenvalues[np.imag(eigenvalues) > 1e-10]) / (2 * np.pi)
        
        results = {
            'equilibrium': equilibrium,
            'jacobian': J,
            'eigenvalues': eigenvalues,
            'max_real_part': max_real_part,
            'is_stable': is_stable,
            'oscillatory_frequencies': oscillatory_freqs,
            'condition_number': np.linalg.cond(J)
        }
        
        return results
    
    def plot_eigenvalue_spectrum(self, equilibrium=None, figsize=(10, 6)):
        """Plot eigenvalue spectrum of the Jacobian"""
        stability_results = self.analyze_stability(equilibrium)
        eigenvalues = stability_results['eigenvalues']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Complex plane plot
        ax1.scatter(np.real(eigenvalues), np.imag(eigenvalues), alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Stability boundary')
        ax1.set_xlabel('Real part')
        ax1.set_ylabel('Imaginary part')
        ax1.set_title('Eigenvalue Spectrum')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Real parts histogram
        ax2.hist(np.real(eigenvalues), bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Stability boundary')
        ax2.set_xlabel('Real part of eigenvalues')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Real Parts')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig, (ax1, ax2), stability_results
    
    def get_population_activities(self, solution):
        """
        Extract activities for each population from solution
        
        Parameters:
        solution (array): Solution array
        
        Returns:
        dict: Dictionary of activities for each population
        """
        activities = {}
        for pop in self.populations:
            pop_indices = self.pop_indices[pop]
            activities[pop] = solution[pop_indices]
        return activities
    
    
    def plot_state(self, t, solution, time, pop1='E'):
        """Plot current state of the system"""
        plt.figure(figsize=(10, 4))
        
        activities = self.get_population_activities(solution)

        time_index = np.argwhere(t >= time)
        if len(time_index) == 0:
            time_index = -1
        else:
            time_index = time_index[0]

        pop_activity = dict({ p: [] for p in self.populations })
        for pop, activity in activities.items():
            for node in self.G.nodes():
                node_pop = self.G.nodes[node]['population']
                if pop != node_pop:
                    continue
                idx = self.node_to_idx[node]
                activity = solution[idx, time_index]
                pop_activity[pop].append(activity)

        for pop, activities in pop_activity.items():
            pop_activity = np.asarray(activities)
            if pop == pop1:
                color = 'r'
            else:
                color = 'b'
            plt.plot(pop_activity, label=pop, color=color)
                
        plt.xlabel('Space')
        plt.ylabel('Activity')
        plt.legend()
        plt.title(f't = {t[time_index]}')
        plt.grid(True)
        plt.show()
    
    def plot_coupling_matrices(self, figsize=(12, 4)):
        """
        Visualize coupling matrices between populations
        
        Parameters:
        figsize (tuple): Figure size (width, height)
        """
        n_pops = len(self.populations)
        fig, axes = plt.subplots(1, n_pops**2, figsize=figsize)
        
        # Flatten axes array if only one population
        if n_pops == 1:
            axes = [axes]
        
        # Get global min and max for consistent color scaling
        all_weights = []
        for source_pop in self.populations:
            for target_pop in self.populations:
                matrix = self.coupling_matrices[(source_pop, target_pop)]
                all_weights.extend(matrix.flatten())
        vmin, vmax = min(all_weights), max(all_weights)
        
        # Create a diverging colormap centered at 0
        vabs = max(abs(vmin), abs(vmax))
        norm = plt.Normalize(-vabs, vabs)
        cmap = plt.cm.RdBu_r
        
        # Plot each coupling matrix
        for i, target_pop in enumerate(self.populations):
            for j, source_pop in enumerate(self.populations):
                ax_idx = i * n_pops + j
                ax = axes[ax_idx]
                
                # Get indices for this population pair
                target_indices = [
                    idx for idx, node in enumerate(self.nodes)
                    if self.G.nodes[node]['population'] == target_pop
                ]
                source_indices = [
                    idx for idx, node in enumerate(self.nodes)
                    if self.G.nodes[node]['population'] == source_pop
                ]
                
                # Extract relevant submatrix
                matrix = self.coupling_matrices[(source_pop, target_pop)][target_indices][:, source_indices]
                
                # Plot matrix
                im = ax.imshow(matrix, cmap=cmap, norm=norm)
                ax.set_title(f'{source_pop} → {target_pop}')
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
                
                # Set labels
                ax.set_xlabel(f'{source_pop} nodes')
                ax.set_ylabel(f'{target_pop} nodes')
                
                # Add grid
                ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_full_coupling_matrix(self, figsize=(8, 8)):
        """
        Visualize the full coupling matrix with population boundaries
        
        Parameters:
        figsize (tuple): Figure size (width, height)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Combine all coupling matrices into one large matrix
        n_nodes = len(self.nodes)
        full_matrix = np.zeros((n_nodes, n_nodes))
        
        # Fill the matrix
        for i, target_node in enumerate(self.nodes):
            for j, source_node in enumerate(self.nodes):
                target_pop = self.G.nodes[target_node]['population']
                source_pop = self.G.nodes[source_node]['population']
                full_matrix[i, j] = self.coupling_matrices[(source_pop, target_pop)][i, j]
        
        # Get color scaling
        vabs = max(abs(full_matrix.min()), abs(full_matrix.max()))
        norm = plt.Normalize(-vabs, vabs)
        
        # Plot matrix
        im = ax.imshow(full_matrix, cmap=plt.cm.RdBu_r, norm=norm)
        plt.colorbar(im)
        
        # Add population boundary lines
        current_idx = 0
        pop_ranges = {}
        for pop in self.populations:
            pop_nodes = [idx for idx, node in enumerate(self.nodes)
                        if self.G.nodes[node]['population'] == pop]
            pop_size = len(pop_nodes)
            pop_ranges[pop] = (current_idx, current_idx + pop_size)
            
            # Draw boundary lines
            if current_idx > 0:
                ax.axhline(y=current_idx - 0.5, color='black', linestyle='-', alpha=0.5)
                ax.axvline(x=current_idx - 0.5, color='black', linestyle='-', alpha=0.5)
            current_idx += pop_size
        
        # Add population labels
        for pop, (start, end) in pop_ranges.items():
            mid = (start + end) / 2
            ax.text(-0.1 * n_nodes, mid, pop, horizontalalignment='right', verticalalignment='center')
            ax.text(mid, -0.1 * n_nodes, pop, horizontalalignment='center', verticalalignment='top')
        
        ax.set_title('Full Coupling Matrix')
        ax.set_xlabel('Source Node')
        ax.set_ylabel('Target Node')
        
        ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def get_layout(self, layout_type='circular_populations'):
        """
        Get node positions using various layout algorithms
        
        Parameters:
        layout_type (str): Type of layout to use:
            - 'circular_populations': Circular layout with populations grouped
            - 'shell_populations': Shell layout with populations in different shells
            - 'bipartite': Bipartite layout for two populations
            - 'force_populations': Force-directed layout with population grouping
            - 'spectral_populations': Spectral layout with population grouping
        
        Returns:
        dict: Node positions
        """
        if layout_type == 'circular_populations':
            # Group nodes by population
            pop_nodes = defaultdict(list)
            for node in self.G.nodes():
                pop = self.G.nodes[node]['population']
                pop_nodes[pop].append(node)
            
            # Calculate positions for each population in a circle
            pos = {}
            n_pops = len(self.populations)
            for i, pop in enumerate(self.populations):
                nodes = pop_nodes[pop]
                n_nodes = len(nodes)
                
                # Calculate radius and center for this population's circle
                radius = 2.0
                theta = 2.0 * np.pi * i / n_pops
                center_x = 2.5 * radius * np.cos(theta)
                center_y = 2.5 * radius * np.sin(theta)
                
                # Position nodes in a circle around the population center
                for j, node in enumerate(nodes):
                    angle = 2.0 * np.pi * j / n_nodes
                    pos[node] = (
                        center_x + radius * np.cos(angle),
                        center_y + radius * np.sin(angle)
                    )
            
            return pos
            
        elif layout_type == 'shell_populations':
            # Create shells based on populations
            shells = [
                [node for node in self.G.nodes()
                 if self.G.nodes[node]['population'] == pop]
                for pop in self.populations
            ]
            return nx.shell_layout(self.G, shells)
            
        elif layout_type == 'bipartite':
            if len(self.populations) != 2:
                raise ValueError("Bipartite layout requires exactly two populations")
            
            # Separate nodes by population
            pop1, pop2 = self.populations
            nodes1 = [n for n in self.G.nodes() if self.G.nodes[n]['population'] == pop1]
            nodes2 = [n for n in self.G.nodes() if self.G.nodes[n]['population'] == pop2]
            
            pos = {}
            # Position first population on the left
            for i, node in enumerate(nodes1):
                pos[node] = (0, 2.0 * i / len(nodes1) - 1)
            # Position second population on the right
            for i, node in enumerate(nodes2):
                pos[node] = (2, 2.0 * i / len(nodes2) - 1)

            return pos
            
        elif layout_type == 'force_populations':
            # Use force-directed layout but add population-based positioning force
            pos = nx.spring_layout(self.G)
            
            # Adjust positions based on populations
            for node in self.G.nodes():
                pop = self.G.nodes[node]['population']
                pop_idx = self.populations.index(pop)
                angle = 2.0 * np.pi * pop_idx / len(self.populations)
                
                # Add slight pull toward population's preferred direction
                pos[node] = (
                    pos[node][0] + 0.2 * np.cos(angle),
                    pos[node][1] + 0.2 * np.sin(angle)
                )
                
            return pos
            
        elif layout_type == 'spectral_populations':
            # Use spectral layout but adjust based on populations
            pos = nx.spectral_layout(self.G)
            
            # Scale positions based on population
            for node in self.G.nodes():
                pop = self.G.nodes[node]['population']
                pop_idx = self.populations.index(pop)
                angle = 2.0 * np.pi * pop_idx / len(self.populations)
                
                # Rotate and scale based on population
                x, y = pos[node]
                pos[node] = (
                    x * np.cos(angle) - y * np.sin(angle),
                    x * np.sin(angle) + y * np.cos(angle)
                )
                
            return pos
        
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")

    def plot_network(self, layout_type='spectral_populations',
                     node_size=300, with_labels=True, figsize=(8, 8)):
        """
        Plot the network with specified layout
        
        Parameters:
        layout_type (str): Type of layout to use
        node_size (int): Size of nodes
        with_labels (bool): Whether to show node labels
        figsize (tuple): Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get layout
        pos = self.get_layout(layout_type)
        
        # Draw nodes for each population with different colors
        for pop in self.populations:
            nodes = [node for node in self.G.nodes()
                    if self.G.nodes[node]['population'] == pop]
            
            nx.draw_networkx_nodes(
                self.G, pos,
                nodelist=nodes,
                node_color=f'C{self.populations.index(pop)}',
                node_size=node_size,
                alpha=0.6,
                ax=ax
            )
        
        # Draw edges with weights determining width and color
        edges = self.G.edges(data=True)
        weights = [abs(e[2].get('weight', 1.0)) for e in edges]
        colors = ['red' if e[2].get('weight', 1.0) < 0 else 'blue' for e in edges]

        nx.draw_networkx_edges(
            self.G, pos,
            edge_color=colors,
            width=[w/max(weights) * 2 for w in weights],
            alpha=0.4,
            ax=ax
        )
        
        if with_labels:
            nx.draw_networkx_labels(self.G, pos)
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=f'C{i}', markersize=10,
                      label=pop)
            for i, pop in enumerate(self.populations)
        ]
        ax.legend(handles=legend_elements)
        
        plt.title(f'Network Structure ({layout_type})')
        
        return fig, ax

    def plot_spectrograms(self, t, solution, window_size=256, overlap=0.75, 
                         max_freq=None, figsize=(12, 8)):
        """
        Plot spectrograms of mean activities for each population
        
        Parameters:
        t (array): Time points
        solution (array): Solution array
        window_size (int): Size of the window for spectrogram
        overlap (float): Fraction of overlap between windows
        max_freq (float): Maximum frequency to plot (Hz)
        figsize (tuple): Figure size
        """
        from scipy import signal
        
        # Compute sampling rate from time array
        dt = t[1] - t[0]
        fs = 1/dt
        
        n_pops = len(self.populations)
        fig, axes = plt.subplots(n_pops, 2, figsize=figsize)
        
        # Process each population
        for i, pop in enumerate(self.populations):
            pop_indices = [
                idx for idx, node in enumerate(self.nodes)
                if self.G.nodes[node]['population'] == pop
            ]
            
            mean_activity = np.mean(solution[pop_indices], axis=0)
            
            # Compute spectrogram
            f, t_spec, Sxx = signal.spectrogram(
                mean_activity,
                fs=fs,
                window='hann',
                nperseg=window_size,
                noverlap=int(window_size * overlap),
                detrend='constant',
                scaling='density'
            )
            
            # Mean activity time series
            axes[i, 0].plot(t, mean_activity, label=f'Mean {pop} activity')
            axes[i, 0].set_xlabel('Time')
            axes[i, 0].set_ylabel('Activity')
            axes[i, 0].set_title(f'{pop} Population Mean Activity')
            axes[i, 0].grid(True)
            
            if max_freq is not None:
                freq_mask = f <= max_freq
                f = f[freq_mask]
                Sxx = Sxx[freq_mask]
            
            # log scale for spectral power
            im = axes[i, 1].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                                       shading='gouraud', cmap='jet')
            axes[i, 1].set_ylabel('Frequency [Hz]')
            axes[i, 1].set_xlabel('Time [s]')
            axes[i, 1].set_title(f'{pop} Population Spectrogram')
            
            plt.colorbar(im, ax=axes[i, 1], label='Power/Frequency [dB/Hz]')
        
        plt.tight_layout()
        return fig, axes

    def compute_nullclines(self, pop1='E', pop2='I', pop1_range=(0, 1), pop2_range=(0, 1), n_points=100, node_indices=None):
        """
        Compute nullclines for selected nodes or population means
        
        Parameters:
        pop1_range (tuple): Range for E values
        pop2_range (tuple): Range for I values
        n_points (int): Number of points for computation
        node_indices (tuple): Indices of E and I nodes to analyze, or None for mean-field
        
        Returns:
        tuple: Arrays for nullclines (pop1_points, pop1_nullcline, pop2_nullcline)
        """
        pop1_points = np.linspace(pop1_range[0], pop1_range[1], n_points)
        pop2_points = np.linspace(pop2_range[0], pop2_range[1], n_points)
        pop1_mesh, pop2_mesh = np.meshgrid(pop1_points, pop2_points)
        
        # Create state arrays for nullcline computation
        n_nodes = len(self.nodes)
        pop1_nullcline = np.zeros(n_points)
        pop2_nullcline = np.zeros(n_points)
        
        if node_indices is None:
            # mean-field nullclines
            for i, e in enumerate(pop1_points):
                # Create state with all E nodes at value e
                state = np.zeros(n_nodes)
                pop1_indices = [idx for idx, node in enumerate(self.nodes)
                                if self.G.nodes[node]['population'] == pop1]
                state[pop1_indices] = e
                
                # Find I value where dE/dt = 0
                pop2_indices = [idx for idx, node in enumerate(self.nodes)
                             if self.G.nodes[node]['population'] == pop2]
                for j, i_val in enumerate(pop2_points):
                    state[pop2_indices] = i_val
                    derivatives = self.dynamics(0, state)
                    if j > 0 and (derivatives[pop1_indices[0]] * prev_deriv) <= 0:
                        pop1_nullcline[i] = pop2_points[j]
                        break
                    prev_deriv = derivatives[pop1_indices[0]]
                
                # Find E value where dI/dt = 0
                state[pop2_indices] = pop2_points
                derivatives = self.dynamics(0, state)
                zero_crossings = np.where(np.diff(np.signbit(derivatives[pop2_indices])))[0]
                if len(zero_crossings) > 0:
                    pop2_nullcline[i] = pop2_points[zero_crossings[0]]

                    
        return pop1_points, pop1_nullcline, pop2_nullcline

    def compute_vector_field(self, E_range=(0, 1), I_range=(0, 1), n_points=20, node_indices=None):
        """
        Compute vector field for phase space
        
        Parameters:
        E_range, I_range (tuple): Ranges for E and I values
        n_points (int): Number of points in each dimension
        node_indices (tuple): Indices of E and I nodes to analyze, or None for mean-field
        
        Returns:
        tuple: Arrays for vector field (E_mesh, I_mesh, dE, dI)
        """
        E = np.linspace(E_range[0], E_range[1], n_points)
        I = np.linspace(I_range[0], I_range[1], n_points)
        E_mesh, I_mesh = np.meshgrid(E, I)
        
        dE = np.zeros_like(E_mesh)
        dI = np.zeros_like(I_mesh)
        
        # Get node indices for E and I populations
        E_indices = [idx for idx, node in enumerate(self.nodes)
                    if self.G.nodes[node]['population'] == 'E']
        I_indices = [idx for idx, node in enumerate(self.nodes)
                    if self.G.nodes[node]['population'] == 'I']
        
        for i in range(n_points):
            for j in range(n_points):
                state = np.zeros(len(self.nodes))
                state[E_indices] = E_mesh[i, j]
                state[I_indices] = I_mesh[i, j]
                
                derivatives = self.dynamics(0, state)
                dE[i, j] = np.mean(derivatives[E_indices])
                dI[i, j] = np.mean(derivatives[I_indices])
        
        return E_mesh, I_mesh, dE, dI

    def animate_with_phase_space(self, t, solution, interval=1.0, save_path=None,
                                 pop1='E', pop2='I', layout_type='circular_populations'):
        """
        Animate the simulation of two populations with phase space visualization
        """
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        
        # Network plot (too expensive to render)
        #ax1 = plt.subplot(gs[0])
        #pos = self.get_layout(layout_type)
        
        # Activity plot
        ax2 = plt.subplot(gs[0])
        
        # Phase space plot
        ax3 = plt.subplot(gs[1])
        
        # Compute mean activities for phase space
        pop1_indices = [idx for idx, node in enumerate(self.nodes)
                        if self.G.nodes[node]['population'] == pop1]
        pop2_indices = [idx for idx, node in enumerate(self.nodes)
                        if self.G.nodes[node]['population'] == pop2]
        
        pop1_mean = np.mean(solution[pop1_indices], axis=0)
        pop2_mean = np.mean(solution[pop2_indices], axis=0)
        
        # Compute nullclines and vector field
        pop1_range = (min(0, pop1_mean.min()), max(1, pop1_mean.max()))
        pop2_range = (min(0, pop2_mean.min()), max(1, pop2_mean.max()))
        
        pop1_points, pop1_null, pop2_null = self.compute_nullclines(pop1, pop2, pop1_range, pop2_range,
                                                                    n_points=len(pop1_indices))
        pop1_mesh, pop2_mesh, dE, dI = self.compute_vector_field(pop1_range, pop2_range, n_points=10)
        
        # Plot vector field
        skip = 2
        ax3.quiver(pop1_mesh[::skip, ::skip], pop2_mesh[::skip, ::skip],
                  dE[::skip, ::skip], dI[::skip, ::skip],
                  alpha=0.2, scale=50)

        # Plot nullclines
        ax3.plot(pop1_points, pop1_null, 'r--', label=f'{pop1}-nullcline', alpha=0.5)
        ax3.plot(pop1_points, pop2_null, 'b--', label=f'{pop2}-nullcline', alpha=0.5)
        
        # Initialize phase space trajectory
        phase_traj, = ax3.plot([], [], 'k-', alpha=0.5)
        phase_point, = ax3.plot([], [], 'ko', markersize=10)
        
        ax3.set_xlabel('Mean E Activity')
        ax3.set_ylabel('Mean I Activity')
        ax3.set_title('Phase Space')
        ax3.legend()
        ax3.grid(True)
        
        def draw_network(frame):
            ax1.clear()
            node_colors = []
            node_sizes = []
            for node in self.G.nodes():
                pop = self.G.nodes[node]['population']
                idx = self.node_to_idx[node]
                activity = solution[idx, frame]
                
                if pop == 'E':
                    color = plt.cm.Blues(activity)
                else:
                    color = plt.cm.Reds(activity)
                    
                node_colors.append(color)
                node_sizes.append(300 * (0.5 + activity))
            
            nx.draw(self.G, pos, ax=ax1, node_color=node_colors,
                   node_size=node_sizes, with_labels=True)
            ax1.set_title('Network State')
        
        # Initialize time series
        line_e, = ax2.plot([], [], 'r-', label=f'{pop1} mean')
        line_i, = ax2.plot([], [], 'b-', label=f'{pop2} mean')
        ax2.set_xlim(0, t[-1])
        ax2.set_ylim(min(solution.min(), 0), max(solution.max(), 1))
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Activity')
        ax2.legend()
        ax2.grid(True)
        
        def update(frame):
            # Update network plot
            #draw_network(frame)
            
            # Update time series
            line_e.set_data(t[:frame+1], pop1_mean[:frame+1])
            line_i.set_data(t[:frame+1], pop2_mean[:frame+1])
            
            # Update phase space trajectory
            phase_traj.set_data(pop1_mean[:frame+1], pop2_mean[:frame+1])
            phase_point.set_data([pop1_mean[frame]], [pop2_mean[frame]])
            
            return line_e, line_i, phase_traj, phase_point
        
        anim = FuncAnimation(
            fig, update, frames=len(t),
            interval=interval, blit=True
        )
        
        plt.tight_layout()
        
        if save_path:
            anim.save(save_path, writer='pillow')
            plt.close()
        else:
            plt.show()
        
        return anim


# Example usage
def create_test_graph(n_each=64):
    """Create a test network"""
    G = nx.DiGraph()
    
    # Add nodes for both populations in a ring topology
    for i in range(n_each):
        angle = 2 * np.pi * i / n_each
        # E population on outer ring
        G.add_node(f'E{i}', 
                  population='E',
                  pos=(1.2 * np.cos(angle), 1.2 * np.sin(angle)))
        # I population on inner ring
        G.add_node(f'I{i}', 
                  population='I',
                  pos=(np.cos(angle), np.sin(angle)))
    
    # Add connections
    for i in range(n_each):
        # E-E connections (recurrent excitation)
        G.add_edge(f'E{i}', f'E{(i+1)%n_each}', weight=10.0)
        #G.add_edge(f'E{i}', f'E{(i-1)%n_each}', weight=8.0)
        
        # I-I connections (recurrent inhibition)
        G.add_edge(f'I{i}', f'I{(i+1)%n_each}', weight=-1.0)
        G.add_edge(f'I{i}', f'I{(i-1)%n_each}', weight=-1.0)
        
        # E->I connections
        G.add_edge(f'E{i}', f'I{i}', weight=4.0)
        G.add_edge(f'E{i}', f'I{(i+1)%n_each}', weight=4.0)
        G.add_edge(f'E{i}', f'I{(i-1)%n_each}', weight=4.0)
        
        # I->E connections
        G.add_edge(f'I{i}', f'E{i}', weight=-11.0)
        G.add_edge(f'I{i}', f'E{(i+1)%n_each}', weight=-11.0)
        G.add_edge(f'I{i}', f'E{(i-1)%n_each}', weight=-11.0)
    
    return G



if __name__ == "__main__":
    # Create model
    n_each = 128
    G = create_test_graph(n_each=n_each)
    
    model = WilsonCowanGraph(G, dx=0.5,
                             diffusion_strength=1.0,
                             diffusion_populations=['E'],
                             spatial_kernel='gaussian',
                             spatial_scale=1.5,
                             tau={'E': 0.05, 'I': 0.03},
                             theta={'E': 4.0, 'I': 3.5},)
    
    # Add localized external input
    center_node = n_each // 2
    input_dict = {}
    for i in range(n_each):
        dist = min(abs(i - center_node), abs(i - (center_node + n_each)))  # Account for periodic boundary
        input_dict[f'E{i}'] = 2.0 * np.exp(-dist/3)
    
    model.set_external_input(input_dict)

    test_state = 0.5 * np.random.rand(model.n_nodes)
    model.verify_jacobian(0.0, test_state)
        
    model.plot_network(layout_type='circular_populations')
    plt.show()
    
    # Plot coupling matrices
    model.plot_coupling_matrices()
    model.plot_full_coupling_matrix()
    plt.show()

    print("Analyzing stability...")
    fig, axes, stability = model.plot_eigenvalue_spectrum()
    plt.show()
    
    print(f"System is {'stable' if stability['is_stable'] else 'unstable'}")
    print(f"Maximum real part: {stability['max_real_part']:.6f}")
    print(f"Oscillatory frequencies: {stability['oscillatory_frequencies']}")

    # Simulate
    start_time = time.time()
    T = 60.0
    dt = 0.001
    t, solution = model.simulate(T, dt, method='BDF')
    run_time = time.time() - start_time
    print(f"Simulation completed in {run_time:.3f} s")
    
    model.plot_state(t, solution, time=T)
    plt.show()
    
    model.plot_spectrograms(t, solution, 
                            window_size=1024,   # Smaller window for better time resolution
                            overlap=0.9,     # 75% overlap between windows
                            max_freq=100)     # Only show frequencies up to 100 Hz
    plt.show()

    # Animate
    #model.animate_with_phase_space(t, solution,
    #                               layout_type='circular_populations')
