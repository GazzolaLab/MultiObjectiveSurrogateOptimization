import numpy as np
from scipy.integrate import solve_ivp
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Arrow

class WilsonCowanDelay1D:
    def __init__(self, size=100, dx=1.0, tau_e=1.0, tau_i=1.0, delay_e=None, delay_i=None):
        """
        Initialize 1D Wilson-Cowan model
        
        Parameters:
        size (int): Number of spatial points
        dx (float): Spatial step size
        tau_e (float): Time constant for excitatory population
        tau_i (float): Time constant for inhibitory population
        delay_e (float): Delay in excitatory connections
        delay_i (float): Delay in inhibitory connections
        """

        self.size = size
        self.dx = dx
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.delay_e = delay_e
        self.delay_i = delay_i
        
        # Coupling parameters
        self.wee = 15.0  # E->E coupling
        self.wei = 10.0  # E->I coupling
        self.wie = 10.0  # I->E coupling
        self.wii = 2.0   # I->I coupling
        
        # Threshold parameters
        self.theta_e = 4.0
        self.theta_i = 3.3
        
        # Initialize spatial grid
        self.x = np.linspace(0, size*dx, size)
        
        # External input
        self.ext_input_e = np.zeros(size)
        self.ext_input_i = np.zeros(size)

        # History buffers for delayed terms
        self.history_length = int(max(delay_e if delay_e is not None else 0,
                                      delay_i if delay_i is not None else 0) * 1000)  # Store enough history
        self.e_history = deque(maxlen=self.history_length)
        self.i_history = deque(maxlen=self.history_length)
        self.t_history = deque(maxlen=self.history_length)
    
    def sigmoid(self, x, theta):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x + theta))
    
    def spatial_coupling(self, activity):
        """Compute spatial coupling using discrete Laplacian"""
        laplacian = np.zeros_like(activity)
        laplacian[1:-1] = (
            activity[:-2] + 
            activity[2:] - 
            2 * activity[1:-1]
        ) / (self.dx ** 2)
        return laplacian
    
    def get_delayed_activity(self, t, delay, history_buffer, t_history):
        """
        Get delayed activity by interpolating from history
        
        Parameters:
        t (float): Current time
        delay (float): Delay amount
        history_buffer (deque): Buffer of past activities
        t_history (deque): Buffer of past times
        
        Returns:
        array: Delayed activity
        """
        if len(history_buffer) < 2:
            return np.zeros(self.size)
            
        t_delayed = t - delay
        if t_delayed <= min(t_history):
            return np.zeros(self.size)
            
        # Find indices for interpolation
        idx = 0
        while idx < len(t_history) - 1 and t_history[idx + 1] <= t_delayed:
            idx += 1
            
        if idx >= len(t_history) - 1:
            return np.array(history_buffer[-1])
            
        # Linear interpolation
        t0, t1 = t_history[idx], t_history[idx + 1]
        y0, y1 = np.array(history_buffer[idx]), np.array(history_buffer[idx + 1])
        alpha = (t_delayed - t0) / (t1 - t0)
        
        return y0 + alpha * (y1 - y0)
    
    def dynamics(self, t, state):
        """
        Compute the right-hand side of the Wilson-Cowan equations with delays
        
        Parameters:
        t (float): Time
        state (array): Current state (E and I activities flattened)
        
        Returns:
        array: Time derivatives of state variables
        """
        # Split state into E and I components
        E = state[:self.size]
        I = state[self.size:]

        # Get delayed activities
        if self.delay_e is None:
            E_delayed = E
        else:
            E_delayed = self.get_delayed_activity(t, self.delay_e, self.e_history, self.t_history)
        if self.delay_i is None:
            I_delayed = I
        else:
            I_delayed = self.get_delayed_activity(t, self.delay_i, self.i_history, self.t_history)
        
        # Compute spatial coupling
        E_coupling = self.spatial_coupling(E)
        I_coupling = self.spatial_coupling(I)
        
        # Compute input to each population using delayed activities
        input_e = self.wee * E_delayed - self.wie * I_delayed + E_coupling + self.ext_input_e
        input_i = self.wei * E_delayed - self.wii * I_delayed + I_coupling + self.ext_input_i
        
        # Compute derivatives
        dE = (-E + self.sigmoid(input_e, self.theta_e)) / self.tau_e
        dI = (-I + self.sigmoid(input_i, self.theta_i)) / self.tau_i
        
        return np.concatenate([dE, dI])
    
    def simulate(self, T, dt, initial_conditions=None):
        """
        Simulate the model
        
        Parameters:
        T (float): Total simulation time
        dt (float): Time step
        initial_conditions (tuple): Initial E and I activities (optional)
        
        Returns:
        tuple: Time points and solution arrays
        """
        t_span = (0, T)
        t_eval = np.arange(0, T, dt)
        
        if initial_conditions is None:
            # Default: small random initial conditions
            E0 = 0.1 * np.random.rand(self.size)
            I0 = 0.1 * np.random.rand(self.size)
        else:
            E0, I0 = initial_conditions
        
        initial_state = np.concatenate([E0, I0])
        
        # Solve the system
        solution = solve_ivp(
            self.dynamics,
            t_span,
            initial_state,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-6
        )
        
        # Split solution into E and I components
        E = solution.y[:self.size]
        I = solution.y[self.size:]
        
        return solution.t, E, I
    
    def plot_state(self, E, I, time=None):
        """Plot current state of the system"""
        plt.figure(figsize=(10, 4))
        plt.plot(self.x, E, 'r-', label='Excitatory')
        plt.plot(self.x, I, 'b-', label='Inhibitory')
        plt.xlabel('Space')
        plt.ylabel('Activity')
        plt.legend()
        if time is not None:
            plt.title(f't = {time:.2f}')
        plt.grid(True)
        plt.show()
    
    def animate_simulation(self, t, E, I, interval=50, save_path=None):
        """
        Animate the simulation results
        
        Parameters:
        t (array): Time points
        E (array): Excitatory population activity history
        I (array): Inhibitory population activity history
        interval (int): Interval between frames in milliseconds
        save_path (str): Path to save the animation (optional)
        """
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        
        # Create subplots
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        
        # Initialize spatial plot
        line_e, = ax1.plot(self.x, E[:, 0], 'r-', label='Excitatory')
        line_i, = ax1.plot(self.x, I[:, 0], 'b-', label='Inhibitory')
        ax1.set_xlabel('Space')
        ax1.set_ylabel('Activity')
        ax1.set_ylim(min(E.min(), I.min())-0.1, max(E.max(), I.max())+0.1)
        ax1.legend()
        ax1.grid(True)
        
        # Initialize time series plot
        line_e_mean, = ax2.plot(t[0:1], [np.mean(E[:, 0])], 'r-', label='E mean')
        line_i_mean, = ax2.plot(t[0:1], [np.mean(I[:, 0])], 'b-', label='I mean')
        ax2.set_xlim(0, t[-1])
        ax2.set_ylim(0, max(np.mean(E, axis=0).max(), np.mean(I, axis=0).max()) * 1.1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Mean Activity')
        ax2.legend()
        ax2.grid(True)
        
        # Time indicator
        time_text = ax1.text(0.02, 1.02, '', transform=ax1.transAxes)
        
        def update(frame):
            # Update spatial plot
            line_e.set_ydata(E[:, frame])
            line_i.set_ydata(I[:, frame])
            
            # Update time series
            line_e_mean.set_data(t[:frame+1], np.mean(E[:, :frame+1], axis=0))
            line_i_mean.set_data(t[:frame+1], np.mean(I[:, :frame+1], axis=0))
            
            # Update time indicator
            time_text.set_text(f't = {t[frame]:.2f}')
            
            return line_e, line_i, line_e_mean, line_i_mean, time_text
        
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


    def compute_nullclines(self, x_range=(0, 1), y_range=(0, 1), n_points=100, position=None):
        """
        Compute E and I nullclines for phase space analysis
        
        Parameters:
        x_range (tuple): Range for E values
        y_range (tuple): Range for I values
        n_points (int): Number of points for computation
        position (int): Spatial position to compute nullclines for (None for mean-field)
        
        Returns:
        tuple: Arrays for nullclines (E_points, E_nullcline, I_nullcline)
        """
        E_points = np.linspace(x_range[0], x_range[1], n_points)
        
        # Get external input at specified position or mean
        if position is not None:
            ext_e = self.ext_input_e[position]
            ext_i = self.ext_input_i[position]
        else:
            ext_e = np.mean(self.ext_input_e)
            ext_i = np.mean(self.ext_input_i)
        
        # E-nullcline: dE/dt = 0
        E_nullcline = self.sigmoid(self.wee * E_points - self.wei * E_points + ext_e, self.theta_e)
        
        # I-nullcline: dI/dt = 0
        I_nullcline = self.sigmoid(self.wie * E_points - self.wii * E_points + ext_i, self.theta_i)
        
        return E_points, E_nullcline, I_nullcline

    def compute_vector_field(self, E_range=(0, 1), I_range=(0, 1), n_points=20, position=None):
        """
        Compute vector field for phase space analysis
        
        Parameters:
        E_range (tuple): Range for E values
        I_range (tuple): Range for I values
        n_points (int): Number of points in each dimension
        position (int): Spatial position to compute vector field for (None for mean-field)
        
        Returns:
        tuple: Arrays for vector field (E_mesh, I_mesh, dE, dI)
        """
        E = np.linspace(E_range[0], E_range[1], n_points)
        I = np.linspace(I_range[0], I_range[1], n_points)
        E_mesh, I_mesh = np.meshgrid(E, I)
        
        # Get external input at specified position or mean
        if position is not None:
            ext_e = self.ext_input_e[position]
            ext_i = self.ext_input_i[position]
        else:
            ext_e = np.mean(self.ext_input_e)
            ext_i = np.mean(self.ext_input_i)
        
        # Compute derivatives
        dE = (-E_mesh + self.sigmoid(self.wee * E_mesh - self.wei * I_mesh + ext_e, self.theta_e)) / self.tau_e
        dI = (-I_mesh + self.sigmoid(self.wie * E_mesh - self.wii * I_mesh + ext_i, self.theta_i)) / self.tau_i
        
        return E_mesh, I_mesh, dE, dI

    def animate_with_phase_space(self, t, E, I, interval=50, save_path=None, track_points=None):
        """
        Animate simulation with phase space visualization
        
        Parameters:
        t (array): Time points
        E (array): Excitatory population activity history
        I (array): Inhibitory population activity history
        interval (int): Interval between frames in milliseconds
        save_path (str): Path to save animation
        track_points (list): List of spatial positions to track in phase space
        """
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        
        # Spatial dynamics plot
        ax1 = plt.subplot(gs[0, 0])
        line_e, = ax1.plot(self.x, E[:, 0], 'r-', label='Excitatory')
        line_i, = ax1.plot(self.x, I[:, 0], 'b-', label='Inhibitory')
        ax1.set_xlabel('Space')
        ax1.set_ylabel('Activity')
        ax1.set_ylim(min(E.min(), I.min())-0.1, max(E.max(), I.max())+0.1)
        ax1.legend()
        ax1.grid(True)
        
        # Time series plot
        ax2 = plt.subplot(gs[1, 0])
        line_e_mean, = ax2.plot(t[0:1], [np.mean(E[:, 0])], 'r-', label='E mean')
        line_i_mean, = ax2.plot(t[0:1], [np.mean(I[:, 0])], 'b-', label='I mean')
        ax2.set_xlim(0, t[-1])
        ax2.set_ylim(0, max(np.mean(E, axis=0).max(), np.mean(I, axis=0).max()) * 1.1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Mean Activity')
        ax2.legend()
        ax2.grid(True)
        
        # Phase space plot
        ax3 = plt.subplot(gs[:, 1])
        
        # Compute and plot nullclines
        E_range = (0, max(E.max(), 1))
        I_range = (0, max(I.max(), 1))
        E_points, E_null, I_null = self.compute_nullclines(E_range, I_range)
        ax3.plot(E_points, E_null, 'r--', label='E-nullcline', alpha=0.5)
        ax3.plot(E_points, I_null, 'b--', label='I-nullcline', alpha=0.5)
        
        # Compute and plot vector field
        E_mesh, I_mesh, dE, dI = self.compute_vector_field(E_range, I_range)
        skip = 2
        ax3.quiver(E_mesh[::skip, ::skip], I_mesh[::skip, ::skip],
                  dE[::skip, ::skip], dI[::skip, ::skip],
                  alpha=0.2, scale=50)
        
        # Initialize trajectory plots
        if track_points is None:
            track_points = [self.size // 2]  # Default: track center point
            
        trajectories = []
        for _ in track_points:
            line, = ax3.plot([], [], 'k-', alpha=0.5)
            point, = ax3.plot([], [], 'ko')
            trajectories.append((line, point))
        
        # Mean trajectory
        mean_traj, = ax3.plot([], [], 'g-', alpha=0.7, label='Mean trajectory')
        mean_point, = ax3.plot([], [], 'go')
        
        ax3.set_xlabel('E activity')
        ax3.set_ylabel('I activity')
        ax3.set_title('Phase Space')
        ax3.legend()
        ax3.grid(True)
        
        # Time indicator
        time_text = ax1.text(0.02, 1.02, '', transform=ax1.transAxes)
        
        def update(frame):
            # Update spatial plot
            line_e.set_ydata(E[:, frame])
            line_i.set_ydata(I[:, frame])
            
            # Update time series
            line_e_mean.set_data(t[:frame+1], np.mean(E[:, :frame+1], axis=0))
            line_i_mean.set_data(t[:frame+1], np.mean(I[:, :frame+1], axis=0))
            
            # Update phase space trajectories
            for i, point_idx in enumerate(track_points):
                line, point = trajectories[i]
                line.set_data(E[point_idx, :frame+1], I[point_idx, :frame+1])
                point.set_data([E[point_idx, frame]], [I[point_idx, frame]])
            
            # Update mean trajectory
            mean_traj.set_data(np.mean(E[:, :frame+1], axis=0),
                               np.mean(I[:, :frame+1], axis=0))
            mean_point.set_data([np.mean(E[:, frame])], [np.mean(I[:, frame])])
            
            # Update time indicator
            time_text.set_text(f't = {t[frame]:.2f}')
            
            return (line_e, line_i, line_e_mean, line_i_mean, time_text,
                   mean_traj, mean_point, *[item for traj in trajectories for item in traj])
        
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
if __name__ == "__main__":
    # Initialize model with delays
    model = WilsonCowanDelay1D(size=100, delay_e=None, delay_i=None)

    # Add a localized stimulus to excitatory population
    x_c = 50  # Center of stimulus
    width = 5
    model.ext_input_e = 2.0 * np.exp(-(model.x - x_c)**2 / (2*width**2))
    
    T = 50.0
    dt = 0.1
    t, E, I = model.simulate(T, dt)
    
    model.plot_state(E[:,-1], I[:,-1], time=T)
    
    # Create animation with phase space
    track_points = [25, 50, 75]  # Track multiple spatial points
    model.animate_with_phase_space(t, E, I, interval=50, track_points=track_points)
