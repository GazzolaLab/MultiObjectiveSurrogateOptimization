import logging
import numpy as np
from math import cos, pi
from dmosopt import dmosopt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Numerical Integration of a 4D wilson-cowan system, simulating dentate gyrus population rate activity
# Code based on https://github.com/selenasingh/DG-Oscillations

oscillatory_frequencies_cos_scale = {
    'theta' : 0.02, # 3 Hz
    'alpha' : 0.08, # 12 Hz
    'gamma' : 0.2, # 35 Hz
    'delta' : 0.005,
    'constant' : 0 # constant of 1
}
    

class DGRate(object):
    def __init__(self,
                 PP_freq,       # frequency of PP inputs
                 fbi,           # scale feedback and feedforward inhibition synaptic weights (between GCs and BCs)
                 PP_weight,     # scale PP synaptic weight (onto GCs and BCs)
                 **kwargs
                ):
        
        self.pars = self.default_parameters
        self.pars['input_freq'] = PP_freq
        self.pars['wPPg'] = PP_weight
        self.pars['wgb'] = fbi
        self.pars['wbg'] = fbi
        self.pars.update(kwargs)
        
        self.flag = f"{PP_freq}_{fbi}_{PP_weight}"

    def parameters(self, **kwargs):

        self.pars.update(kwargs)
        
        # Vector of time points [ms]
        self.pars['range_t'] = np.arange(0, self.pars['T'], self.pars['dt'])

        # PP input
        cos_scale = oscillatory_frequencies_cos_scale[self.pars['input_freq']]

        periodic_forcing = []
        for x in self.pars['range_t']:
            periodic_forcing.append((1 + cos(cos_scale * x))/2) #remove 2, get bifc'n 

        self.pars['PP'] = periodic_forcing

        return self.pars
    
    @property
    def default_parameters(self):
        """Returns default parameters of DG model."""
        params = {}

        # gc parameters
        params['tau_g'] = 3.1  # membrane timescale of granule cell [ms]
        params['gain_g'] = 60  # gain of granule cell 
        params['thresh_g'] = 0.055  # threshold of granule cell

        # bc parameters
        params['tau_b'] = 1.0  # membrane timescale of basket cell [ms]
        params['gain_b'] = 250  # gain of basket cell
        params['thresh_b'] = 0.025  # threshold of basket cell

        # mc parameters
        params['tau_m'] = 3.5  # membrane timescale of mossy cell [ms]
        params['gain_m'] = 25  # gain of mossy cell
        params['thresh_m'] = 0.005  # threshold of mossy cell

        # hc parameters
        params['tau_h'] = 1.5  # membrane timescale of hipp cell [ms]
        params['gain_h'] = 20  # gain of hipp cell
        params['thresh_h'] = 0  # threshold of hipp cell

        # synaptic weights
        params['wgg'] = 0.  # GC to GC   ; mossy fiber "sprouting"
        params['wmg'] = 1.  # MC to GC
        params['wbg'] = 3  # BC to GC ; 1 for lesion study
        params['whg'] = 1.  # HC to GC
        params['wbb'] = 1.  # BC to BC
        params['wgb'] = 3.  # GC to BC ; 0 for lesion study
        params['wmb'] = 1.  # MC to BC
        params['whb'] = 1.  # HC to BC
        params['wmm'] = 1.  # MC to MC
        params['wgm'] = 1.  # GC to MC
        params['wbm'] = 1.  # BC to MC
        params['whm'] = 1.  # HC to MC
        params['wmh'] = 1.  # MC to HC
        params['wgh'] = 1.  # GC to HC

        params['wPPg'] = 1  # scale PP synaptic weight to gcs 
        params['wPPb'] = params['wPPg']/2  # scale PP synaptic input to bcs

        # integration parameters
        params['T'] = 1000.  # Total duration of simulation [ms]
        params['dt'] = .001  # Simulation time step [ms]
        params['g_init'] = 0.001  # Initial value of granule cells
        params['b_init'] = 0.001  # Initial value of basket cells
        params['m_init'] = 0.001  # Initial value of mossy cells
        params['h_init'] = 0.001  # Initial value of hipp cells

        return params

    def F(self, i, gain, thresh):
        """
        Population activation function, F-I curve

        Args:
          i     : the population input
          gain  : the gain of the function
          thresh : the threshold of the function

        Returns:
          f     : the population activation response f(x) for input x
        """

        # add the expression of f = F(x)
        f = (1 + np.exp(-gain * (i - thresh))) ** -1 - (1 + np.exp(gain * thresh)) ** -1

        return f

    def simulate_DG(self,
                    # gc
                    tau_g, gain_g, thresh_g,
                    # bc
                    tau_b, gain_b, thresh_b,
                    # mc
                    tau_m, gain_m, thresh_m,
                    # hc
                    tau_h, gain_h, thresh_h,

                    # synaptic weights
                    wgg, wmg, wbg, whg, wbb, wgb, wmb, whb, wmm, wgm, wbm, whm, wmh, wgh,

                    # perforant path
                    wPPg, wPPb, PP,

                    # simulation params
                    range_t, dt, g_init, b_init, m_init, h_init,

                    **other_pars):
        """
            Simulate two sets of Wilson-Cowan equations, modelling dentate gyrus population activity

            Args:
              Parameters of the 4D system

            Returns:
              g, b, m, h (arrays) : Activity of granule, basket, mossy and hipp cells.
            """
        # Initialize activity arrays
        Lt = range_t.size
        g = np.append(g_init, np.zeros(Lt - 1))
        b = np.append(b_init, np.zeros(Lt - 1))
        m = np.append(m_init, np.zeros(Lt - 1))
        h = np.append(h_init, np.zeros(Lt - 1))

        # Simulate the 4D system
        for k in range(Lt - 1):
            # Calculate the derivative of the granule cell population
            dg = dt / tau_g * (-g[k] + self.F(wgg * g[k] + wmg * m[k] - wbg * b[k] - whg * h[k] + wPPg * PP[k],
                                              gain_g, thresh_g))

            # Calculate the derivative of basket cell population
            db = dt / tau_b * (-b[k] + self.F(-wbb * b[k] + wgb * g[k] + wmb * m[k] - whb * h[k] + wPPb * PP[k],
                                              gain_b, thresh_b))

            # Calculate the derivative of the mossy cell population
            dm = dt / tau_m * (-m[k] + self.F(wmm * m[k] + wgm * g[k] - wbm * b[k] - whm * h[k],
                                              gain_m, thresh_m))

            # Calculate the derivative of the hipp cell population
            dh = dt / tau_h * (-h[k] + self.F(wmh * m[k] + wgh * g[k],
                                              gain_h, thresh_h))

            # Update using Euler's method
            g[k + 1] = g[k] + dg
            b[k + 1] = b[k] + db
            m[k + 1] = m[k] + dm
            h[k + 1] = h[k] + dh

        return g, b, m, h

    def run(self, **kwargs):
        params = self.parameters(**kwargs)
        g, b, m, h = self.simulate_DG(**params)
        return { 'g': g, 'b': b, 'm': m, 'h': h }


feature_dtypes = [
    (
        "mean_g_rate",
        np.float32,
    ),
    (
        "mean_b_rate",
        np.float32,
    ),
    (
        "mean_m_rate",
        np.float32,
    ),
    (
        "mean_h_rate",
        np.float32,
    ),
]

objective_names = ["g", "b", "m", "h"]


objective_targets = {
    "g": 0.01,
    "b": 0.2,
    "h": 0.1,
    "m": 0.05,
} # mean firing rate targets

constraint_names = ["pos_g", "pos_b", "pos_m", "pos_h"]

def obj_fun(pp):
    """Objective function to be minimized."""
    
    network_model = DGRate(PP_freq='theta', **pp)
    output = network_model.run()

    logger.info(f"DGRate: \t pp:{pp}, output:{output}")
    res = np.asarray([ np.abs(np.mean(output[k]) - objective_targets[k]) for k in objective_names ],
                     dtype=np.float32)

    feature_values = np.asarray(
        [
            (
                np.mean(output["g"]),
                np.mean(output["b"]),
                np.mean(output["m"]),
                np.mean(output["h"]),
            )
        ],
        dtype=np.dtype(feature_dtypes),
    )
    
    
    return res, feature_values


def optimize():

    space = {
        'wmg':  (0.1, 2.),  # MC to GC
        'wbg':  (3., 5.),  # BC to GC
        'whg':  (1., 5.), # HC to GC
        'wbb':  (0.1, 1.),  # BC to BC
        'wgb':  (2., 5.),  # GC to BC
        'wmb':  (1., 5.), # MC to BC
        'whb':  (0.1, 1.),  # HC to BC
        'wmm':  (0.1, 1.),  # MC to MC
        'wgm':  (0.1, 1.),  # GC to MC
        'wbm':  (1., 5.),  # BC to MC
        'whm':  (1., 5.),  # HC to MC
        'wmh':  (1., 2.), # MC to HC
        'wgh':  (1., 2.), # GC to HC
    }
    
    problem_parameters = {'fbi': 1.65,
                          'PP_weight': 1.0, }

    # Create an optimizer
    dmosopt_params = {
        "opt_id": "dmosopt_DGRate",
        "obj_fun_name": "DG_rate.obj_fun",
        "problem_parameters": problem_parameters,
        "space": space,
        "objective_names": objective_names,
#        "constraint_names": constraint_names,
        "feature_dtypes": feature_dtypes,
        "population_size": 400,
        "num_generations": 400,
        "initial_maxiter": 10,
        "surrogate_method_name": "gpr",
        "optimizer_name": "nsga2",
        "optimizer_kwargs": [
            {
                "crossover_prob": 0.9,
                "mutation_prob": 0.1,
            },
            {},
        ],
        "termination_conditions": True,
        "n_initial": 3,
        "n_epochs": 5,
        "save_surrogate_eval": True,
        "save": True,
        "file_path": "results/DG_rate.h5",
    }

    best = dmosopt.run(dmosopt_params, verbose=True)
    return best


    

def plot_DG_rates(network_model, results):
    import matplotlib.pyplot as plt

    params = network_model.pars
    
    g, b, m, h = ((results[k] for k in ["g", "b", "m", "h"]))

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(4,5))
    
    ax1.plot(params['range_t'], h, color='0.5', label='HIPP')
    ax1.set_ylabel("HIPP")
    
    ax2.plot(params['range_t'], b, color='0.5', label='BC')
    ax2.set_ylabel("BC")
    
    ax3.plot(params['range_t'], m, color='0.5', label='MC')
    ax3.set_ylabel("MC")
    
    ax4.plot(params['range_t'], g, color='0.5', label='GC')
    ax4.set_ylabel("GC")
    
    ax5.plot(params['range_t'], params['PP'], color='0.5', label='PP')
    ax5.set_ylabel("PP")
    ax5.set_xlabel("Time (ms)")

    fig.tight_layout()
    fig.align_ylabels()
    fig.savefig('figures/DG_population_rates_%s.pdf' % network_model.flag, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":

    best = optimize()
    if best is not None:
        
        bestx, besty = best
        bestx_dict = dict(bestx)

        network_model = DGRate(PP_freq='theta', fbi=1.65, PP_weight=1.0, **bestx_dict)
        output = network_model.run()
        plot_DG_rates(network_model, output)

    

