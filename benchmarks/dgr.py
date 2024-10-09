import logging
import numpy as np
from math import cos
from scipy.integrate import solve_ivp
from scipy import interpolate, signal
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Numerical Integration of a 4D wilson-cowan system, simulating dentate gyrus population rate activity
# Code based on https://github.com/selenasingh/DG-Oscillations

oscillatory_frequencies_cos_scale = {
    "theta": 0.02,  # 3 Hz
    "alpha": 0.08,  # 12 Hz
    "gamma": 0.2,  # 35 Hz
    "delta": 0.005,
    "constant": 0,  # constant of 1
}


class DGRate(object):
    def __init__(
        self,
        PP_freq,  # frequency of PP inputs
        fbi,  # scale feedback and feedforward inhibition synaptic weights (between GCs and BCs)
        PP_weight,  # scale PP synaptic weight (onto GCs and BCs)
        **kwargs,
    ):
        self.pars = self.default_parameters
        self.pars["input_freq"] = PP_freq
        self.pars["wPPg"] = PP_weight
        self.pars["wgb"] = fbi
        self.pars["wbg"] = fbi
        self.pars.update(kwargs)

        self.flag = f"{PP_freq}_{fbi}_{PP_weight}"

        # solver
        self.simulate_DG = self.simulate_DG_ivp

    def parameters(self, **kwargs):
        self.pars.update(kwargs)

        # Vector of time points [ms]
        self.pars["range_t"] = np.arange(0, self.pars["T"], self.pars["dt"])

        # PP input
        cos_scale = oscillatory_frequencies_cos_scale[self.pars["input_freq"]]

        periodic_forcing = []
        for x in self.pars["range_t"]:
            periodic_forcing.append(
                (1 + cos(cos_scale * x)) / 2
            )  # remove 2, get bifc'n

        self.pars["PP"] = periodic_forcing
        self.pars["PP_interp"] = interpolate.Akima1DInterpolator(
            self.pars["range_t"], periodic_forcing
        )

        return self.pars

    @property
    def default_parameters(self):
        """Returns default parameters of DG model."""
        params = {}

        # gc parameters
        params["tau_g"] = 3.1  # membrane timescale of granule cell [ms]
        params["gain_g"] = 60  # gain of granule cell
        params["thresh_g"] = 0.055  # threshold of granule cell

        # bc parameters
        params["tau_b"] = 1.0  # membrane timescale of basket cell [ms]
        params["gain_b"] = 250  # gain of basket cell
        params["thresh_b"] = 0.025  # threshold of basket cell

        # mc parameters
        params["tau_m"] = 3.5  # membrane timescale of mossy cell [ms]
        params["gain_m"] = 25  # gain of mossy cell
        params["thresh_m"] = 0.01  # threshold of mossy cell

        # hc parameters
        params["tau_h"] = 1.5  # membrane timescale of hipp cell [ms]
        params["gain_h"] = 20  # gain of hipp cell
        params["thresh_h"] = 0.0  # threshold of hipp cell

        # synaptic weights
        params["wgg"] = 0.0  # GC to GC   ; mossy fiber "sprouting"
        params["wmg"] = 2.0  # MC to GC
        params["wbg"] = 1.0  # BC to GC ; 1 for lesion study
        params["whg"] = 1.0  # HC to GC
        params["wbb"] = 1.0  # BC to BC
        params["wgb"] = 3.0  # GC to BC ; 0 for lesion study
        params["wmb"] = 2.0  # MC to BC
        params["whb"] = 1.0  # HC to BC
        params["wmm"] = 1.0  # MC to MC
        params["wgm"] = 1.0  # GC to MC
        params["wbm"] = 4.0  # BC to MC
        params["whm"] = 3.0  # HC to MC
        params["wmh"] = 1.0  # MC to HC
        params["wgh"] = 1.0  # GC to HC
        params["whh"] = 3.0  # HC to HC

        params["wPPg"] = 1  # scale PP synaptic weight to gcs
        params["wPPb"] = params["wPPg"] / 2  # scale PP synaptic input to bcs

        # integration parameters
        params["T"] = 1000.0  # Total duration of simulation [ms]
        params["dt"] = 0.001  # Simulation time step [ms]
        params["g_init"] = 0.001  # Initial value of granule cells
        params["b_init"] = 0.001  # Initial value of basket cells
        params["m_init"] = 0.001  # Initial value of mossy cells
        params["h_init"] = 0.001  # Initial value of hipp cells

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

        #    f = 1 / ((1 + np.exp(-gain * (i - thresh))) - (1 + np.exp(gain * thresh)))

        f = 1.0 / (1.0 + np.exp(-gain * (i - thresh)))

        return f

    def simulate_DG_euler(
        self,
        # gc
        tau_g,
        gain_g,
        thresh_g,
        # bc
        tau_b,
        gain_b,
        thresh_b,
        # mc
        tau_m,
        gain_m,
        thresh_m,
        # hc
        tau_h,
        gain_h,
        thresh_h,
        # synaptic weights
        wgg,
        wmg,
        wbg,
        whg,
        wbb,
        wgb,
        wmb,
        whb,
        wmm,
        wgm,
        wbm,
        whm,
        wmh,
        wgh,
        whh,
        # perforant path
        wPPg,
        wPPb,
        PP,
        # simulation params
        range_t,
        dt,
        g_init,
        b_init,
        m_init,
        h_init,
        **other_pars,
    ):
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
            dg = (
                dt
                / tau_g
                * (
                    -g[k]
                    + self.F(
                        wgg * g[k]
                        + wmg * m[k]
                        - wbg * b[k]
                        - whg * h[k]
                        + wPPg * PP[k],
                        gain_g,
                        thresh_g,
                    )
                )
            )

            # Calculate the derivative of basket cell population
            db = (
                dt
                / tau_b
                * (
                    -b[k]
                    + self.F(
                        -wbb * b[k]
                        + wgb * g[k]
                        + wmb * m[k]
                        - whb * h[k]
                        + wPPb * PP[k],
                        gain_b,
                        thresh_b,
                    )
                )
            )

            # Calculate the derivative of the mossy cell population
            dm = (
                dt
                / tau_m
                * (
                    -m[k]
                    + self.F(
                        wmm * m[k] + wgm * g[k] - wbm * b[k] - whm * h[k],
                        gain_m,
                        thresh_m,
                    )
                )
            )

            # Calculate the derivative of the hipp cell population
            dh = (
                dt
                / tau_h
                * (
                    -h[k]
                    + self.F(wmh * m[k] + wgh * g[k] - whh * h[k], gain_h, thresh_h)
                )
            )

            # Update using Euler's method
            g[k + 1] = g[k] + dg
            b[k + 1] = b[k] + db
            m[k + 1] = m[k] + dm
            h[k + 1] = h[k] + dh

        return g, b, m, h

    def system_dynamics(
        self,
        t,
        y,
        tau_g,
        gain_g,
        thresh_g,
        tau_b,
        gain_b,
        thresh_b,
        tau_m,
        gain_m,
        thresh_m,
        tau_h,
        gain_h,
        thresh_h,
        wgg,
        wmg,
        wbg,
        whg,
        wbb,
        wgb,
        wmb,
        whb,
        wmm,
        wgm,
        wbm,
        whm,
        wmh,
        wgh,
        whh,
        wPPg,
        wPPb,
        PP,
    ):
        """
        Defines the system of differential equations
        """
        g, b, m, h = y
        PP_interp = self.pars["PP_interp"]

        # logger.info(f"at time {t}: PP_interp(t): {PP_interp(t)} initial g: {g} b: {b} m: {m} h: {h}")
        # logger.info(f"at time {t}: F(g): {self.F(wgg * g + wmg * m - wbg * b - whg * h + wPPg * PP_interp(t), gain_g, thresh_g,)}")
        # logger.info(f"at time {t}: PP(g): {wPPg * PP_interp(t)}")
        # logger.info(f"at time {t}: input(g): {wgg * g + wmg * m - wbg * b - whg * h + wPPg * PP_interp(t)}")
        # logger.info(f"at time {t}: F(b): {self.F(-wbb * b + wgb * g + wmb * m - whb * h + wPPb * PP_interp(t), gain_b, thresh_b,)}")
        # logger.info(f"at time {t}: F(m): {self.F(wmm * m + wgm * g - wbm * b - whm * h, gain_m, thresh_m)}")
        # logger.info(f"at time {t}: F(h): {self.F(wmh * m + wgh * g - whh * h, gain_h, thresh_h)}")
        # logger.info(f"at time {t}: exc input m: {wmm * m + wgm * g}")
        # logger.info(f"at time {t}: inh input m: {wbm * b + whm * h}")

        dg = (
            -g
            + self.F(
                wgg * g + wmg * m - wbg * b - whg * h + wPPg * PP_interp(t),
                gain_g,
                thresh_g,
            )
        ) / tau_g
        db = (
            -b
            + self.F(
                -wbb * b + wgb * g + wmb * m - whb * h + wPPb * PP_interp(t),
                gain_b,
                thresh_b,
            )
        ) / tau_b
        dm = (
            -m + self.F(wmm * m + wgm * g - wbm * b - whm * h, gain_m, thresh_m)
        ) / tau_m
        dh = (-h + self.F(wmh * m + wgh * g - whh * h, gain_h, thresh_h)) / tau_h

        # logger.info(f"at time {t}: PP: {PP_interp(t)} g: {g} dg: {dg} db: {db} dm: {dm} dh: {dh}")

        return [dg, db, dm, dh]

    def simulate_DG_ivp(
        self,
        tau_g,
        gain_g,
        thresh_g,
        tau_b,
        gain_b,
        thresh_b,
        tau_m,
        gain_m,
        thresh_m,
        tau_h,
        gain_h,
        thresh_h,
        wgg,
        wmg,
        wbg,
        whg,
        wbb,
        wgb,
        wmb,
        whb,
        wmm,
        wgm,
        wbm,
        whm,
        wmh,
        wgh,
        whh,
        wPPg,
        wPPb,
        PP,
        range_t,
        dt,
        g_init,
        b_init,
        m_init,
        h_init,
        **kwargs,
    ):
        """
        Use scipy solver instead of manual Euler
        """
        t_span = [range_t[0], range_t[-1]]
        y0 = [g_init, b_init, m_init, h_init]
        numpoints = kwargs.get("numpoints", len(range_t))
        t_eval = np.linspace(range_t[0], range_t[-1], numpoints)

        sol = solve_ivp(
            self.system_dynamics,
            t_span,
            y0,
            args=(
                tau_g,
                gain_g,
                thresh_g,
                tau_b,
                gain_b,
                thresh_b,
                tau_m,
                gain_m,
                thresh_m,
                tau_h,
                gain_h,
                thresh_h,
                wgg,
                wmg,
                wbg,
                whg,
                wbb,
                wgb,
                wmb,
                whb,
                wmm,
                wgm,
                wbm,
                whm,
                wmh,
                wgh,
                whh,
                wPPg,
                wPPb,
                PP,
            ),
            method="BDF",
            max_step=0.01,
            rtol=1e-4,
            atol=1e-6,
            t_eval=t_eval,
        )

        assert sol.success

        return sol.y

    def run(self, **kwargs):
        params = self.parameters(**kwargs)
        g, b, m, h = self.simulate_DG(**params)
        return {"g": g, "b": b, "m": m, "h": h}

    def compute_PSD(self, x, window_size=256, frequency_range=(0, 100.0), overlap=0.9):
        Fs = 1.0 / self.pars["dt"]

        nperseg = window_size
        win = signal.get_window("hann", nperseg)
        noverlap = int(overlap * nperseg)

        freqs, psd = signal.welch(
            x,
            fs=Fs,
            scaling="density",
            nperseg=nperseg,
            noverlap=noverlap,
            window=win,
            return_onesided=True,
        )

        freqinds = np.where(
            (freqs >= frequency_range[0]) & (freqs <= frequency_range[1])
        )

        freqs = freqs[freqinds]
        psd = psd[freqinds]
        if np.all(psd):
            psd = 10.0 * np.log10(psd)

        peak_index = np.where(psd == np.max(psd))[0]

        return freqs, psd, peak_index

    def _test_integration_method_equivalence(self, params=None, region=None):
        if params is None:
            params = self.parameters()
        if region is None:
            region = slice(None)

        t = time.time()
        g_orig, b_orig, m_orig, h_orig = self.simulate_DG_euler(**params)
        print(f"euler: {time.time() - t}")
        t = time.time()
        g_scipy, b_scipy, m_scipy, h_scipy = self.simulate_DG_ivp(**params)
        print(f"scipy: {time.time() - t}")
        np.testing.assert_allclose(
            g_orig[region], g_scipy[region], rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(
            b_orig[region], b_scipy[region], rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(
            m_orig[region], m_scipy[region], rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(
            h_orig[region], h_scipy[region], rtol=1e-3, atol=1e-5
        )


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
}  # mean firing rate targets

constraint_names = ["pos_g", "pos_b", "pos_m", "pos_h"]


def obj_fun(pp):
    """Objective function to be minimized."""

    network_model = DGRate(PP_freq="theta", **pp)
    output = network_model.run()

    logger.info(f"DGRate: \t pp:{pp}, output:{output}")
    res = np.asarray(
        [np.abs(np.mean(output[k]) - objective_targets[k]) for k in objective_names],
        dtype=np.float32,
    )

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
