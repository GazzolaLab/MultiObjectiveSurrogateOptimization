# %%
import os
from machinable import get
from matplotlib import pyplot as plt
from utils import globus_download

get("machinable.index", os.environ.get("STORAGE", None)).__enter__()

# Retrieve experiment
# %%
# automatic globus download, comment out if using manual download
assert globus_download("98bc01162f9d-0008-d467-5bca-064cc560")
# %%

experiment = get(
    "interface.dg_rate",
    {
        "dopt_params": {
            "n_epochs": 10,
        },
    },
)

# if using manual download, uncomment following line
# experiment = get('machinable.component').from_directory(YOUR DOWNLOAD PATH HERE)

assert experiment.cached()

# %%

# Plot solution

selected_solution = experiment.get_best()["x"][1]

network_model = experiment.get_model()

output = network_model.run()
params = network_model.pars

g, b, m, h = (output[k] for k in ["g", "b", "m", "h"])

fig, axs = plt.subplots(5, 2, figsize=(4, 5))

axs[0, 0].plot(params["range_t"], h, color="0.5", label="HIPP")
axs[0, 0].set_ylabel("HIPP")

axs[1, 0].plot(params["range_t"], b, color="0.5", label="BC")
axs[1, 0].set_ylabel("BC")

axs[2, 0].plot(params["range_t"], m, color="0.5", label="MC")
axs[2, 0].set_ylabel("MC")

axs[3, 0].plot(params["range_t"], g, color="0.5", label="GC")
axs[3, 0].set_ylabel("GC")

axs[4, 0].plot(params["range_t"], params["PP"], color="0.5", label="PP")
axs[4, 0].set_ylabel("PP")
axs[4, 0].set_xlabel("Time (ms)")


h_freqs, h_psd, h_peak_index = network_model.compute_PSD(h)
axs[0, 1].plot(h_freqs, h_psd, linewidth=3)
axs[0, 1].set_title("PSD (peak: %.3g Hz)" % (h_freqs[h_peak_index]))
axs[0, 1].set_ylabel("Power Spectral Density (dB/Hz)")

b_freqs, b_psd, b_peak_index = network_model.compute_PSD(b)
axs[1, 1].plot(b_freqs, b_psd, linewidth=3)
axs[1, 1].set_title("PSD (peak: %.3g Hz)" % (b_freqs[b_peak_index]))

m_freqs, m_psd, m_peak_index = network_model.compute_PSD(m)
axs[2, 1].plot(m_freqs, m_psd, linewidth=3)
axs[2, 1].set_title("PSD (peak: %.3g Hz)" % (m_freqs[m_peak_index]))

g_freqs, g_psd, g_peak_index = network_model.compute_PSD(g)
axs[3, 1].plot(g_freqs, g_psd, linewidth=3)
axs[3, 1].set_title("PSD (peak: %.3g Hz)" % (g_freqs[g_peak_index]))

pp_freqs, pp_psd, pp_peak_index = network_model.compute_PSD(params["PP"])
axs[4, 1].plot(pp_freqs, pp_psd, linewidth=3)
axs[4, 1].set_title("PSD (peak: %.3g Hz)" % (pp_freqs[pp_peak_index]))
axs[4, 1].set_xlabel("Frequency (Hz)")

fig.tight_layout()
fig.align_ylabels()


plt.show()

# %%
