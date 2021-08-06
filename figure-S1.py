import matplotlib.pyplot as plt
from cell_models import kernik, protocols


k = kernik.KernikModel()
k_art = kernik.KernikModel(is_exp_artefact=True)


p = protocols.VoltageClampProtocol([
        protocols.VoltageClampStep(-80, 200),
        protocols.VoltageClampStep(-30, 200)])

tr = k.generate_response(p, is_no_ion_selective=False)
tr_art = k_art.generate_response(p, is_no_ion_selective=False)

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

labs = ['Baseline', 'With Artifact']
cols = ['k', 'b--']

for i, tr in enumerate([tr, tr_art]):
    axs[0].plot(tr.t, tr.y, cols[i], label=labs[i])
    axs[1].plot(tr.t, tr.current_response_info.get_current_summed(), cols[i])

axs[0].set_xlim(190, 220)
fs = 12
axs[0].set_ylabel('Membrane Voltage (mV)', fontsize=fs)

axs[1].set_xlabel('Time (ms)', fontsize=fs)
axs[1].set_ylabel('Measured Current (pA/pF)', fontsize=fs)

for ax in axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.rcParams['svg.fonttype'] = 'none'

axs[0].legend()
plt.savefig(f'./figS1-data/figureS1.svg', format='svg')
plt.show()
