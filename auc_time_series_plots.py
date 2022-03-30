import os

import matplotlib.pyplot as plt
import numpy as np

#  Data directory
data_dir = os.path.join(os.getcwd(), "time_series")

# Onsets dir
onsets_dir = os.path.join(data_dir, "onsets")

#  Number of onset groups
onsets_groups = 6

#  List of trials
trials = ["BMOT", "FTAP", "HOUS", "MUSI", "READ"]

#  List of AUCs
aucs = ["AUC", "ST-THR", "TD-THR"]

#  Colors
colors = ["#357FA6", "#F28B66", "#56A669"]

# Generate time array with 215 points and TR = 2s
TR = 2
time = np.arange(0, 215 * TR, TR)

#  Generate figure with 5 vertical subplots
fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 15))

# Loop over trials
for trial_idx, trial in enumerate(trials):

    # Loop over onset groups
    for group in range(onsets_groups):
        #  Load onset file with delimeter ,
        onset_file = os.path.join(onsets_dir, f"P3SBJ07_Events01_{trial}_EV0{group+1}_EVols.1D")
        onset_idxs = np.loadtxt(onset_file, dtype=int, delimiter=",")

        #  Map onset indexes to time array
        onset_times = time[onset_idxs]
        onsets = [onset_times[0], onset_times[-1]]

        # Plot onsets as background shade when onsets = 1
        axs[trial_idx].fill_between(onsets, y1=0, y2=1, color="grey", alpha=0.2)

    # Loop over AUCs
    for auc in aucs:
        # Load data
        if auc == "AUC":
            data = np.loadtxt(os.path.join(data_dir, f"{trial.lower()}_auc_ts.1D"))
            color = colors[0]
        elif auc == "ST-THR":
            data = np.loadtxt(os.path.join(data_dir, f"{trial.lower()}_auc_thr_ts.1D"))
            color = colors[1]
        elif auc == "TD-THR":
            data = np.loadtxt(os.path.join(data_dir, f"{trial.lower()}_auc_thr_td_ts.1D"))
            color = colors[2]

        # Plot data
        axs[trial_idx].plot(time, data, label=f"{auc}", color=color)

    # Set y-axis limits
    axs[trial_idx].set_ylim(0, 1)

    # Set y-axis label
    axs[trial_idx].set_ylabel(f"{trial}")

    # Remove blank space inside of the box
    axs[trial_idx].spines["top"].set_visible(False)
    axs[trial_idx].spines["right"].set_visible(False)
    axs[trial_idx].margins(x=0)

# Set x-axis label
axs[-1].set_xlabel("Time (s)")

# Set legend
axs[0].legend()

# Tight layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, bottom=0.05, top=0.95)

# plt.show()

fig.savefig(
    os.path.join(os.getcwd(), "figures", "auc_time_series.png"),
    dpi=300,
    bbox_inches="tight",
)
