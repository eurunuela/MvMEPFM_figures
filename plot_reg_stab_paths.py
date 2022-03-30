import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import lars_path

# Global variables
nscans = 215
TR = 2
TE = np.array([16.3, 32.2, 48.1]) / 1000
nTE = len(TE)
nsurr = 50
trial_list = ["hous", "musi", "ftap", "read", "bmot"]


def hrf_afni(tr, lop_hrf):
    dur_hrf = 8
    last_hrf_sample = 1
    #  Increases duration until last HRF sample is zero
    while last_hrf_sample != 0:
        dur_hrf = 2 * dur_hrf
        npoints_hrf = np.round(dur_hrf, tr)
        hrf_command = (
            "3dDeconvolve -x1D_stop -nodata %d %f -polort -1 -num_stimts 1 -stim_times 1 '1D:0' '%s' -quiet -x1D stdout: | 1deval -a stdin: -expr 'a'"
            % (dur_hrf, tr, lop_hrf)
        )
        hrf_tr_str = subprocess.check_output(
            hrf_command, shell=True, universal_newlines=True
        ).splitlines()
        hrf_tr = np.array([float(i) for i in hrf_tr_str])
        last_hrf_sample = hrf_tr[len(hrf_tr) - 1]
        if last_hrf_sample != 0:
            print(
                "Duration of HRF was not sufficient for specified model. Doubling duration and computing again."
            )

    #  Removes tail of zero samples
    while last_hrf_sample == 0:
        hrf_tr = hrf_tr[0 : len(hrf_tr) - 1]
        last_hrf_sample = hrf_tr[len(hrf_tr) - 1]

    return hrf_tr


def generate_hrf(TR, nscans, TE):
    hrf_shape = hrf_afni(TR, "SPMG1")

    L_hrf = len(hrf_shape)  # Length
    max_hrf = max(abs(hrf_shape))  # Max value
    filler = np.zeros(nscans - hrf_shape.shape[0], dtype=np.int)
    hrf_shape = np.append(hrf_shape, filler)  # Fill up array with zeros until nscans

    temp = hrf_shape

    for i in range(nscans - 1):
        foo = np.append(np.zeros(i + 1), hrf_shape[0 : (len(hrf_shape) - i - 1)])
        temp = np.column_stack((temp, foo))

    if TE is not None and len(TE) > 1:
        tempTE = -TE[0] * temp
        for teidx in range(len(TE) - 1):
            tempTE = np.vstack((tempTE, -TE[teidx + 1] * temp))
    else:
        tempTE = temp

    hrf = tempTE

    hrf_norm = hrf / max_hrf

    return hrf_norm


def calculate_stability_path(coefs, lambdas):
    # Sorting and getting indexes
    lambdas_merged = lambdas.copy()
    lambdas_merged = lambdas_merged.reshape((nscans * nsurr,))
    sort_idxs = np.argsort(-lambdas_merged)
    lambdas_merged = -np.sort(-lambdas_merged)

    # Initialize stability path matrix with zeros
    stability_path = np.zeros((nscans, nsurr * nscans), dtype=np.float64)

    for surrogate_idx in range(nsurr):
        if surrogate_idx == 0:
            first = 0
            last = nscans - 1
        else:
            first = last + 1
            last = first + nscans - 1

        same_lambda_idxs = np.where((first <= sort_idxs) & (sort_idxs <= last))[0]

        # Find indexes of changes in value (0 to 1 changes are expected).
        # nonzero_change_scans, nonzero_change_idxs = np.where(np.squeeze(coef_path[surrogate_idx, :, :-1]) != np.squeeze(coef_path[surrogate_idx, :, 1:]))
        coefs_temp = np.squeeze(coefs[surrogate_idx, :, :])
        if len(coefs_temp.shape) == 1:
            coefs_temp = coefs_temp[:, np.newaxis]
        diff = np.diff(coefs_temp)
        nonzero_change_scans, nonzero_change_idxs = np.where(diff)
        nonzero_change_idxs = nonzero_change_idxs + 1

        # print(f'{nonzero_change_idxs}')

        coefs_squeezed = np.squeeze(coefs[surrogate_idx, :, :])
        coefs_merged = np.full((nscans, nscans * nsurr), False, dtype=bool)
        coefs_merged[:, same_lambda_idxs] = coefs_squeezed.copy()

        for i in range(len(nonzero_change_idxs)):
            coefs_merged[
                nonzero_change_scans[i],
                same_lambda_idxs[nonzero_change_idxs[i]] :,
            ] = True

        # Sum of non-zero coefficients
        stability_path += coefs_merged

    # Divide by number of surrogates
    stability_path = stability_path / nsurr

    return stability_path, lambdas_merged


def main(trial):
    # Read data and onsets
    print("Reading data...")
    onsets = np.loadtxt(
        os.path.join("time_series", "onsets", f"P3SBJ07_Events01_{trial.upper()}_EVols.1D"),
        delimiter=",",
        dtype=int,
    )
    echo_1 = np.loadtxt(os.path.join("time_series", f"{trial}_echo-01.1D"))
    echo_2 = np.loadtxt(os.path.join("time_series", f"{trial}_echo-02.1D"))
    echo_3 = np.loadtxt(os.path.join("time_series", f"{trial}_echo-03.1D"))

    y = np.concatenate((np.concatenate((echo_1, echo_2)), echo_3))

    # Indices outside of onsets
    non_onset = np.delete(np.arange(nscans), onsets)

    # Generate HRF
    print("Generating HRF...")
    hrf = generate_hrf(TR, nscans, TE)

    # Calculate regularization path
    print("Calculating regularization path...")
    lambdas, _, coef_path = lars_path(
        hrf,
        np.squeeze(y),
        method="lasso",
        Gram=np.dot(hrf.T, hrf),
        Xy=np.dot(hrf.T, np.squeeze(y)),
        max_iter=nscans,
        eps=1e-6,
    )

    # Plot regularization path on the left subplot of 2
    print("Plotting regularization path...")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].margins(x=0)
    ax[0].plot(coef_path[non_onset, :].T, color="#357FA6", linewidth=0.5)
    ax[0].plot(coef_path[onsets, :].T, color="#F28B66", linewidth=2)
    ax[0].set_xlabel("λ")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Regularization path")
    print("Regularization path plotted.")

    # Initialize lambdas and coef_path for stability selection with nsurr surrogates
    lambdas_surr = np.zeros((nsurr, nscans))
    coef_path_surr = np.zeros((nsurr, nscans, nscans))

    # Loop through number of surrogates to calculate the stability path
    print("Calculating stability path...")
    for surr_idx in range(nsurr):
        # Generate an array with random indices for 60% of the nscans
        rand_idx = np.random.choice(np.arange(nscans), int(np.round(nscans * 0.6)), 0)

        # Sort the indices
        rand_idx.sort()

        # Use same indices for the other 2 echos
        rand_idx = np.concatenate(
            (np.concatenate((rand_idx, rand_idx + nscans)), rand_idx + 2 * nscans)
        )

        # Generate surrogate data with random indices
        surr_y = y[rand_idx]

        # Get HRF for surrogate data
        surr_hrf = hrf[rand_idx, :]

        # Calculate regularization path for surrogate data
        temp_lambda, _, temp_coef_path = lars_path(
            surr_hrf,
            np.squeeze(surr_y),
            method="lasso",
            Gram=np.dot(surr_hrf.T, surr_hrf),
            Xy=np.dot(surr_hrf.T, np.squeeze(surr_y)),
            max_iter=nscans,
            eps=1e-6,
        )

        # Save lambda and coef_path for surrogate with zero padding to fit the shape of lambda_surr and coef_path_surr
        lambdas_surr[surr_idx, : temp_lambda.shape[0]] = temp_lambda
        coef_path_surr[surr_idx, :, : temp_coef_path.shape[1]] = temp_coef_path

        print(f"Surrogate {surr_idx + 1} of {nsurr} calculated.")

    # Calculate stability path
    stability_path, lambdas_merged = calculate_stability_path(
        coef_path_surr,
        lambdas_surr,
    )

    # Plot stability path
    print("Plotting stability path...")
    ax[1].margins(x=0)
    ax[1].plot(lambdas_merged, stability_path[non_onset, :].T, color="#357FA6", linewidth=0.5)
    ax[1].plot(lambdas_merged, stability_path[onsets, :].T, color="#F28B66", linewidth=2)
    # Flip x axis
    ax[1].invert_xaxis()
    ax[1].set_xlabel("λ")
    ax[1].set_ylabel("Probability")
    ax[1].set_title("Stability path")
    ax[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join("figures", f"regulariation_and_stability_paths_{trial}.png"), dpi=300)
    plt.close()
    print("Stability path plot saved.")


if __name__ == "__main__":
    # Loop through all trials
    for trial in trial_list:
        print(f"Trial {trial}")
        main(trial)
    print("Done.")
