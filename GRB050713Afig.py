# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import FuncFormatter
from math import isclose

# Load original data
data_orig = pd.read_table('C:/Users/Nur/Desktop/GRB/GRB050713A.dat', sep='\s+')
after_burst = data_orig.sort_values(by=['!Time'])

# Normalization as used in the models
after_burst['!Time'] = after_burst['!Time'] / 10**2
b = after_burst['Flux'].min()
after_burst['Flux'] = np.log(after_burst['Flux'] / b) + 1

# Load predictions
X_bilstm = np.load("X_new_bilstm.npy")
Y_bilstm = np.load("Y_new_bilstm.npy")[:, 0]

X_tcn = np.load("X_new_tcn.npy")
Y_tcn = np.load("Y_new_tcn.npy")[:, 0]

X_gru = np.load("X_new_gru.npy")
Y_gru = np.load("Y_new_gru.npy")[:, 0]

# Time grid for interpolation
t_vals = after_burst['!Time'].values

# Interpolate log-flux predictions
flux_bilstm_log = interp1d(X_bilstm, Y_bilstm, kind='linear', fill_value='extrapolate')(t_vals)
flux_tcn_log = interp1d(X_tcn, Y_tcn, kind='linear', fill_value='extrapolate')(t_vals)
flux_gru_log = interp1d(X_gru, Y_gru, kind='linear', fill_value='extrapolate')(t_vals)

# Inverse log-transformation
flux_bilstm = b * np.exp(flux_bilstm_log - 1)
flux_tcn = b * np.exp(flux_tcn_log - 1)
flux_gru = b * np.exp(flux_gru_log - 1)
flux_true = b * np.exp(after_burst['Flux'].values - 1)

# Plotting
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})

fig, main_ax = plt.subplots(figsize=(10, 6))

# Main plot
main_ax.scatter(t_vals, flux_true, color='firebrick', s=20, label='Original Data')
main_ax.plot(t_vals, flux_bilstm, label='BiLSTM', color='blue', linewidth=2, linestyle='--')
main_ax.plot(t_vals, flux_tcn, label='TCN', color='orange', linewidth=2, linestyle='--')
main_ax.plot(t_vals, flux_gru, label='BiGRU', color='green', linewidth=2, linestyle='--')

main_ax.set_xscale('log')
main_ax.set_yscale('log')
main_ax.set_xlabel("Time (s)")
main_ax.set_ylabel("Flux (erg/cm²/s)")
main_ax.legend()
main_ax.tick_params(axis='both', which='both', direction='in')

# Inset: zoomed-in region (highlighted yellow zone)
inset_ax = main_ax.inset_axes([0.5, 0.56, 0.45, 0.4])  # inset position
inset_ax.set_xscale('log')
inset_ax.set_yscale('log')
inset_ax.set_xlim(0.96, 2)       # region of interest (time)
inset_ax.set_ylim(5e-10, 1.2e-8)  # region of interest (flux)

# Redraw curves in the inset
inset_ax.scatter(t_vals, flux_true, color='firebrick', s=10)
inset_ax.plot(t_vals, flux_bilstm, color='blue', linestyle='--', linewidth=1.5)
inset_ax.plot(t_vals, flux_tcn, color='orange', linestyle='--', linewidth=1.5)
inset_ax.plot(t_vals, flux_gru, color='green', linestyle='--', linewidth=1.5)
inset_ax.tick_params(axis='both', which='both', labelsize=8)

# Connect inset to main axis
main_ax.indicate_inset_zoom(inset_ax, edgecolor="black")

plt.tight_layout()
plt.savefig("GRB050713A.png", dpi=600)
plt.show()
