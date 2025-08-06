# -*- coding: utf-8 -*-
"""TOTAL PREDICTION.ipynb with TCN model"""

import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import swiftxrt_clean
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold

# GPU check
device_name = tf.test.gpu_device_name()
if len(device_name) > 0:
    print("Found GPU at: {}".format(device_name))
else:
    device_name = "/device:CPU:0"
    print("No GPU found, using {}.".format(device_name))

# Step 1: Cleaning SWIFT XRT Data
print("\nStep 1: Cleaning SWIFT XRT Data")
swiftxrt_clean.clean_file('C:/Users/Nur/Desktop/GRB/GRB090426.dat')
data_orig = pd.read_table('C:/Users/Nur/Desktop/GRB/GRB090426.dat', sep='\s+')

# Step 2: Sorting data
print("Step 2: Sorting data by time")
after_burst = data_orig.sort_values(by=['!Time'])

# Step 3: Preprocessing
print("Step 3: Data preprocessing")
after_burst['!Time'] = after_burst['!Time'] / 10**2
b = after_burst['Flux'].min()
after_burst['Flux'] = np.log(after_burst['Flux'] / b) + 1
a = after_burst['Fluxpos'].min()
after_burst['Fluxpos'] = np.log(after_burst['Fluxpos'] / a) + 1

# Function definitions

def create_batches(N, Ba_Sz, updated_after_burst):
    print(f"Creating batches: total points = {N}, batch size = {Ba_Sz}")
    batches = []
    start = 0
    while start < N:
        end = min(start + Ba_Sz, N)
        print(f"  - Creating batch from {start} to {end}")
        after_burst_even = updated_after_burst.iloc[start:end].reset_index(drop=True)
        batches.append(upsample_data(after_burst_even))
        start += Ba_Sz
    return batches

def create_data(t, f):
    updated_after_burst = pd.DataFrame(columns=[t, f])
    for i in range(after_burst.shape[0] - 1):
        start_time = after_burst[t].iloc[i]
        end_time = after_burst[t].iloc[i + 1]
        start_flux = after_burst[f].iloc[i]
        end_flux = after_burst[f].iloc[i + 1]
        time_upd = (end_time - start_time) / 20
        flux_upd = (end_flux - start_flux) / 20
        for j in range(19):
            new_row = {t: start_time, f: start_flux}
            updated_after_burst = pd.concat([updated_after_burst, pd.DataFrame([new_row])], ignore_index=True)
            start_time += time_upd
            start_flux += flux_upd
    return updated_after_burst

def upsample_data(data):
    while data.shape[0] < 20000:
        data = pd.concat([data, data], ignore_index=True)
    return data

def create_sequence(dataset):
    return dataset.reshape(-1, 1, 1)

def create_tcn(X, Y):
    input_shape = (X.shape[1], X.shape[2])
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=5, padding='causal', activation='relu', dilation_rate=1)(inputs)
    x = Conv1D(64, kernel_size=5, padding='causal', activation='relu', dilation_rate=2)(x)
    x = Conv1D(64, kernel_size=5, padding='causal', activation='relu', dilation_rate=4)(x)
    x = Flatten()(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def fit_model(model, X, Y):
    print("    Starting model training...")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(X, Y, epochs=100, validation_split=0.4, batch_size=15, shuffle=True, callbacks=[early_stop])
    print("    Training completed")
    return history

def train_model(batches, B, t, f):
    print(f"Step 4: Training the model on {B} batches")
    preds = []
    for b in range(B):
        batch = batches[b]
        X = create_sequence(batch[[t]].values)
        Y = batch[f].values[-X.shape[0]:]
        print(f"  - Training on batch {b + 1}/{B}, X shape: {X.shape}")
        model_b = create_tcn(X, Y)
        fit_model(model_b, X, Y)
        pr = model_b.predict(X)
        preds.extend(pr)
    return preds

def time_seq(batches, B, t):
    return [batches[b][t].iloc[i] for b in range(B) for i in range(len(batches[b]))]

# Step 4: Flux-Time prediction
print("Step 5: Generating predictions from data")
flux_time_upd_data = create_data('!Time', 'Flux')
Ba_Sz = int(input("Enter batch size: "))
N = flux_time_upd_data.shape[0]
batches_flux_time = create_batches(N, Ba_Sz, flux_time_upd_data)
B = len(batches_flux_time)

with tf.device(device_name):
    flux_preds = train_model(batches_flux_time, B, '!Time', 'Flux')

print("Step 6: Postprocessing and plotting")
ti, fl = [], []
for batch in batches_flux_time:
    fl.extend(batch['Flux'])
    ti.extend(batch['!Time'])

X_new, idx = np.unique(np.array(ti), return_index=True)
Y_new = np.array([flux_preds[i] for i in idx])

plt.scatter(after_burst['!Time'], after_burst['Flux'])
plt.plot(X_new, Y_new[:, 0], color='orange', linestyle=(0, (0.1, 2)), dash_capstyle='round', linewidth=7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("t (s)")
plt.ylabel("Flux (erg/cm^2/s)")
plt.legend(['Original Points', 'Predicted Points'])
plt.title("Flux vs Time Prediction for GRB GRB060206 (TCN Model)")
plt.show()

print("\nStep 7: Calculating MAE, RMSE, R²")

f_log = after_burst['Flux'].values
f_true_real = b * np.exp(f_log - 1)

f_pred_log_interp = interp1d(X_new, Y_new[:, 0], kind='linear', fill_value='extrapolate')(after_burst['!Time'].values)
f_pred_real = b * np.exp(f_pred_log_interp - 1)

mae = mean_absolute_error(f_true_real, f_pred_real)
rmse = np.sqrt(mean_squared_error(f_true_real, f_pred_real))
r2 = r2_score(f_true_real, f_pred_real)

print(f"MAE: {mae:.6e}")
print(f"RMSE: {rmse:.6e}")
print(f"R²: {r2:.4f}")

print("Done!")

np.save("X_new_tcn.npy", X_new)
np.save("Y_new_tcn.npy", Y_new)
