import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ------------ CONFIG ------------
csv_file = "../Data/foam_data.csv"
time_col = 1
y_foam_col = 4
y_beer_col = 6

jump_threshold = 0.05        # Filter threshold for |Δy|

plot_t_min = 0.0            # Minimum time plotted
plot_t_max = 2522.0           # Maximum time plotted
# --------------------------------


def load_and_clean(csv):
    df = pd.read_csv(csv, sep=None, engine="python")
    df = df.dropna(how="all", axis=1)

    # use different variable names for the actual column data
    t_vals = pd.to_numeric(df.iloc[:, time_col], errors="coerce")
    foam_vals = pd.to_numeric(df.iloc[:, y_foam_col], errors="coerce")
    beer_vals = pd.to_numeric(df.iloc[:, y_beer_col], errors="coerce")

    mask = (~t_vals.isna()) & (~foam_vals.isna()) & (~beer_vals.isna())

    return t_vals[mask].values, foam_vals[mask].values, beer_vals[mask].values



def filter_spikes(t, y1,y2, threshold):
    keep = [True]  # first point always kept
    for i in range(1, len(y1)):
        if abs(y1[i] - y1[i-1] and y2[i]-y2[i-1]) > threshold:
            keep.append(False)
        else:
            keep.append(True)
    keep = np.array(keep)
    return t[keep], y1[keep], y2[keep]


def exp_func(t, a, b, c):
    return a * np.exp(b * t) + c


def compute_r2(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


# Load + filter
t, y1, y2 = load_and_clean(csv_file)
t, y1, y2 = filter_spikes(t, y1,y2, jump_threshold)

# Restrict plotting interval
mask_plot = (t >= plot_t_min) & (t <= plot_t_max)
t_plot = t[mask_plot]
y1_plot = y1[mask_plot]
y2_plot = y2[mask_plot]

log_data = np.log(y1_plot)

a, b = np.polyfit(t_plot, log_data, 1)

y_pred = np.exp(a*(t_plot) + b)

r2 = compute_r2(y1_plot, y_pred)

print("R^2 = ", r2)

# Plot
plt.figure(figsize=(8,5), dpi=120)
plt.scatter(t_plot, y1_plot, s=2, label="Filtered Data Foam Area", color="purple", marker="x")
plt.scatter(t_plot, y2_plot, s=2, label="Filtered Data Beer Area", color="hotpink", marker="x")
plt.plot(t_plot, y_pred, color="b")
# plt.plot(t_fit, y_fit, linewidth=2, label="Exponential Fit")
plt.xlabel(r"$t \; \left[s \right]$")
plt.ylabel("Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

