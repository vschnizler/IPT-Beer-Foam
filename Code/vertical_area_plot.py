import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ------------ CONFIG ------------
csv_file = "../Data/Vertical_Data/foam_data_vertical.csv"
time_col = 1
y_foam_col = 4
y_beer_col = 6

jump_threshold = 0.05     # Filter threshold for |Δy|

plot_t_min = 140.0            # Minimum time plotted
plot_t_max = 2400.0           # Maximum time plotted
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

def filter_spikes_single(t, y1, threshold):
    out =filter_spikes(t, y1, y1, threshold=threshold)
    return out[0], out[1]


def filter_spikes(t, y1, y2, threshold):
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

# Plot Single File

def plot_beer(csv_file, show=True, save=False, target=None):

    # Load + filter
    lowbnd = 200
    t, y1, _ = load_and_clean(csv_file)
    t, y1 = filter_spikes_single(t, y1, jump_threshold)

    # Restrict plotting interval
    mask_plot = (t >= plot_t_min) & (t <= plot_t_max)
    t_plot = t[mask_plot]
    t_plot = t_plot[lowbnd:]
    y1_plot = y1[mask_plot]
    y1_plot = y1_plot[lowbnd:]

    # Exponential fit
    log_data = np.log(y1_plot)
    a, b = np.polyfit(t_plot, log_data, 1)
    y_pred = np.exp(a*(t_plot) + b)
    r2 = compute_r2(y1_plot, y_pred)


    plt.figure(figsize=(8,5), dpi=120)
    plt.scatter(t_plot, y1_plot, s=2, label="Filtered Data Foam Area " + csv_file, color="purple", marker="x")
    plt.plot(t_plot, y_pred, color="b", linewidth=2, label=f"Exponential Fit (R²={r2:.4f})")
    plt.xlabel(r"$t \; \left[s \right]$")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if(save and target != None):
        plt.savefig(target, dpi=1200)

    if(show):
        plt.show()

def plot_beer_multiple(files, save=False, targets=None):

    for file, target in zip(files, targets):
        plot_beer(file, show=False, save=save, target=target)

    plt.show()

def fit_with_curvature(area_csv, curv_csv, low_frame, high_frame):
    """
    Synchronizes area and curvature data by merging on frame number,
    then fits the model: Area = exp(-Curvature * alpha * t)
    """
    # 1. Load both datasets
    df_area = pd.read_csv(area_csv, sep=None, engine="python")
    df_curv = pd.read_csv(curv_csv)

    # 2. Merge on frame number to align datasets
    # Area CSV frame column (col 0) should match Curvature CSV 'frame'
    df_curv = df_curv.rename(columns={'frame': 'frame_number'})
    # Ensure the area dataframe has a matching column name for the merge
    df_area.columns.values[0] = 'frame_number'
    
    # Inner join keeps only rows present in both files
    merged = pd.merge(df_area, df_curv, on='frame_number')

    # 3. Clean the merged data (equivalent to load_and_clean logic)
    t_vals = pd.to_numeric(merged.iloc[:, time_col], errors="coerce")
    foam_vals = pd.to_numeric(merged.iloc[:, y_foam_col], errors="coerce")
    kappa_vals = merged['avg_curvature']

    # Apply mask for NaNs and the requested frame range
    mask = (~t_vals.isna()) & (~foam_vals.isna()) & \
           (merged['frame_number'] >= low_frame) & \
           (merged['frame_number'] <= high_frame)

    t_slice = t_vals[mask].values
    area_slice = foam_vals[mask].values
    kappa_slice = kappa_vals[mask].values

    # 4. Define Model: Area = exp(-kappa * alpha * t)
    def curvature_decay_func(combined_input, alpha):
        t_val, k_val = combined_input
        return np.exp(-k_val * alpha * t_val)

    # 5. Perform Fit
    # p0 is an initial guess for alpha
    try:
        popt, _ = curve_fit(curvature_decay_func, (t_slice, kappa_slice), area_slice, p0=[0.0001])
        alpha_fit = popt[0]
    except Exception as e:
        # print(f"Fit failed: {e}")
        return None

    # 6. Generate Predictions and R^2
    area_pred = curvature_decay_func((t_slice, kappa_slice), alpha_fit)
    
    # Calculate R^2 using your compute_r2 logic
    ss_res = np.sum((area_slice - area_pred) ** 2)
    ss_tot = np.sum((area_slice - np.mean(area_slice)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return t_slice, area_slice, area_pred, r2, alpha_fit

def plot_curvature_fit(area_csv, curv_csv, low, high, save, target=None):
    # print("Plotting")
    result = fit_with_curvature(area_csv, curv_csv, low, high)
    if result is None: return

    t_plot, y_plot, y_pred, r2, alpha = result

    plt.figure(figsize=(8,5), dpi=120)
    plt.scatter(t_plot, y_plot, s=2, color="purple", label="Raw Area Data", alpha=0.5)
    plt.plot(t_plot, y_pred, color="red", linewidth=2, label=f"Fit: $e^{{-\kappa \\alpha t}}$ (R²={r2:.4f})")
    
    plt.title(f"Curvature-Dependent Decay Fit (α = {alpha:.2e})")
    plt.xlabel(r"$t \; [s]$")
    plt.ylabel("Foam Area Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if(save):
        plt.savefig(target, dpi=150)
    plt.clf()
    return t_plot, y_plot, y_pred, r2, alpha

# file_area = "../Data/Vertical_Data/uhighsens.csv"
# file_curve = "../Data/Vertical_Data/Curve_Data/curvature_6.csv"

# plot_curvature_fit(file_area, file_curve, 0, 150)
# path = "../Data/Vertical_Data/"
# files = [path + "highsens.csv", path + "lowsens.csv", path + "medsens.csv", path + "uhighsens.csv", path + "vhighsens.csv", path + "uuhighsens.csv"]

# path = "../Plots/Vertical_Analysis/"
# targets = [path + "highsens.png", path + "lowsens.png", path + "medsens.png", path + "uhighsens.png", path + "vhighsens.png", path + "uuhighsens.png"]

# plot_beer_multiple(files=files, save=False, targets=targets)

# Load + filter
# t, y1, y2 = load_and_clean(csv_file)
# t, y1, y2 = filter_spikes(t, y1,y2, jump_threshold)

# # Restrict plotting interval
# mask_plot = (t >= plot_t_min) & (t <= plot_t_max)
# t_plot = t[mask_plot]
# y1_plot = y1[mask_plot]
# y2_plot = y2[mask_plot]

# log_data = np.log(y1_plot)

# a, b = np.polyfit(t_plot, log_data, 1)

# y_pred = np.exp(a*(t_plot) + b)

# r2 = compute_r2(y1_plot, y_pred)

# # print("R^2 = ", r2)

# # Plot
# plt.figure(figsize=(8,5), dpi=120)
# plt.scatter(t_plot, y1_plot, s=2, label="Filtered Data Foam Area", color="purple", marker="x")
# plt.scatter(t_plot, y2_plot, s=2, label="Filtered Data Beer Area", color="hotpink", marker="x")
# plt.plot(t_plot, y_pred, color="b")
# # plt.plot(t_fit, y_fit, linewidth=2, label="Exponential Fit")
# plt.xlabel(r"$t \; \left[s \right]$")
# plt.ylabel("Ratio")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()