import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from vert_ana import analyze_beer_foam
from pathlib import Path
from dataclasses import dataclass

@dataclass
class FoamAreaConfig:
    """Configuration for Foam Area fitting with specific plot directories."""
    video_file: str
    study_name: str = "Area_Analysis_01"
    base_output_dir: str = "../Data"
    base_plot_dir: str = "../Plots"
    
    sensitivity: int = 100
    low_bnd: int = 300
    high_bnd: int = 1500
    
    # Filtering parameters
    filter: bool = False  # Activates the delta filter
    max_delta: float = 0.05  # Max allowed change in area ratio per frame
    
    save: bool = True
    show: bool = False

    @property
    def root_dir(self) -> Path:
        return Path(self.base_output_dir) / f"{self.study_name}_Data"

    @property
    def sweep_plot_dir(self) -> Path:
        return Path(self.base_plot_dir) / self.study_name / "Sweep"

    @property
    def best_plot_dir(self) -> Path:
        return Path(self.base_plot_dir) / self.study_name

    @property
    def area_csv(self) -> str:
        video_stem = Path(self.video_file).stem
        return str(self.root_dir / f"{video_stem}_sens{self.sensitivity}.csv")


class SingleAreaRun:
    def __init__(self, config: FoamAreaConfig):
        self.cfg = config
        self.cfg.root_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.sweep_plot_dir.mkdir(parents=True, exist_ok=True)

    def execute_run(self):
        if not Path(self.cfg.area_csv).exists():
            analyze_beer_foam(
                video_path=self.cfg.video_file,
                output_csv=self.cfg.area_csv,
                sensitivity="custom",
                custom_threshold=self.cfg.sensitivity,
                show_preview=self.cfg.show,    
            )
        return self.run_fit()

    def run_fit(self) -> dict:
        try:
            df = pd.read_csv(self.cfg.area_csv)
            frames = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            t_vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            y_vals = pd.to_numeric(df.iloc[:, 4], errors="coerce")

            # Initial mask for bounds and non-zero values
            mask = (frames >= self.cfg.low_bnd) & (frames <= self.cfg.high_bnd) & (y_vals > 0)
            
            t_plot = t_vals[mask].values
            y_plot = y_vals[mask].values

            # Apply Delta Filter if enabled
            if self.cfg.filter and len(y_plot) > 1:
                # Calculate absolute differences between consecutive points
                deltas = np.abs(np.diff(y_plot))
                # Create a boolean mask: True if delta is below threshold
                # We prepend True because diff results in N-1 elements
                delta_mask = np.concatenate(([True], deltas <= self.cfg.max_delta))
                
                t_plot = t_plot[delta_mask]
                y_plot = y_plot[delta_mask]

            if len(y_plot) < 5:
                return {"r2": -np.inf}

            # Exponential fit
            b, a_log = np.polyfit(t_plot, np.log(y_plot), 1)
            y_pred = np.exp(b * t_plot + a_log)
            r2 = 1 - (np.sum((y_plot - y_pred)**2) / np.sum((y_plot - np.mean(y_plot))**2))

            if self.cfg.save:
                fname = f"fit_S{self.cfg.sensitivity}_L{self.cfg.low_bnd}_H{self.cfg.high_bnd}.png"
                self._save_plot(t_plot, y_plot, y_pred, r2, self.cfg.sweep_plot_dir / fname)
                
            return {
                "r2": r2, "decay": b, "sens": self.cfg.sensitivity,
                "low": self.cfg.low_bnd, "high": self.cfg.high_bnd,
                "t_plot": t_plot, "y_plot": y_plot, "y_pred": y_pred
            }
        except Exception as e:
            return {"r2": -np.inf, "error": str(e)}

    def _save_plot(self, t, y, y_pred, r2, full_path):
        plt.figure(figsize=(10, 6), dpi=400)
        plt.scatter(t, y, s=5, alpha=0.4, color="purple", label="Filtered Data" if self.cfg.filter else "Data")
        plt.plot(t, y_pred, 'r-', label=f"R²={r2:.4f}")
        plt.title(f"Foam Decay Fit (Filter: {self.cfg.filter}, MaxDelta: {self.cfg.max_delta})")
        plt.xlabel("Time (s)")
        plt.ylabel("Area Ratio")
        plt.legend()
        plt.savefig(full_path, dpi=400)
        plt.close()

class AreaSensitivityOptimizer:
    def __init__(self, base_config: FoamAreaConfig):
        self.base_cfg = base_config

    def run_sweep(self, sensitivity_list: list):
        results = []
        best_r2 = -np.inf
        best_params = None
        total = len(sensitivity_list)

        print(f"\n--- Starting Sweep: {total} Sensitivity Settings ---")

        for i, sens in enumerate(sensitivity_list, 1):
            self.base_cfg.sensitivity = int(sens)
            runner = SingleAreaRun(self.base_cfg)
            
            # Progress print
            print(f"[{i}/{total}] Current Best R²: {best_r2:.4f} | Testing Sens: {sens}...", end="\r")
            
            data = runner.run_from_video()
            results.append(data)
            
            if data['r2'] > best_r2:
                best_r2 = data['r2']
                best_params = data
                print(f"\n  >>> NEW LEADER: Sens {sens} | R²: {best_r2:.4f}")

        # Final Summary
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(self.base_cfg.root_dir / "sensitivity_sweep_summary.csv", index=False)
        
        print("\n" + "="*40)
        print(f"OPTIMIZATION COMPLETE")
        print(f"Best Sensitivity: {best_params['sens']}")
        print(f"Best R²: {best_r2:.4f}")
        print("="*40)
        
        return best_params

class FullAreaOptimizer:
    def __init__(self, base_config: FoamAreaConfig):
        self.base_cfg = base_config

    def run_optimization(self, sens_range: list, low_range: list, high_range: list):
        results = []
        best_r2 = -np.inf
        best_data = None
        
        combos = list(itertools.product(sens_range, low_range, high_range))
        total = len(combos)

        for i, (sens, low, high) in enumerate(combos, 1):
            if low >= high: continue

            self.base_cfg.sensitivity = sens
            self.base_cfg.low_bnd = low
            self.base_cfg.high_bnd = high
            
            runner = SingleAreaRun(self.base_cfg)
            data = runner.execute_run()
            
            if data['r2'] > best_r2:
                best_r2 = data['r2']
                best_data = data
                print(f"\n[NEW BEST] R²: {best_r2:.4f} | Sens: {sens}, Window: [{low}-{high}]")
            
            print(f"\rProgress: {i}/{total} | Current Best R²: {best_r2:.4f}", end="")
            results.append(data)

        # SAVE THE BEST FIT SEPARATELY
        if best_data:
            print(f"\nSaving Best Fit to {self.base_cfg.best_plot_dir}...")
            # We use the internal _save_plot from a temporary runner
            final_runner = SingleAreaRun(self.base_cfg)
            final_path = self.base_cfg.best_plot_dir / "BEST_FIT_RESULTS.png"
            final_runner._save_plot(
                best_data['t_plot'], 
                best_data['y_plot'], 
                best_data['y_pred'], 
                best_data['r2'], 
                final_path
            )

        return best_data

if __name__ == "__main__":
    conf = FoamAreaConfig(video_file="../Videos/Cropped_Vertical.MOV", save=False, show=True, study_name="Presentation_Run3.0", max_delta=0.05, filter=True, low_bnd=25000)
    
    run = SingleAreaRun(conf)
    run.execute_run()
    # Define Sweep Ranges
    # sensitivities = [40, 200, 10]
    # lower_bounds = [1, 30000, 40]
    # upper_bounds = [73500, 73500, 1]
    
    # optimizer = FullAreaOptimizer(conf)
    # optimizer.run_optimization(sensitivities, lower_bounds, upper_bounds)
