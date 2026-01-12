import os
import itertools
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from threading import Thread

# Core Module Imports
from vert_ana import analyze_edge, analyze_beer_foam
from curvature_analysis import calculate_average_curvature_closed
from vertical_area_plot import plot_curvature_fit

@dataclass
class OptimizerConfig:
    study_name: str = "Drainage_Optimization"
    video_path: str = "../Videos/Final_Cropped.MOV"
    base_output_dir: str = "../Data"
    
    # Frames
    low_bnd: int = 300
    high_bnd: int = 1500
    
    # Ranges: (min, max, steps)
    custom_thresh_range: tuple = (100, 130, 4)
    survival_thresh_range: tuple = (200, 600, 3)
    size_thresh_range: tuple = (1000, 2500, 2)
    max_k_range: tuple = (1.0, 2.5, 3)
    
    check_validity: bool = True
    show_results: bool = True
    cleanup_temp: bool = True  # Fixed: Now present in config
    
    # Internal state
    _validity_checked: bool = False

class SingleRun:
    def __init__(self, cfg: OptimizerConfig):
        self.cfg = cfg

    def _fix_and_pad_csv(self, csv_path):
        """Standardizes CSV to start exactly at low_bnd and end at high_bnd."""
        df = pd.read_csv(csv_path)
        df.rename(columns={df.columns[0]: 'frame'}, inplace=True)
        
        master_frames = pd.DataFrame({'frame': range(self.cfg.low_bnd, self.cfg.high_bnd + 1)})
        df_fixed = pd.merge(master_frames, df, on='frame', how='left')
        
        # Fill missing area values (index 4)
        area_col = df_fixed.columns[4]
        df_fixed[area_col] = df_fixed[area_col].ffill().fillna(0)
        
        df_fixed.to_csv(csv_path, index=False)
        return df_fixed

    def execute(self, ct, st, zt, kt, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)
        area_csv = run_dir / f"area_thresh_{ct}.csv"
        edge_dir = run_dir / "Edges"
        edge_dir.mkdir(exist_ok=True)
        plot_path = run_dir / "fit_visualization.png"

        # 1. Area Analysis + Padding
        analyze_beer_foam(self.cfg.video_path, str(area_csv), custom_threshold=ct)
        df_valid = self._fix_and_pad_csv(str(area_csv))
        
        if self.cfg.check_validity and not self.cfg._validity_checked:
            self.show_validity(df_valid)

        # 2. Edge Tracking
        analyze_edge(self.cfg.video_path, str(edge_dir), custom_threshold=ct, 
                     survival_threshold=st, size_threshold=zt, 
                     low_bnd=self.cfg.low_bnd, high_bnd=self.cfg.high_bnd)
        
        edge_files = list(edge_dir.glob("*.csv"))
        if not edge_files:
            return None 
        
        # 3. Curvature
        best_edge = max(edge_files, key=lambda f: f.stat().st_size)
        curv_csv = run_dir / "curv.csv"
        calculate_average_curvature_closed(str(best_edge), str(curv_csv), kt)
        
        # 4. Fit & Save Plot at DPI=400
        # Passing 'target' and 'save=True' to your plotting function
        return plot_curvature_fit(
            str(area_csv), 
            str(curv_csv), 
            self.cfg.low_bnd, 
            self.cfg.high_bnd, 
            save=True, 
            target=str(plot_path)
            # Ensure your plot_curvature_fit uses plt.savefig(target, dpi=400)
        )

    def show_validity(self, df):
        plt.figure(figsize=(8,3))
        plt.plot(df['frame'], df.iloc[:, 4], label="Foam Area")
        plt.axvspan(self.cfg.low_bnd, self.cfg.high_bnd, color='green', alpha=0.1)
        plt.title(f"Area Check: Frames {self.cfg.low_bnd}-{self.cfg.high_bnd}")
        plt.show()


class FoamOptimizer:
    def __init__(self, cfg: OptimizerConfig):
        self.cfg = cfg
        self.runner = SingleRun(cfg)

    def run_optimization(self):
        root = Path(self.cfg.base_output_dir) / f"{self.cfg.study_name}_Opt"
        runs_dir = root / "Temp_Runs"
        
        # 1. Generate Parameter Arrays
        cts = np.linspace(*self.cfg.custom_thresh_range, dtype=int)
        sts = np.linspace(*self.cfg.survival_thresh_range, dtype=int)
        zts = np.linspace(*self.cfg.size_thresh_range, dtype=int)
        kts = np.linspace(*self.cfg.max_k_range)
        
        # 2. Calculate Total Combinations
        total_combos = len(cts) * len(sts) * len(zts) * len(kts)
        current_iteration = 0
        
        best_r2 = -np.inf
        results = []
        winner_dir = None

        print(f"\n{'='*60}")
        print(f"STARTING GRID SEARCH: {total_combos} combinations")
        print(f"{'='*60}\n")

        for ct in cts:
            # Show validity plot once for each unique area threshold
            self.cfg._validity_checked = False
            
            for st, zt, kt in itertools.product(sts, zts, kts):
                current_iteration += 1
                run_id = f"CT{ct}_ST{st}_ZT{zt}_K{kt:.1f}".replace(".", "p")
                
                # PROGRESS PRINT
                print(f"[{current_iteration}/{total_combos}] Processing: {run_id}...", end="\r")
                
                current_run_dir = runs_dir / run_id
                res = self.runner.execute(ct, st, zt, kt, current_run_dir)
                
                self.cfg._validity_checked = True 
                
                if res:
                    r2, alpha = res[3], res[4]
                    results.append({
                        "CT": ct, "ST": st, "ZT": zt, "KT": kt, 
                        "R2": r2, "Alpha": alpha, "Dir": current_run_dir
                    })
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        winner_dir = current_run_dir
                        # Clean print for new leader so it doesn't get overwritten by progress line
                        print(f"\n  >>> NEW LEADER FOUND: R² = {r2:.4f} (Combo {current_iteration})")
                else:
                    if self.cfg.cleanup_temp and current_run_dir.exists():
                        shutil.rmtree(current_run_dir)

        print(f"\n\n{'='*60}")
        print(f"GRID SEARCH COMPLETE")
        
        if not results:
            print("No valid fits found. Check thresholds or frame alignment.")
            return

        df_res = pd.DataFrame(results)
        df_res.to_csv(root / "optimization_summary.csv", index=False)

        if self.cfg.cleanup_temp:
            print("Cleanup: Preserving winning run and deleting others...")
            for d in runs_dir.iterdir():
                if d.is_dir() and d != winner_dir:
                    shutil.rmtree(d)

        if self.cfg.show_results:
            print("\n--- Summary ---")
            print(df_res.drop(columns=['Dir']).sort_values("R2", ascending=False).head(10))

@dataclass
class OptimizerConfig:
    study_name: str = "Test1"
    video_path: str = "../Videos/Final_Cropped.MOV"
    base_output_dir: str = "../Data"
    low_bnd: int = 1200
    high_bnd: int = 2500
    custom_thresh_range: tuple = (100, 100, 1)
    survival_thresh_range: tuple = (100, 1000, 5)
    size_thresh_range: tuple = (100, 2500, 5)
    max_k_range: tuple = (0.1, 2.5, 5)
    check_validity: bool = False
    show_results: bool = False
    cleanup_temp: bool = True
    

if __name__ == "__main__":

    cfg1 = OptimizerConfig(
        study_name = "Test1",
        video_path= "../Videos/Final_Cropped.MOV",
        base_output_dir = "../Data",
        low_bnd = 1200,
        high_bnd = 2500,
        custom_thresh_range = (120, 140, 3),
        survival_thresh_range = (100, 1000, 5),
        size_thresh_range = (100, 2500, 5),
        max_k_range = (0.01, 0,5, 5),
        check_validity = False,
        show_results = False,
        cleanup_temp = True
    )
    opt1 = FoamOptimizer(cfg1)
    opt1.run_optimization()