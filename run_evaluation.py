import os
import csv
import argparse
from pathlib import Path

# evo imports
from evo.main_ape import ape
from evo.core import metrics
from evo.tools import file_interface, plot
import evo.core.sync as sync
import matplotlib.pyplot as plt

def find_trajectory_file(directory, traj_type: str):
    """Search for files and prioritize 'slam' over 'odom' and others."""
    patterns = ['*trajectory*', f'*{traj_type}*']
    all_found_files = []

    # 1. Collect all matching files
    for pattern in patterns:
        found = list(directory.glob(pattern))
        all_found_files.extend([f for f in found if f.is_file()])

    if not all_found_files:
        return None

    # 2. Define priority weights (lower number = higher priority)
    priority = {
        "slam": 1,
        "odom": 2,
        "trajectory": 3,
        "tum": 4,
        "kitti": 5
    }

    def get_priority(path):
        name = path.name.lower()
        # Assign priority based on the first keyword found in the filename
        for key, val in priority.items():
            if key in name:
                return val
        return 100 # Default for files that match a glob but not a specific keyword

    # 3. Sort by priority, then alphabetically for consistency
    all_found_files.sort(key=lambda x: (get_priority(x), x.name))

    return all_found_files[0]

def find_gt_traj( seq_name: str, ref_dir: Path):
    split_seq = seq_name.split('_')[-3]
    list_of_files = os.listdir(ref_dir)
    # Returns the first Path object where split_seq is in the filename
    gt_file = next((f for f in ref_dir.iterdir() if f.is_file() and split_seq in f.name), None)
    return gt_file
    
    

def run_evaluation():
    parser = argparse.ArgumentParser(description="Batch evaluate trajectories using evo.")
    parser.add_argument("input_dir", type=str, help="Directory containing subfolders of estimated trajectories")
    parser.add_argument("type", type=str, help="Type of trajectories kitti/tum")
    parser.add_argument("ref_dir", type=str, help="Directory containing ground truth (GT) files")
    parser.add_argument("output_dir", type=str, help="Directory to save plots and CSV results")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    traj_type = args.type
    ref_path = Path(args.ref_dir)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_results = []

    # Iterate through each subfolder in the input directory
    for subfolder in [f for f in input_path.iterdir() if f.is_dir()]:
        print(f"--- Processing: {subfolder.name} ---")
        
        if not os.path.exists(subfolder/'time_details.png'):
            print(f"Skipping {subfolder.name}: the directory is not processed.")
            continue
        
        
        # 1. Find Est Trajectory
        est_file = find_trajectory_file(subfolder,traj_type)
        
        # 2. Find matching GT file in Ref Dir (matching by subfolder name)
        # Assuming GT files are named similar to the subfolder (e.g., 'sequence_01.txt')

        gt_file = find_gt_traj(est_file.parts[-2], ref_path)

        if not est_file or not gt_file:
            print(f"Skipping {subfolder.name}: Missing file(s).")
            continue

        try:
            # 3. Load Trajectories (Auto-detecting format is best handled by file_interface)
            # We assume TUM format here; change to kitti if needed
            
            if traj_type == "kitti":
                traj_est = file_interface.read_kitti_poses_file(str(est_file))
                traj_ref = file_interface.read_kitti_poses_file(str(gt_file))
            elif traj_type == "tum":
                traj_est = file_interface.read_tum_trajectory_file(str(est_file))
                traj_ref = file_interface.read_tum_trajectory_file(str(gt_file))
                if traj_est.timestamps[0]==0:
                    traj_ref.timestamps = traj_ref.timestamps - traj_ref.timestamps[0]

            else:
                print(f"Unsupported trajectory type: {type}. Skipping evaluation.")
                exit(-1)

            traj_ref_sync, traj_est_sync = sync.associate_trajectories(traj_ref, traj_est)

            traj_est_sync.align(traj_ref_sync, correct_scale=True, correct_only_scale=False)

            # 4. Create Output Sub-directory
            sub_out_dir = out_path / subfolder.name
            sub_out_dir.mkdir(exist_ok=True)

            # 5. Run APE with Alignment
            # PoseRelation.translation_part is the standard for APE RMSE
            result = ape(traj_ref_sync, traj_est_sync,
                         pose_relation=metrics.PoseRelation.translation_part, 
                         est_name=subfolder.name, 
                         align=False, 
                         correct_scale=False) # Set to False if not Monocular

            # 6. Save Results to Summary
            stats = result.stats
            stats['name'] = subfolder.name
            summary_results.append(stats)

            # 7. Plot and Save
            plot_mode = plot.PlotMode.xy
            fig = plt.figure(figsize=(8, 8))
            ax = plot.prepare_axis(fig, plot_mode)
            plot.traj(ax, plot_mode, traj_ref_sync, style="--", label="GT")
            plot.traj(ax, plot_mode, traj_est_sync, label="Estimate")
            
            fig.savefig(sub_out_dir / "trajectory_plot.png")
            plt.close(fig)

        except Exception as e:
            print(f"Error processing {subfolder.name}: {e}")

    # 8. Write CSV Summary
    if summary_results:
        keys = summary_results[0].keys()
        with open(out_path / "summary_results.csv", 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(summary_results)
        print(f"\nDone! Summary saved to {out_path / 'summary_results.csv'}")

if __name__ == "__main__":
    run_evaluation()