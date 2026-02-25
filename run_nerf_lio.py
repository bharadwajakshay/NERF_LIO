import os
import sys
import yaml
import subprocess
import shlex
from typing import List, Union
import shutil
import re

#kitti_base_config = "/src/NERF_LIO/config/lidar_slam/run_kitti.yaml"
kitti_base_config = "/src/NERF_LIO/config/lidar_slam/run_kitti_downsample.yaml"
#oxford_spires_config = "/src/NERF_LIO/config/lidar_slam/run_oxford_spires.yaml"
oxford_spires_config = "/src/NERF_LIO/config/lidar_slam/run_oxford_spires_w_intertial.yaml"
#spidey_sense_config = "/src/NERF_LIO/config/lidar_slam/run_spidey_sense.yaml"
spidey_sense_config = "/src/NERF_LIO/config/lidar_slam/run_spidey_sense_w_intertial.yaml"
calib_txt_file ="calib.txt"
#pc_dir = "velodyne"
pc_dir = "downsampled_velodyne"

working_config = "/src/NERF_LIO/config/lidar_slam/run_curr.yaml"

def run_python_script(script_path: str, args: List[str] = None) -> subprocess.CompletedProcess:
    """
    Executes a Python script as a subprocess via the bash shell.
    
    Args:
        script_path (str): The path to the .py file to execute.
        args (List[str]): A list of arguments to pass to the script.
    
    Returns:
        subprocess.CompletedProcess: The result object containing return code, stdout, and stderr.
    """
    if args is None:
        args = []

    # Construct the full command
    # We use sys.executable to ensure we use the same Python interpreter 
    # running this script (e.g., venv friendly)
    command_line = [sys.executable, script_path] + args

    try:
        # subprocess.run is the recommended approach for modern Python (3.5+)
        # capture_output=True allows us to get stdout/stderr as strings
        # text=True decodes bytes to strings automatically
        print(f"Executing command: {' '.join(shlex.quote(arg) for arg in command_line)}")
        result = subprocess.run(
            command_line,
            #capture_output=True,
            text=True,
            check=True # Raises CalledProcessError if exit code is non-zero
        )
        return result

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running script '{script_path}':")
        print(f"Exit Code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
        return e
    except FileNotFoundError:
        print(f"Error: The file '{script_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
    

def run_nerf_lio(dataset_path):
    for each_dataset in dataset_path:
        if "kitti" in each_dataset:
            print("Processing KITTI dataset")
            # Get sequences 
            sequences_dir = os.path.join(each_dataset,"sequences")
            poses_dir = os.path.join(each_dataset,"poses")
            
            # Run all the data processing on the sequences where GT is available
            poses_path = os.listdir(poses_dir)
            for pose_path in poses_path:
                if os.path.isfile(os.path.join(poses_dir, pose_path)):
                    with open(kitti_base_config) as yamlstream:
                        yaml_data = yaml.safe_load(yamlstream)
                        
                    seq_no = pose_path.split('.')[0]
                    sequence_path = f'{os.path.join(sequences_dir,seq_no,pc_dir)}'
                    pose_filepath = f'{os.path.join(poses_dir,pose_path)}'
                    calib_filepath = f'{os.path.join(sequences_dir,seq_no,calib_txt_file)}'
                    name = f'"kitti_seq_{seq_no}"'
                    
                
                    yaml_data['setting']['name'] = f'kitti_downsample_seq_{seq_no}'
                    yaml_data['setting']['pc_path'] = sequence_path
                    yaml_data['setting']['pose_path'] = pose_filepath
                    yaml_data['setting']['calib_path'] = calib_filepath
                    yaml_data['eval']['o3d_vis_on'] = False
                    

                    
                    with open(working_config,'w') as out_yaml:
                        yaml.dump(yaml_data,out_yaml,default_flow_style=False)
                        
                    command_script = "./nerf_lio.py"
                    #args = [working_config, "kitti", seq_no, "-i", each_dataset, "-dsm"]
                    args = [working_config, "kitti", seq_no, "-dsm"]
                    
                    results = run_python_script(command_script, args)
                    #results = run_python_script_popen(command_script, args)
                    print('Breakpoint')
                    
                else:
                    continue
                
            
        elif "oxford-spires" in each_dataset:
            print("Processing Oxford-Spires dataset")

            # Get sequences 
            sequences_dir = os.path.join(each_dataset,"sdk/data/sequences")
            sequence_folders = os.listdir(sequences_dir)
            for seq_folder in sequence_folders:
                
                sequence_path = os.path.join(sequences_dir,seq_folder)
                if not os.path.isdir(sequence_path):
                    continue
                
                pc_path = os.path.join(sequence_path,"raw/lidar-clouds/lidar-clouds")
                name = f'"oxford_spires_{seq_folder}"'
                
                with open(oxford_spires_config) as yamlstream:
                    yaml_data = yaml.safe_load(yamlstream)
                    
                yaml_data['setting']['name'] = name
                yaml_data['setting']['pc_path'] = pc_path
                yaml_data['eval']['o3d_vis_on'] = False
                
                with open(working_config,'w') as out_yaml:
                    yaml.dump(yaml_data,out_yaml,default_flow_style=False)
                    
                command_script = "./nerf_lio.py"
                args = [working_config, "oxford_spires", seq_folder, "-dsm"]
                
                results = run_python_script(command_script, args)
                #results = run_python_script_popen(command_script, args)
                print('Breakpoint')

        elif "spidey-sense" in each_dataset:
            print("Processing Spidey Sense dataset")

            # Get sequences 
            sequences_dir = each_dataset
            sequence_folders = os.listdir(sequences_dir)
            for seq_folder in sequence_folders:
                
                sequence_path = os.path.join(sequences_dir,seq_folder)
                if not os.path.isdir(sequence_path):
                    continue
                
                pc_path = os.path.join(sequence_path,"pointclouds")
                imu_data_path = os.path.join(sequence_path,f"{seq_folder}_imu_data.csv")
                name = f'"spidey-spense-{seq_folder}"'
                
                with open(spidey_sense_config) as yamlstream:
                    yaml_data = yaml.safe_load(yamlstream)
                    
                yaml_data['setting']['name'] = name
                yaml_data['setting']['pc_path'] = pc_path
                yaml_data['setting']['imu_path'] = imu_data_path
                yaml_data['eval']['o3d_vis_on'] = False

                # Check for incomplete runs
                outputdir = yaml_data['setting']['output_root']
                
                if  os.path.exists(outputdir):
                    exp_dirs = os.listdir(outputdir)

                    skip_this_dir = False

                    for each_dir in exp_dirs:
                        if name in each_dir:
                            # check the contents of the dir 
                            dir_contents = os.listdir(os.path.join(outputdir,each_dir))

                            if any("slam_poses" in s for s in dir_contents):
                                print("THis sequence has already been processed. Skipping....")
                                skip_this_dir = True
                                break

                    if skip_this_dir:
                        continue
                
                with open(working_config,'w') as out_yaml:
                    yaml.dump(yaml_data,out_yaml,default_flow_style=False)
                    
                command_script = "./nerf_lio.py"
                args = [working_config, "spidey_sense", seq_folder, "-dsm"]
                
                results = run_python_script(command_script, args)
                #results = run_python_script_popen(command_script, args)
                print('Breakpoint')
        else:
            print(f"No known dataset was found at {each_dataset}")
    
        

if __name__=='__main__':
    run_nerf_lio(sys.argv[1:])