# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
# 2024 Yue Pan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to mse, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE msE OR OTHER DEALINGS IN THE
# SOFTWARE.
import glob
import importlib
import os
import sys
import yaml

import pathlib
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

from pointcloud_utils import read_pcd


# https://www.cvlibs.net/datasets/kitti/setup.php

class SpideySenseDataset:
    def __init__(self, data_dir, sequence: str, *args, **kwargs):
        
        pc_dir = data_dir.parts[-1]
        self.spidey_sense_sequences_dir = os.path.join(*data_dir.parts[:-1])

        
        self.velodyne_dir = os.path.join(self.spidey_sense_sequences_dir, f"{pc_dir}/")

        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
        scan_times = [float(os.path.basename(f).split('.')[0])*1e-9 for f in self.scan_files]
        scan_count = len(self.scan_files)

        # point cloud semantic labels (from semantic kitti) # TODO
        self.sem_labels_dir = None
        self.sem_files = None
        sem_file_count = 0
        if sem_file_count == scan_count:
            self.sem_available = True
        else:
            self.sem_available = False

        self.load_img: bool = False
        self.use_only_colorized_points: bool = True

        # cam 0 (front)
        self.img0_dir = os.path.join(self.spidey_sense_sequences_dir, "images",'cam0/')
        self.img0_files = sorted(glob.glob(self.img0_dir + "*.jpg"))
        img0_count = len(self.img0_files)
        
        # ASB TODO. Camera is running at 10Hz, so will have double the number of images
        if img0_count == scan_count:
            self.image_available = True
            try:
                self.cv2 = importlib.import_module("cv2")
            except ModuleNotFoundError: # TODO
                print(
                    'img files requires opencv-python and is not installed on your system run "pip install opencv-python"'
                )
                sys.exit(1)
        else:
            self.image_available = False

        # cam 1 (left)
        self.img1_dir = os.path.join(self.spidey_sense_sequences_dir, "images",'cam1/')
        self.img1_files = sorted(glob.glob(self.img1_dir + "*.jpg"))
        
        self.img2_dir = os.path.join(self.spidey_sense_sequences_dir, "images",'cam2/')
        self.img2_files = sorted(glob.glob(self.img2_dir + "*.jpg"))


        self.calibration = self.read_calib_file(kwargs['calib_file_path'])

        self._load_calib() # load all calib first

        # Read imu messges
        try:
            self.inertialmsgs = self.load_imu_data(kwargs['imu_path'])
            self.inertial_intrinsics = self.load_inertial_intrinsics(kwargs['imu_intrinsic_path'])
        except:
            self.inertialmsgs = None
            self.inertial_intrinsics = None
            print("Tried to load IMU data. Continuing without it")

        # Load GT Poses (if available)
        try:
            self.poses_fn = kwargs['gt_pose_path']
            if os.path.exists(self.poses_fn):
                [pose_times, gt_poses] = self.load_poses(self.poses_fn)
                self.gt_poses = np.asarray(gt_poses)
                mask = np.isin(scan_times,pose_times)
                scan_files_array = np.array(self.scan_files)
                self.scan_files = scan_files_array[mask].tolist()
        except:
            print("Tried to load GT poses. Continuing without it")

    def __getitem__(self, idx):
        
        points = self.scans(idx)
        if self.scan_time.min() == 0 and self.scan_time.max()== 0:
            self.scan_time = self.get_timestamps(points)

        point_ts = self.scan_time
        scan_time = float(pathlib.PurePath(self.scan_files[idx].split('/')[-1]).stem)

        if idx != 0 and self.inertialmsgs!= None:
            prev_scan_time = float(pathlib.PurePath(self.scan_files[idx - 1].split('/')[-1]).stem)
            currentmask = (self.inertialmsgs['timestamps'] >= prev_scan_time) & (self.inertialmsgs['timestamps'] <= scan_time)
            filtered_accel = self.inertialmsgs['accels'][currentmask]
            filtered_gyro = self.inertialmsgs['gyros'][currentmask]
            filtered_ts = self.inertialmsgs['timestamps'][currentmask]
            imu = {
                'timestamps': filtered_ts,
                'accel': filtered_accel,
                'gyro': filtered_gyro,
                'intrinsics': {'accel': {'noise_density': self.inertial_intrinsics['accelerometer_noise_density'],
                                         'random_walk': self.inertial_intrinsics['accelerometer_random_walk']},
                                'gyro': {'noise_density': self.inertial_intrinsics['gyroscope_noise_density'],
                                        'random_walk': self.inertial_intrinsics['gyroscope_random_walk']}}
            }   
        #read imu messages after the previous scan untill the current scan  #
        else:
            imu=None


        if self.image_available and self.load_img:
            img = self.read_img(self.img2_files[idx])
            img_dict = {self.left_cam_name: img}

            points_rgb = np.ones_like(points)

            # project to the image plane to get the corresponding color
            points_rgb = self.project_points_to_cam(points, points_rgb, img, self.T_c_l_mats[self.left_cam_name], self.K_mats[self.left_cam_name])

            if self.use_only_colorized_points:
                with_rgb_mask = (points_rgb[:, 3] == 0)
                points = points[with_rgb_mask]
                points_rgb = points_rgb[with_rgb_mask]
                point_ts = point_ts[with_rgb_mask]

            # we skip the intensity here for now (and also the color mask)
            points = np.hstack((points[:,:3], points_rgb[:,:3]))

            frame_data = {"points": points, "point_ts": point_ts, "img": img_dict, "scan_time": scan_time, "imus": imu}
        else:
            frame_data = {"points": points, "point_ts": point_ts, "scan_time": scan_time,  "imus": imu}

        return frame_data

    def __len__(self):
        return len(self.scan_files)

    def scans(self, idx):
        return self.read_point_cloud(self.scan_files[idx])

    def apply_calibration(self, poses: np.ndarray) -> np.ndarray:
        """Converts from Velodyne to Camera Frame"""
        Tr = np.eye(4, dtype=np.float64)
        Tr[:3, :4] = self.calibration["Tr"].reshape(3, 4)
        return Tr @ poses @ np.linalg.inv(Tr)

    def read_point_cloud(self, scan_file: str):
        data = np.fromfile(scan_file,dtype=np.float32).reshape(-1,6)
        points = data[:,:4]
        self.scan_intensity = data[:,3]
        self.scan_time = data[:,-1]
        return points # N, 4
    
    def read_img(self, img_file: str):
        img = self.cv2.imread(img_file)
        img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        return img
    
    # velodyne lidar
    @staticmethod
    def get_timestamps(points, rotation_hz=10.0):
        """
        Args:
            points: (N, 3) array [x, y, z]
            scan_start_time: Unix timestamp of the start of rotation
            rotation_hz: Spin rate (10Hz)
        """
        x = points[:, 0]
        y = points[:, 1]
        
        # Calculate yaw in radians
        yaw = np.arctan2(y, x) 
        
        # Map [-pi, pi] to [0, 1] for the rotation cycle
        # We use (yaw % (2 * np.pi)) to ensure a clean 0 to 2pi range
        norm_yaw = (yaw + np.pi) / (2 * np.pi)
        
        # Duration of one full sweep (0.1s for 10Hz)
        sweep_duration = 1.0 / rotation_hz
        
        # Per-point offset
        time_offsets = norm_yaw * sweep_duration
    
        return time_offsets

    def load_poses(self, poses_file):
        def _lidar_pose_gt(poses_gt):
           times = poses_gt[0]
           rot = R.from_quat(poses_gt[4:])
           t = poses_gt[1:4]
           tr = np.eye(4, dtype=np.float32)
           tr[:3,:3] = rot.as_matrix()
           tr[:3, 3] = t
           return [times, tr]

        poses = np.genfromtxt(poses_file,dtype=np.float64, delimiter=" ")
        poses_gt = []
        times = []
        n = poses.shape[0]
        for each in poses:
            [scan_time, gt_pose] = _lidar_pose_gt(each)
            poses_gt.append(gt_pose)  # [N, 4, 4]
            times.append(scan_time)
        return [times, poses_gt]

    def get_frames_timestamps(self) -> np.ndarray:
        timestamps = np.loadtxt(os.path.join(self.spidey_sense_sequences_dir, "times.txt")).reshape(-1, 1)
        return timestamps

    @staticmethod
    def read_calib_file(file_path: str) -> dict:
        calib_dict = {}
        try:
            with open(file_path, "r") as calib_file:
                calib_dict = yaml.safe_load(calib_file)
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            # Handle error appropriately (e.g., raise an exception or return an empty dict)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML in '{file_path}': {exc}")
            # Handle error appropriately
            
        return calib_dict
    
    # partially from kitti360 dev kit
    def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        
        # points as np.numpy (N,4)
        points[:,3] = 1 # homo coordinate

        # transfrom velodyne points to camera coordinate
        points_cam = np.matmul(T_c_l, points.T).T # N, 4
        points_cam = points_cam[:,:3] # N, 3

        # project to image space
        u, v, depth= self.persepective_cam2image(points_cam.T, K_mat) 
        u = u.astype(np.int32)
        v = v.astype(np.int32)

        img_height, img_width, _ = np.shape(img)

        # prepare depth map for visualization
        depth_map = np.zeros((img_height, img_width))
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<img_width), v>=0), v<img_height)
        # depth mask
        min_depth = 0.5
        max_depth = 100.0
        mask = np.logical_and(np.logical_and(mask, depth>min_depth), depth<max_depth)
        
        v_valid = v[mask]
        u_valid = u[mask]

        depth_map[v_valid,u_valid] = depth[mask]

        points_rgb[mask, :3] = img[v_valid,u_valid].astype(np.float64)/255.0 # 0-1
        points_rgb[mask, 3] = 0 # has color

        return points_rgb
    
    def persepective_cam2image(self, points, K_mat):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(K_mat[:3,:3].reshape([1,3,3]), points)
        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(int)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(int)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth
    
    # from pykitti
    def _load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later

        calib_data = self.calibration
        
        # deal with camera 0
        self.T_I_cam1  = np.asarray(calib_data['T_xsens_cam']['data']).reshape(calib_data['T_xsens_cam']['rows'], calib_data['T_xsens_cam']['cols'])

        self.T_cam1_I = np.asarray(calib_data['T_cam_xsens']['data']).reshape(calib_data['T_cam_xsens']['rows'], calib_data['T_cam_xsens']['cols'])
         
        self.T_cam1_L = np.asarray(calib_data['T_cam_lidar']['data']).reshape(calib_data['T_cam_lidar']['rows'], calib_data['T_cam_lidar']['cols'])

        self.T_L_cam1 = np.asarray(calib_data['T_lidar_cam']['data']).reshape(calib_data['T_lidar_cam']['rows'], calib_data['T_lidar_cam']['cols'])
        
        
        self.T_I_L = np.asarray(calib_data['T_xsens_lidar']['data']).reshape(calib_data['T_xsens_lidar']['rows'], calib_data['T_xsens_lidar']['cols'])
        self.T_L_I = np.asarray(calib_data['T_lidar_xsens']['data']).reshape(calib_data['T_lidar_xsens']['rows'], calib_data['T_lidar_xsens']['cols'])

    def load_imu_data(self, imu_path):
            '''
            Docstring for load_imu_data
            
            :param self: Description
            :param imu_path: Description

            inertialmsgs = []
            with open(imu_path, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # skip header
                    parts = line.strip().split(',')
                    timestamp = float(parts[0]) + float(parts[1]) * 1e-9
                    #timestamp['sec'] = float(parts[0])
                    #timestamp['nsec'] = float(parts[1]) * 1e-9 # seconds
                    accel = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                    gyro = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
                    inertialmsgs.append({'timestamp': timestamp, 'accel': accel, 'gyro': gyro})
            return inertialmsgs
            '''
            data = np.loadtxt(imu_path, delimiter=',', skiprows=1,dtype=np.float64)
            timestamps = data[:, 0] * 1e-9  # seconds
            accels = data[:, -3:]
            gyros = data[:, 5:8]
            return{
                'timestamps': timestamps,
                'accels': accels,
                'gyros': gyros
            }

            

    def load_inertial_intrinsics(self, imu_intrinsic_path):
            with open(imu_intrinsic_path, 'r') as f:
                intrinsics = yaml.safe_load(f)
            return intrinsics