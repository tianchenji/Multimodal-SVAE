import os
import torch
import numpy as np
from torch.utils.data import Dataset

class TerraDataset(Dataset):
    def __init__(self, data_root, clip_thres, test_flag):
        self.samples        = []
        self.num_normal     = 0
        self.num_untvbl_obs = 0
        self.num_tvbl_obs   = 0
        self.num_crash      = 0
        self.num_undefined  = 0
        self.data_root      = data_root
        self.clip_thres     = clip_thres
        self.test_flag      = test_flag

        self.read_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lidar_data  = torch.from_numpy(self.samples[idx][0]).float()
        obs_data    = torch.from_numpy(self.samples[idx][1]).float()
        system_data = torch.from_numpy(self.samples[idx][2]).float()
        label_data  = self.samples[idx][3]
        return (lidar_data, obs_data, system_data, label_data)

    def read_data(self):
        '''
        We construct a datapoint as (lidar_data, obs_data, system_data, label_data), where
        lidar_data              - high dimensional input x_h
        (obs_data, system_data) - low dimensional input x_l, defined by equation (6) in the paper
        label_data              - ground truth label y
        '''

        left_enc_v_index    = 40
        right_enc_v_index   = 41
        label_index         = -1
        count               = 0

        np.random.seed(0)

        map_float = lambda x: np.array(list(map(float, x)))

        lidar = os.listdir(self.data_root)[0]
        system = os.listdir(self.data_root)[1]

        lidar_folder = os.path.join(self.data_root, lidar)
        system_folder = os.path.join(self.data_root, system)

        for flidar, fsystem in zip(os.listdir(lidar_folder), os.listdir(system_folder)):
            flidar_path  = os.path.join(lidar_folder, flidar)
            fsystem_path = os.path.join(system_folder, fsystem)

            fsystem_len = self.file_len(fsystem_path)

            with open(flidar_path, 'r') as file_lid, open(fsystem_path, 'r') as file_sys:
                for i in range(fsystem_len):
                    lid_line  = file_lid.readline()
                    dist      = lid_line.split(',')
                    dist      = map_float(dist)[1:-1]
                    clip_dist = np.clip(dist, a_min=0, a_max=self.clip_thres)/self.clip_thres
                    obs_flag  = self.detect_obstacles(dist)

                    sys_line  = file_sys.readline()
                    sys_data  = sys_line.split(',')
                    enc_left  = float(sys_data[left_enc_v_index])
                    enc_right = float(sys_data[right_enc_v_index])
                    label     = int(sys_data[label_index])
                    encoders  = np.array([enc_left, enc_right])

                    # under-sampling normal cases and over-sampling anomalies by replicating
                    if label == 0:
                        if self.test_flag == 0:
                            if count % 7 == 0:
                                self.num_normal += 1
                                self.samples.append([clip_dist, obs_flag, encoders, label])
                            count += 1
                        else:
                            self.num_normal += 1
                            self.samples.append([clip_dist, obs_flag, encoders, label])
                    elif label == 1:
                        if self.test_flag == 0:
                            self.num_untvbl_obs += 1
                            self.samples.append([clip_dist, obs_flag, encoders, label])
                            for j in range(2):
                                self.num_untvbl_obs += 1
                                clip_dist_new, obs_flag_new, encoders_new = self.data_augmentation(
                                                                     clip_dist, obs_flag, encoders)
                                self.samples.append([clip_dist_new, obs_flag_new, encoders_new, label])
                        else:
                            self.num_untvbl_obs += 1
                            self.samples.append([clip_dist, obs_flag, encoders, label])
                    elif label == 2:
                        if self.test_flag == 0:
                            self.num_tvbl_obs += 1
                            self.samples.append([clip_dist, obs_flag, encoders, label])
                            for j in range(1):
                                self.num_tvbl_obs += 1
                                clip_dist_new, obs_flag_new, encoders_new = self.data_augmentation(
                                                                     clip_dist, obs_flag, encoders)
                                self.samples.append([clip_dist_new, obs_flag_new, encoders_new, label])
                        else:
                            self.num_tvbl_obs += 1
                            self.samples.append([clip_dist, obs_flag, encoders, label])
                    elif label == 3:
                        if self.test_flag == 0:
                            self.num_crash += 1
                            self.samples.append([clip_dist, obs_flag, encoders, label])
                            for j in range(1):
                                self.num_crash += 1
                                clip_dist_new, obs_flag_new, encoders_new = self.data_augmentation(
                                                                     clip_dist, obs_flag, encoders)
                                self.samples.append([clip_dist_new, obs_flag_new, encoders_new, label])
                        else:
                            self.num_crash += 1
                            self.samples.append([clip_dist, obs_flag, encoders, label])
                    else:
                        pass

                    for j in range(7):
                        lid_line  = file_lid.readline()

    def detect_obstacles(self, distance):
        
        obs_flag = np.array([0] * 4)
        local_distance = np.clip(distance, a_min=0, a_max=250)/250
        obs_flag[0] = local_distance[420:480].mean()
        obs_flag[1] = local_distance[480:540].mean()
        obs_flag[2] = local_distance[540:600].mean()
        obs_flag[3] = local_distance[600:660].mean()

        return obs_flag

    def data_augmentation(self, clip_dist, obs_flag, encoders):
        '''
        augment training data with additive Gaussian noise
        '''

        clip_dist_new = clip_dist + 0.1 * clip_dist * np.random.randn(1080)
        clip_dist_new = np.clip(clip_dist_new, a_min=0, a_max=1)

        obs_flag_new  = obs_flag + 0.4 * obs_flag * np.random.randn(4)
        obs_flag_new  = np.clip(obs_flag_new, a_min=0, a_max=1)

        encoders_new  = encoders + 0.0 * encoders * np.random.randn(2)

        return clip_dist_new, obs_flag_new, encoders_new

    def get_labels_num(self):
        print("The number of normal cases:            {:d}".format(self.num_normal))
        print("The number of untraversable obstacles: {:d}".format(self.num_untvbl_obs))
        print("The number of traversable obstacles:   {:d}".format(self.num_tvbl_obs))
        print("The number of crashes:                 {:d}".format(self.num_crash))
        print("The number of undefined points:        {:d}".format(self.num_undefined))

    def file_len(self, fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1