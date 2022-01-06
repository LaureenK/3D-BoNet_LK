import os
import glob
import numpy as np
import random
import copy
from random import shuffle
import h5py

#NUM_POINTS = 2**14
NUM_POINTS = 2**14

def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

class Data_Configs:
    sem_names = ['person', 'dog', 'bicycle', 'sportsball']
    sem_ids = [0,1,2,3]

    points_cc = 3 #3?
    sem_num = len(sem_names)
    ins_max_num = 24 #48
    # train_pts_num = 16384
    # test_pts_num = 16384 

    train_pts_num = NUM_POINTS
    test_pts_num = NUM_POINTS
    batchsize = 2

class Data_S3DIS:
    def __init__(self, train_dataset_path, test_dataset_path, train_batch_size=8):
        # init__( dataset_path = './data_s3dis/', train_areas = ['Area_1'], test_areas = ['Area_5'], train_batch_size=4):

        self.train_files = glob.glob(os.path.join(train_dataset_path, "*.csv"))
        self.test_files = glob.glob(os.path.join(test_dataset_path, "*.csv"))
        print('train files:', len(self.train_files))
        print('test files:', len(self.test_files))

        

        self.ins_max_num = Data_Configs.ins_max_num
        self.train_batch_size = train_batch_size
        self.total_train_batch_num = len(self.train_files)//self.train_batch_size

        self.train_next_bat_index = 0

    # anstatt load_raw_data_file_s3dis_block
    @staticmethod
    def load_ascii_cloud_prepared(fname):
        points = []
        labels = []
        instances = []

        with open(fname, 'r') as fd:
            for line in fd.readlines():
                if "//" in line:
                    continue

                x, y, t, class_label, instance_label = line.strip().split(' ')
                x, y, t, class_label, instance_label = float(x), float(y), float(t), int(class_label), int(instance_label)

                points.append(np.array([x, y, t], dtype=np.float32))
                labels.append(class_label)
                instances.append(instance_label)

        npPoints = np.array(points, dtype=np.float32)
        npSeg = np.array(labels, dtype=np.uint8)
        npIns = np.array(instances, dtype=np.uint16)

        npPoints, npSeg, npIns = unison_shuffled_copies(npPoints, npSeg, npIns)

        npPoints = npPoints[0:NUM_POINTS,:]
        npSeg = npSeg[0:NUM_POINTS]
        npIns = npIns[0:NUM_POINTS]

        if len(npIns) != NUM_POINTS:
            raise ValueError("Wrong NUM_POINTS of cloud: ", fname)
 
        #pc
        #<type 'numpy.ndarray'>
        #(4096, 9)

        #npPoints
        #<type 'numpy.ndarray'>
        #(16384, 3)

        #sem_labels
        #<type 'numpy.ndarray'>
        #(4096,)

        #npSeg
        #<type 'numpy.ndarray'>
        #(16384,)

        #ins_labels
        #<type 'numpy.ndarray'>
        #(4096,)

        #npIns
        #<type 'numpy.ndarray'>
        #(16384,)

        # testarray1 = np.zeros_like(npPoints)
        # testarray2 = np.zeros_like(npPoints)
        # testarray3 = np.concatenate([testarray1, testarray2], axis=-1)
        #testarray1 = np.random.rand(npPoints.shape[0],npPoints.shape[1])
        #testarray2 = np.random.rand(npPoints.shape[0],npPoints.shape[1])
        #testarray3 = np.concatenate([testarray1, testarray2], axis=-1)
        #npPoints = np.concatenate([npPoints, testarray3], axis=-1)


        #print("npPoints")
        #print(npPoints.shape)

        return npPoints, npSeg, npIns

    @staticmethod
    def get_bbvert_pmask_labels(pc, ins_labels):
        gt_bbvert_padded = np.zeros((Data_Configs.ins_max_num, 2, 3), dtype=np.float32)
        gt_pmask = np.zeros((Data_Configs.ins_max_num, pc.shape[0]), dtype=np.float32)
        count = -1
        unique_ins_labels = np.unique(ins_labels)
        for ins_ind in unique_ins_labels:
            if ins_ind <= -1: continue
            count += 1
            if count >= Data_Configs.ins_max_num: print('ignored! more than max instances:', len(unique_ins_labels)); continue
        
            ins_labels_tp = np.zeros(ins_labels.shape, dtype=np.int8)
            ins_labels_tp[ins_labels == ins_ind] = 1
            ins_labels_tp = np.reshape(ins_labels_tp, [-1])
            gt_pmask[count,:] = ins_labels_tp
        
            ins_labels_tp_ind = np.argwhere(ins_labels_tp == 1)
            ins_labels_tp_ind = np.reshape(ins_labels_tp_ind, [-1])

            ###### bb min_xyz, max_xyz
            pc_xyz_tp = pc[:, 0:3]
            pc_xyz_tp = pc_xyz_tp[ins_labels_tp_ind]
            gt_bbvert_padded[count, 0, 0] = x_min = np.min(pc_xyz_tp[:, 0])
            gt_bbvert_padded[count, 0, 1] = y_min = np.min(pc_xyz_tp[:, 1])
            gt_bbvert_padded[count, 0, 2] = z_min = np.min(pc_xyz_tp[:, 2])
            gt_bbvert_padded[count, 1, 0] = x_max = np.max(pc_xyz_tp[:, 0])
            gt_bbvert_padded[count, 1, 1] = y_max = np.max(pc_xyz_tp[:, 1])
            gt_bbvert_padded[count, 1, 2] = z_max = np.max(pc_xyz_tp[:, 2])

        return gt_bbvert_padded, gt_pmask

    @staticmethod
    def load_fixed_points(file_path):
        pc_xyzrgb, sem_labels, ins_labels = Data_S3DIS.load_ascii_cloud_prepared(file_path)

        # ## center xy within the block
        # min_x = np.min(pc_xyzrgb[:,0]); max_x = np.max(pc_xyzrgb[:,0])
        # min_y = np.min(pc_xyzrgb[:,1]); max_y = np.max(pc_xyzrgb[:,1])
        # min_z = np.min(pc_xyzrgb[:,2]); max_z = np.max(pc_xyzrgb[:,2])

        # #print("min_x : ", min_x, " max_x: ", max_x)
        # #print("min_y : ", min_y, " max_y: ", max_y)
        # #print("min_z : ", min_z, " max_x: ", max_z)

        # ori_xyz = copy.deepcopy(pc_xyzrgb[:, 0:3])  # reserved for final visualization
        # use_zero_one_center = True
        # if use_zero_one_center:
        #     pc_xyzrgb[:, 0:1] = (pc_xyzrgb[:, 0:1] - min_x)/ np.maximum((max_x - min_x), 1e-3)
        #     pc_xyzrgb[:, 1:2] = (pc_xyzrgb[:, 1:2] - min_y)/ np.maximum((max_y - min_y), 1e-3)
        #     pc_xyzrgb[:, 2:3] = (pc_xyzrgb[:, 2:3] - min_z)/ np.maximum((max_z - min_z), 1e-3)

        # min_x = np.min(pc_xyzrgb[:,0]); max_x = np.max(pc_xyzrgb[:,0])
        # min_y = np.min(pc_xyzrgb[:,1]); max_y = np.max(pc_xyzrgb[:,1])
        # min_z = np.min(pc_xyzrgb[:,2]); max_z = np.max(pc_xyzrgb[:,2])

        # #print("2 min_x : ", min_x, " max_x: ", max_x)
        # #print("2 min_y : ", min_y, " max_y: ", max_y)
        # #print("2 min_z : ", min_z, " max_z: ", max_z)

        # pc_xyzrgb = np.concatenate([pc_xyzrgb, ori_xyz], axis=-1)

        #######
        sem_labels = sem_labels.reshape([-1])
        ins_labels = ins_labels.reshape([-1])
        
        bbvert_padded_labels, pmask_padded_labels = Data_S3DIS.get_bbvert_pmask_labels(pc_xyzrgb, ins_labels)

        psem_onehot_labels = np.zeros((pc_xyzrgb.shape[0], Data_Configs.sem_num), dtype=np.int8)
        for idx, s in enumerate(sem_labels):
            if sem_labels[idx]==-1: continue # invalid points
            sem_idx = Data_Configs.sem_ids.index(s)
            psem_onehot_labels[idx, sem_idx] =1

        return pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels

    def load_train_next_batch(self):
        bat_files = self.train_files[self.train_next_bat_index*self.train_batch_size:(self.train_next_bat_index+1)*self.train_batch_size]
        bat_pc=[]
        bat_sem_labels=[]
        bat_ins_labels=[]
        bat_psem_onehot_labels =[]
        bat_bbvert_padded_labels=[]
        bat_pmask_padded_labels =[]
        for file in bat_files:
            pc, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = Data_S3DIS.load_fixed_points(file)
            bat_pc.append(pc)
            bat_sem_labels.append(sem_labels)
            bat_ins_labels.append(ins_labels)
            bat_psem_onehot_labels.append(psem_onehot_labels)
            bat_bbvert_padded_labels.append(bbvert_padded_labels)
            bat_pmask_padded_labels.append(pmask_padded_labels)

        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

        self.train_next_bat_index+=1
        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels
    
    def load_test_next_batch_random(self):
        idx = random.sample(range(len(self.test_files)), self.train_batch_size)
        bat_pc=[]
        bat_sem_labels=[]
        bat_ins_labels=[]
        bat_psem_onehot_labels =[]
        bat_bbvert_padded_labels=[]
        bat_pmask_padded_labels =[]
        for i in idx:
            file = self.test_files[i]
            pc, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = Data_S3DIS.load_fixed_points(file)
            bat_pc.append(pc)
            bat_sem_labels.append(sem_labels)
            bat_ins_labels.append(ins_labels)
            bat_psem_onehot_labels.append(psem_onehot_labels)
            bat_bbvert_padded_labels.append(bbvert_padded_labels)
            bat_pmask_padded_labels.append(pmask_padded_labels)

        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels
    
    def load_test_next_batch_sq(self, bat_files):
        bat_pc=[]
        bat_sem_labels=[]
        bat_ins_labels=[]
        bat_psem_onehot_labels =[]
        bat_bbvert_padded_labels=[]
        bat_pmask_padded_labels =[]
        for file in bat_files:
            pc, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = Data_S3DIS.load_fixed_points(file)
            bat_pc += [pc]
            bat_sem_labels += [sem_labels]
            bat_ins_labels += [ins_labels]
            bat_psem_onehot_labels += [psem_onehot_labels]
            bat_bbvert_padded_labels += [bbvert_padded_labels]
            bat_pmask_padded_labels += [pmask_padded_labels]

        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels, bat_files
    
    def shuffle_train_files(self, ep):
        index = list(range(len(self.train_files)))
        random.seed(ep)
        shuffle(index)
        train_files_new=[]
        for i in index:
            train_files_new.append(self.train_files[i])
        self.train_files = train_files_new
        self.train_next_bat_index=0
        print('train files shuffled!')