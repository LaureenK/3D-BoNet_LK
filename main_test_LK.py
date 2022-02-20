import numpy as np
import scipy.stats
import os
import scipy.io
import tensorflow as tf
import glob
import h5py
import helper_data_dvs

MODELPATH = "/home/klein/bonet/results/run4/log/train_mod/model050.cptk"
INPUTPATH = "/hdd/klein/prepared/TestFiles/"
OUTPUTPATH = "/home/klein/bonet/results/run4/result/"

ROOM_PATH_LIST = glob.glob(os.path.join(INPUTPATH, "*.csv"))
len_pts_files = len(ROOM_PATH_LIST)

if not os.path.exists(OUTPUTPATH):
    os.makedirs(OUTPUTPATH)

def safeFile(pts, gt_sem, gt_group, pred_sem, labels, sem_pred_val, ins_pred_val, file_path):
	filename = file_path.split('/')[-1].split('.')[0]
	#print(filename)
	
	with open(file_path, 'r') as fd:
		head = fd.readlines()[0]
	
	gt_sem = np.reshape(gt_sem, (len(pred_sem),1))
	gt_group = np.reshape(gt_group,(len(labels),1))
	sem_labels = np.reshape(pred_sem, (len(pred_sem),1))
	instances = np.reshape(labels,(len(labels),1))
	sem_conf = np.reshape(sem_pred_val, (len(sem_pred_val),1))
	instances_conf = np.reshape(ins_pred_val,(len(ins_pred_val),1))

    #sem_labels = pred_sem
    #instances = labels
	all = np.append(pts, gt_sem, axis=1)
	all = np.append(all, gt_group, axis=1)
	all = np.append(all, sem_labels, axis=1)
	all = np.append(all, instances, axis=1)
	all = np.append(all, sem_conf, axis=1)
	all = np.append(all, instances_conf, axis=1)

    #print(all.shape)
	name = OUTPUTPATH + filename + ".csv"
	print("Save ", name)
	
	np.savetxt(name, all, delimiter=" ", header=head, fmt='%d %d %.10f %d %d %d %d %.3f %.3f', comments='//')

def load_net(model_path):
		#######
		from main_3D_BoNet_LK import BoNet
		from helper_data_dvs import Data_Configs as Data_Configs
		configs = Data_Configs()
		net = BoNet(configs=configs)

		####### 1. networks
		net.X_pc = tf.placeholder(shape=[None, None, net.points_cc], dtype=tf.float32, name='X_pc')
		net.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
		with tf.variable_scope('backbone'):
			#net.point_features, net.global_features, net.y_psem_pred = net.backbone_pointnet(net.X_pc, net.is_train)
			net.point_features, net.global_features, net.y_psem_pred = net.backbone_pointnet2(net.X_pc, net.is_train)
		with tf.variable_scope('bbox'):
			net.y_bbvert_pred_raw, net.y_bbscore_pred_raw = net.bbox_net(net.global_features)
		with tf.variable_scope('pmask'):
			net.y_pmask_pred_raw = net.pmask_net(net.point_features, net.global_features, net.y_bbvert_pred_raw, net.y_bbscore_pred_raw)

		####### 2. restore trained model
		if not os.path.isfile(model_path + '.data-00000-of-00001'):
			print ('please download the released model!'); return
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.visible_device_list = '0'
		net.sess = tf.Session(config=config)
		tf.train.Saver().restore(net.sess, model_path)
		print('Model restored sucessful!')

		return net

def test():
	net = load_net(MODELPATH)

	from helper_data_dvs import Data_S3DIS as Data

	train_dataset_path = '/hdd/klein/prepared/TrainFiles'
	test_dataset_path = '/hdd/klein/prepared/TestFiles'

	data = Data(train_dataset_path, test_dataset_path, train_batch_size=1)

	for file_idx in range(len_pts_files):
		file_path = ROOM_PATH_LIST[file_idx]
		file_list = []
		file_list.append(file_path)
		
		print("Processsing: File [%d] of [%d]" % (file_idx, len_pts_files))

		bat_pc, bat_sem_gt, bat_ins_gt, bat_psem_onehot, bat_bbvert, bat_pmask, bat_files = data.load_test_next_batch_sq(bat_files=file_list)

		#run session
		[y_psem_pred_sq_raw, y_bbvert_pred_sq_raw, y_bbscore_pred_sq_raw, y_pmask_pred_sq_raw] = \
		net.sess.run([net.y_psem_pred, net.y_bbvert_pred_raw, net.y_bbscore_pred_raw, net.y_pmask_pred_raw],feed_dict={net.X_pc: bat_pc[:, :, 0:3], net.is_train: False})

		pc = np.asarray(bat_pc[0], dtype=np.float16)
		sem_pred_raw = np.asarray(y_psem_pred_sq_raw[0], dtype=np.float16)
		bbvert_pred_raw = np.asarray(y_bbvert_pred_sq_raw[0], dtype=np.float16)
		bbscore_pred_raw = np.asarray(y_bbscore_pred_sq_raw[0], dtype=np.float16)
		pmask_pred_raw = np.asarray(y_pmask_pred_sq_raw[0], dtype=np.float16)

		sem_pred = np.argmax(sem_pred_raw, axis=-1)
		sem_pred_val = np.max(sem_pred_raw, axis=-1)
		pmask_pred = pmask_pred_raw * np.tile(bbscore_pred_raw[:, None], [1, pmask_pred_raw.shape[-1]])
		ins_pred = np.argmax(pmask_pred, axis=-2)
		ins_pred_val = np.max(pmask_pred, axis=-2)

		safeFile(pc, bat_sem_gt, bat_ins_gt, sem_pred, ins_pred, sem_pred_val, ins_pred_val, file_path)

#######################
if __name__=='__main__':
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'   ## specify the GPU to use

	test()