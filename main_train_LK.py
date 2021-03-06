import os
import numpy as np

import logging

from helper_data_dvs import NUM_POINTS
logging.basicConfig(format='%(asctime)s %(message)s')

NUM_POINT = 2**14

def train(net, data):
	batchsize = data.train_batch_size
	num_points = 16384
	print(batchsize)
	for ep in range(0, 51,1):
		print('#################################################')
		logging.warning('Start epoch %d' % ep)

		l_rate = max(0.0005/(2**(ep//20)), 0.00001)

		data.shuffle_train_files(ep)
		total_train_batch_num = data.total_train_batch_num
		print('total train batch num:', total_train_batch_num)

		#acc
		acc_sum = 0.0
		diff_sum = 0.0
		num_sum = 0.0

		for i in range(total_train_batch_num):
			###### training
			bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot, bat_bbvert, bat_pmask = data.load_train_next_batch()

			y_psem_pred_sq_raw, y_bbvert_pred_sq_raw, y_bbscore_pred_sq_raw, y_pmask_pred_sq_raw, _, ls_psemce, ls_bbvert_all, ls_bbvert_l2, ls_bbvert_ce, ls_bbvert_iou, ls_bbscore, ls_pmask = net.sess.run([
			net.y_psem_pred, net.y_bbvert_pred_raw, net.y_bbscore_pred_raw, net.y_pmask_pred_raw, net.optim, net.psemce_loss, net.bbvert_loss, net.bbvert_loss_l2, net.bbvert_loss_ce, net.bbvert_loss_iou,net.bbscore_loss, net.pmask_loss],
			feed_dict={net.X_pc:bat_pc[:, :, 0:3], net.Y_bbvert:bat_bbvert, net.Y_pmask:bat_pmask, net.Y_psem:bat_psem_onehot, net.lr:l_rate, net.is_train:True})

			#acc
			sum_acc = 0
			sum_diff = 0
			sum_num = 0

			for b in range(len(y_psem_pred_sq_raw)):
				#sem
				sem_gt = bat_sem_labels[b]
				sem_pred_raw = np.asarray(y_psem_pred_sq_raw[b], dtype=np.float16)
				sem_pred = np.argmax(sem_pred_raw, axis=-1)
				right_pred = np.count_nonzero(sem_gt==sem_pred)
				sum_acc += ((float(right_pred)/float(num_points)) * 100)
	
				#print("Acc: ", ((float(right_pred)/float(num_points)) * 100))

				#ins
				ins_gt = bat_ins_labels[b]
				bbscore_pred_raw = np.asarray(y_bbscore_pred_sq_raw[b], dtype=np.float16)
				pmask_pred_raw = np.asarray(y_pmask_pred_sq_raw[b], dtype=np.float16)
				pmask_pred = pmask_pred_raw * np.tile(bbscore_pred_raw[:, None], [1, pmask_pred_raw.shape[-1]])
				ins_pred = np.argmax(pmask_pred, axis=-2)

				ins1num = len(np.unique(ins_gt))
				ins2num = len(np.unique(ins_pred))
				sum_diff += abs(ins1num - ins2num)
				sum_num += ins2num

				#print("Right num of instances: ", ins1num, " Predicted num: ", ins2num, " Diff: ",abs(ins1num - ins2num), " Total: ", diff_sum)

			acc_sum += float((sum_acc/(batchsize)))
			diff_sum += float((sum_diff/(batchsize)))
			num_sum += float((sum_num/(batchsize)))


			#print("Done")
			if i%200==0:
				sum_train = net.sess.run(net.sum_merged,
				feed_dict={net.X_pc: bat_pc[:, :, 0:3], net.Y_bbvert: bat_bbvert, net.Y_pmask: bat_pmask, net.Y_psem: bat_psem_onehot, net.lr: l_rate, net.is_train: False})
				net.sum_writer_train.add_summary(sum_train, ep*total_train_batch_num + i)
			#print ('ep', ep, 'i', i, 'psemce', ls_psemce, 'bbvert', ls_bbvert_all, 'l2', ls_bbvert_l2, 'ce', ls_bbvert_ce, 'siou', ls_bbvert_iou, 'bbscore', ls_bbscore, 'pmask', ls_pmask)

			if i%1000==0:
				print ('ep', ep, 'i', i, 'psemce', ls_psemce, 'bbvert', ls_bbvert_all, 'l2', ls_bbvert_l2, 'ce', ls_bbvert_ce, 'siou', ls_bbvert_iou, 'bbscore', ls_bbscore, 'pmask', ls_pmask)


			#print("testing")
			###### random testing
			if i%1000==0:
				bat_pc, _, _, bat_psem_onehot, bat_bbvert, bat_pmask = data.load_test_next_batch_random()
				ls_psemce, ls_bbvert_all, ls_bbvert_l2, ls_bbvert_ce, ls_bbvert_iou, ls_bbscore, ls_pmask, sum_test, pred_bborder = net.sess.run([
				net.psemce_loss, net.bbvert_loss, net.bbvert_loss_l2, net.bbvert_loss_ce, net.bbvert_loss_iou, net.bbscore_loss, net.pmask_loss, net.sum_merged, net.pred_bborder],
				feed_dict={net.X_pc:bat_pc[:, :, 0:3], net.Y_bbvert:bat_bbvert, net.Y_pmask:bat_pmask, net.Y_psem:bat_psem_onehot, net.is_train:False})
				net.sum_write_test.add_summary(sum_test, ep*total_train_batch_num+i)
				print('ep',ep,'i',i,'test psem', ls_psemce, 'bbvert', ls_bbvert_all, 'l2', ls_bbvert_l2, 'ce', ls_bbvert_ce, 'siou', ls_bbvert_iou, 'bbscore', ls_bbscore, 'pmask', ls_pmask)

			#print("Saving")
			###### saving model
			if i==(total_train_batch_num-1) or i==0:
				net.saver.save(net.sess, save_path=net.train_mod_dir + 'model.cptk')
				print ("ep", ep, " i", i, " model saved!")
			if ep % 5 == 0 and i == total_train_batch_num - 1:
				net.saver.save(net.sess, save_path=net.train_mod_dir + 'model' + str(ep).zfill(3) + '.cptk')

		
		logging.warning('Semantic mean accuracy: %.2f' % ((acc_sum / float(total_train_batch_num))))
		logging.warning('Instance mean difference: %.2f' % (diff_sum / float(total_train_batch_num)))
		logging.warning('Instance mean: %.2f' % (num_sum / float(total_train_batch_num)))
		logging.warning('End epoch %d' % ep)

############
if __name__=='__main__':

	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'  ## specify the GPU to use

	from main_3D_BoNet_LK import BoNet
	from helper_data_dvs import Data_Configs as Data_Configs

	configs = Data_Configs()
	net = BoNet(configs = configs)
	net.creat_folders(name='log', re_train=False)
	net.build_graph()

	####
	from helper_data_dvs import Data_DVS as Data

	train_dataset_path = '/hdd/klein/prepared/TrainFiles'
	test_dataset_path = '/hdd/klein/prepared/TestFiles'

	data = Data(train_dataset_path, test_dataset_path, train_batch_size=1)
	train(net, data)