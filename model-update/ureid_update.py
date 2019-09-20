from __future__ import print_function

import os
import os.path as osp
import re
import sys
import time
import shutil
import argparse

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel

import numpy as np
from scipy.stats import norm,alpha,levy_stable
from PIL import Image
from sklearn.cluster import KMeans
from collections import defaultdict
from feature_extraction import UReIDFeatureExtraction

sys.path.append('./beyond-part-models')
from bpm.dataset import create_dataset
from bpm.model.PCBModel import PCBModel as Model

from bpm.utils.utils import time_str
from bpm.utils.utils import str2bool
from bpm.utils.utils import may_set_mode
from bpm.utils.utils import load_state_dict
from bpm.utils.utils import load_ckpt
from bpm.utils.utils import load_pickle
from bpm.utils.utils import save_ckpt
from bpm.utils.utils import save_pickle
from bpm.utils.utils import set_devices
from bpm.utils.utils import AverageMeter
from bpm.utils.utils import to_scalar
from bpm.utils.utils import ReDirectSTD
from bpm.utils.utils import set_seed
from bpm.utils.utils import adjust_lr_staircase
from bpm.utils.dataset_utils import parse_im_name as parse_new_im_name
from bpm.utils.dataset_utils import partition_train_val_set

class Config(object):
    def __init__(self, gpus, dataset, feat_dims_per_part, \
        only_test, loop, save_dir):
        self.run = loop
        self.seed = None
        if self.seed is not None:
            self.prefetch_threads = 1
        else:
            self.prefetch_threads = 2
        self.trainset_part = 'trainval'
        
        # gpu ids
        self.sys_device_ids = gpus

        # dataset name
        self.dataset = dataset

        # Just for training set
        self.crop_prob = 0
        self.crop_ratio = 1

        self.resize_h_w = (384, 128)
        self.scale_im = True
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]

        self.train_mirror_type = 'random'
        self.train_batch_size = 64
        self.train_final_batch = False
        self.train_shuffle = True

        self.test_mirror_type = None
        self.test_batch_size = 32
        self.test_final_batch = True
        self.test_shuffle = False

        ## Dataset
        dataset_kwargs = dict(
            name=self.dataset,
            resize_h_w=self.resize_h_w,
            scale=self.scale_im,
            im_mean=self.im_mean,
            im_std=self.im_std,
            batch_dims='NCHW',
            num_prefetch_threads=self.prefetch_threads)

        # Training dataset
        prng = np.random
        if self.seed is not None:
            prng = np.random.RandomState(self.seed)
        self.train_set_kwargs = dict(
            part=self.trainset_part,
            batch_size=self.train_batch_size,
            final_batch=self.train_final_batch,
            shuffle=self.train_shuffle,
            crop_prob=self.crop_prob,
            crop_ratio=self.crop_ratio,
            mirror_type=self.train_mirror_type,
            prng=prng)
        self.train_set_kwargs.update(dataset_kwargs)
        # Valid dataset
        prng = np.random
        if self.seed is not None:
            prng = np.random.RandomState(self.seed)
        self.val_set_kwargs = dict(
            part='val',
            batch_size=self.test_batch_size,
            final_batch=self.test_final_batch,
            shuffle=self.test_shuffle,
            mirror_type=self.test_mirror_type,
            prng=prng)
        self.val_set_kwargs.update(dataset_kwargs)
        self.val_set = None
        # Test dataset
        prng = np.random
        if self.seed is not None:
            prng = np.random.RandomState(self.seed)
        self.test_set_kwargs = dict(
            part='test',
            batch_size=self.test_batch_size,
            final_batch=self.test_final_batch,
            shuffle=self.test_shuffle,
            mirror_type=self.test_mirror_type,
            prng=prng)
        self.test_set_kwargs.update(dataset_kwargs)

        ## ReID model
        self.last_conv_stride = 1
        self.last_conv_dilation = 1
        self.num_stripes = 6
        self.local_conv_out_channels = feat_dims_per_part
        
        ## Training
        self.momentum = 0.9
        self.weight_decay = 0.0005
        # Initial learning rate
        self.new_params_lr = 0.1
        self.finetuned_params_lr = 0.01
        self.staircase_decay_at_epochs = (41,)
        self.staircase_decay_multiply_factor = 0.1
        # Number of epochs to train
        self.total_epochs = 60
        # How often (in epochs) to test on val set.
        self.epochs_per_val = 1
        # How often (in batches) to log. If only need to log the average
        # information for each epoch, set this to a large value, e.g. 1e10.
        self.steps_per_log = 20
        # Only test and without training.
        self.only_test = only_test

        self.resume = False

        self.log_to_file = True
        # The root dir of logs.
        assert loop > 0
        if save_dir == '':
            self.exp_dir = osp.join(
            'exp/train',
            '{}'.format(self.dataset),
            'run{}'.format(self.run),
        )
        else:
            self.exp_dir = save_dir + '/' + str(loop)
        self.stdout_file = osp.join(
            self.exp_dir, 'stdout_{}.txt'.format(time_str()))
        self.stderr_file = osp.join(
            self.exp_dir, 'stderr_{}.txt'.format(time_str()))
        #ckpt_fname = 'ckpt' + str(loop) + '.pth'
        self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')


class ExtractFeature(object):
    """A function to be called in the val/test set, to extract features.
    Args:
        TVT: A callable to transfer images to specific device.
    """
    def __init__(self, model, TVT):
        self.model = model
        self.TVT = TVT
    def __call__(self, ims):
        old_train_eval_model = self.model.training
        # Set eval mode.
        # Force all BN layers to use global mean and variance, also disable
        # dropout.
        self.model.eval()

        ims = Variable(self.TVT(torch.from_numpy(ims).float()))
        try:
            local_feat_list, logits_list = self.model(ims)
        except:
            local_feat_list = self.model(ims)
        feat = [lf.data.cpu().numpy() for lf in local_feat_list]
        feat = np.concatenate(feat, axis=1)

        # Restore the model to its old train/eval mode.
        self.model.train(old_train_eval_model)
        return feat
    
class UReID(object):

    def __init__(self, args, loop, num_classes_dict):
        gpu_ids = args['gpus']
        only_test = args['only_test']
        feat_dims_per_part = args['stripe_feats_dims']
        # Parameters
        self.num_classes_dict = num_classes_dict
        self.dataset_nm = args['dataset']
        self.update_trainingset_path = args['update_trainingset_path']
        root_save_dir = args['root_save_dir']
        save_dir = osp.join(root_save_dir, self.dataset_nm)
        self.save_dir = save_dir
        self.loop = loop
        self.num_clusters = int(args['num_clusters'][self.dataset_nm])
        # For uReID
        self.simlarity_thresh = float(args['lambda'])
        cfg = Config(gpu_ids, self.dataset_nm, feat_dims_per_part, \
            only_test, loop, save_dir)
        
        # Redirect logs to both console and file.
        if cfg.log_to_file:
            ReDirectSTD(cfg.stdout_file, 'stdout', False)
            ReDirectSTD(cfg.stderr_file, 'stderr', False)
        
        # Dump the configurations to log.
        self.cfg = cfg
        import pprint
        print('-' * 60)
        print('cfg.__dict__')
        pprint.pprint(cfg.__dict__)
        print('-' * 60)

        # Load data
        self._loadDataset()
        print('INFO: 1. Datasets are loaded ...')

        # Updata number of classes
        self.num_classes_dict[self.dataset_nm] = len(self.train_set.ids2labels)

        # Initialize Model
        self._initModel()
        print('INFO: 2. Model is initialized ...')
 
        print('INFO: ==== Project initialize successfully!')

    def _initModel(self):
        # Current dataset name:
        dname = self.dataset_nm
        # Number of classes
        num_classes = self.num_classes_dict[dname]
        # Parameters
        cfg = self.cfg
        # ReID model
        TVT, TMO = set_devices(cfg.sys_device_ids)
        if cfg.seed is not None:
            set_seed(cfg.seed)
        model = Model(
            last_conv_stride=cfg.last_conv_stride,
            num_stripes=cfg.num_stripes,
            local_conv_out_channels=cfg.local_conv_out_channels,
            num_classes=num_classes
        )
        # Model wrapper
        model_w = DataParallel(model)

        # Criteria and Optimizers
        criterion = torch.nn.CrossEntropyLoss()
        # To finetune from ImageNet weights
        finetuned_params = list(model.base.parameters())
        # To train from scratch
        new_params = [p for n, p in model.named_parameters()
            if not n.startswith('base.')]
        param_groups = [{'params': finetuned_params, \
            'lr': cfg.finetuned_params_lr}, \
            {'params': new_params, 'lr': cfg.new_params_lr}]
        optimizer = optim.SGD(
            param_groups,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay)
        
        # Bind them together just to save some codes in the following usage.
        modules_optims = [model, optimizer]

        # May Transfer Models and Optims to Specified Device. 
        # Transferring optimizer is to cope with the case when 
        # you load the checkpoint to a new device.
        TMO(modules_optims)

        self.model_w = model_w
        self.modules_optims = modules_optims
        self.criterion = criterion
        self.TVT = TVT

    def _loadDataset(self):
        cfg = self.cfg
        # training set
        self.train_set = create_dataset(**cfg.train_set_kwargs)
        # Valid set
        self.val_set = None if cfg.dataset == 'combined' else \
            create_dataset(**cfg.val_set_kwargs)
        # Test set
        test_sets = []
        test_set_names = []
        if cfg.dataset == 'combined':
            for name in ['market1501', 'cuhk03', 'duke']:
                cfg.test_set_kwargs['name'] = name
                test_sets.append(create_dataset(**cfg.test_set_kwargs))
                test_set_names.append(name)
        else:
            test_sets.append(create_dataset(**cfg.test_set_kwargs))
            test_set_names.append(cfg.dataset)
        self.test_sets = test_sets
        self.test_set_names = test_set_names

    def _trainPCB(self):
        cfg = self.cfg
        TVT = self.TVT
        model_w = self.model_w
        criterion = self.criterion
        modules_optims = self.modules_optims
        optimizer = modules_optims[1]
        #train_set = create_dataset(**cfg.train_set_kwargs)
        train_set = self.train_set
        start_ep = 0
        for ep in range(start_ep, cfg.total_epochs):
            # Adjust Learning Rate
            adjust_lr_staircase(
                optimizer.param_groups,
                [cfg.finetuned_params_lr, cfg.new_params_lr],
                ep + 1,
                cfg.staircase_decay_at_epochs,
                cfg.staircase_decay_multiply_factor)

            may_set_mode(modules_optims, 'train')

            # For recording loss
            loss_meter = AverageMeter()

            ep_st = time.time()
            step = 0
            epoch_done = False
            while not epoch_done:
                step += 1
                step_st = time.time()
                ims, im_names, labels, mirrored, epoch_done = \
                    train_set.next_batch()
                ims_var = Variable(TVT(torch.from_numpy(ims).float()))
                labels_var = Variable(TVT(torch.from_numpy(labels).long()))
                _, logits_list = model_w(ims_var)
                loss = torch.sum(
                    torch.cat([criterion(logits, labels_var) for \
                    logits in logits_list]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Step log
                loss_meter.update(to_scalar(loss))
                if step % cfg.steps_per_log == 0:
                    log = '\tStep {}/Ep {}, {:.2f}s, loss {:.4f}'.format(
                        step, ep + 1, time.time() - step_st, loss_meter.val)
                    print(log)

            # Epoch Log
            log = 'Ep {}, {:.2f}s, loss {:.4f}'.format(
                ep + 1, time.time() - ep_st, loss_meter.avg)
            print(log)

            # Test on Validation Set
            mAP, Rank1 = 0, 0
            val_set = self.val_set
            if ((ep + 1) % cfg.epochs_per_val == 0) and (val_set is not None):
                mAP, Rank1 = self._validatePCB()

            # Save ckpt
            if cfg.log_to_file:
                save_ckpt(modules_optims, ep + 1, 0, cfg.ckpt_file)

    def _testPCB(self):
        cfg = self.cfg
        # test set
        test_sets = self.test_sets
        test_set_names = self.test_set_names

        load_ckpt(self.modules_optims, cfg.ckpt_file)
        model_w = self.model_w
        TVT = self.TVT
        for test_set, name in zip(test_sets, test_set_names):
            test_set.set_feat_func(ExtractFeature(model_w, TVT))
            print('\n=========> Test on dataset: {} <=========\n'.format(name))
            test_set.eval(normalize_feat=True, verbose=True)

    def _validatePCB(self):
        cfg = self.cfg
        model_w = self.model_w
        TVT = self.TVT
        val_set = self.val_set
        if val_set.extract_feat_func is None:
            val_set.set_feat_func(ExtractFeature(model_w, TVT))
        print('\n===== Test on validation set =====\n')
        mAP, cmc_scores, _, _ = val_set.eval(
            normalize_feat=True,
            to_re_rank=False,
            verbose=True)
        print()
        return mAP, cmc_scores[0]

    def _extractFeaturesTrainingSet(self):
        cfg = self.cfg
        # Load model
        modules_optims = self.modules_optims
        ckpt_file = self.save_dir + '/' + str(self.loop-1) + '/ckpt.pth'
        load_ckpt(modules_optims, ckpt_file)
        model = self.model_w
        # Create extractor
        TVT = self.TVT
        batchsize = cfg.test_batch_size
        extractor = UReIDFeatureExtraction(model, TVT, cfg)
        # Extract features
        real_dirnm = osp.join('./dataset/real', self.dataset_nm)
        ### test
        #real_dirnm = 'testbed_dataset2/market1501'
        ### end
        self.real_training_imgs_dir = real_dirnm
        feats_global, feat_local_list = \
            extractor.extractFeatures(real_dirnm, batchsize)
        return feats_global, feat_local_list

    def _confirmReliableSample(self, similarities_vec, fnames):
        # Confirm the initial threshold of similarity.
        INIT_PERCENTILE = 0.5
        similarities_vec_sorted = np.sort(similarities_vec)
        num = similarities_vec.shape[0]
        pos = int(num * INIT_PERCENTILE) - 1
        init_threshold = similarities_vec_sorted[pos]
        # Get the reliable samples
        reliable_samples, = np.where(similarities_vec > init_threshold)
        return reliable_samples, init_threshold

    def _calSimilarityThreshold(self, similarities_mat, src_fnames, labels):
        similarities_vec = \
            similarities_mat[labels, np.arange(labels.shape[0])]
        if self.loop == 1:
            reliable_samples, _ = \
                self._confirmReliableSample(similarities_vec, src_fnames)
            self.setReliableSamples(reliable_samples)
        else:
            reliable_samples = self.getReliableSamples()
        similarities_reliable = similarities_vec[reliable_samples]
        # Fit a norm distribution
        distance = 1.0 - similarities_reliable
        bad_pos, = np.where(distance <= 0)
        distance[bad_pos] = 1e-6
        loc, scale = norm.fit(distance)
        # Get the threshold
        threshold = norm.ppf(0.95, loc=loc, scale=scale)
        threshold = 1 - threshold
        print('INFO: %.6f is chosen as the similarity threshold!' % threshold)
        # Record the threshold
        save_dir = 'models/' + self.dataset_nm
        filename = 'similarity_threshold_PCB_' + str(self.loop) + '.txt'
        save_path = osp.join(save_dir, filename)
        to_save_data = np.asarray([threshold])
        np.savetxt(save_path, to_save_data, fmt='%f')
        return threshold

    def _saveMiningRes(self, save_fpath, fnames, labels):
        '''The fnames here are under orignal format; while labels are new
           generated using KMeans.
        '''
        lines = []
        for fname, label in zip(fnames, labels):
            line = [fname, label]
            lines.append(line)
        lines = np.asarray(lines)
        np.savetxt(save_fpath, lines, fmt='%s', delimiter=',')

    def _updatePartitionOnlyClusteringRes(self, img_fnames):
        specific_dir = self.dataset_nm
        ## Old
        pkl_fpath_old = osp.join(self.update_trainingset_path, specific_dir, \
            'original')
        pkl_fname = osp.join(pkl_fpath_old, 'partitions.pkl')
        old_partitions = load_pickle(pkl_fname)
        ## Update
        # Train Val
        trainval_im_names = list(img_fnames)
        trainval_im_names.sort()
        trainval_ids = list(set([parse_new_im_name(n, 'id')
            for n in trainval_im_names]))
        trainval_ids.sort()
        trainval_ids2labels = dict(zip(trainval_ids, range(len(trainval_ids))))
        partitions = partition_train_val_set(
            trainval_im_names, parse_new_im_name, num_val_ids=100)
        # Train
        train_im_names = partitions['train_im_names']
        train_ids = list(set([parse_new_im_name(n, 'id')
            for n in partitions['train_im_names']]))
        train_ids.sort()
        train_ids2labels = dict(zip(train_ids, range(len(train_ids))))
        # Val
        val_marks = [0, ] * len(partitions['val_query_im_names']) \
            + [1, ] * len(partitions['val_gallery_im_names'])
        val_im_names = list(partitions['val_query_im_names']) \
            + list(partitions['val_gallery_im_names'])
        ## Save
        new_partitions = {'trainval_im_names': trainval_im_names,
                          'trainval_ids2labels': trainval_ids2labels,
                          'train_im_names': train_im_names,
                          'train_ids2labels': train_ids2labels,
                          'val_im_names': val_im_names,
                          'val_marks': val_marks,
                          'test_im_names': old_partitions['test_im_names'],
                          'test_marks': old_partitions['test_marks']}
        pkl_fpath_new = osp.join(self.update_trainingset_path, specific_dir, \
            'new')
        pkl_fname = osp.join(pkl_fpath_new, 'partitions.pkl')
        save_pickle(new_partitions, pkl_fname)
        return pkl_fname

    def _updatePartition(self, img_fnames):
        specific_dir = self.dataset_nm
        ## Old
        pkl_fpath_old = osp.join(self.update_trainingset_path, specific_dir, \
            'original')
        pkl_fname = osp.join(pkl_fpath_old, 'partitions.pkl')
        old_partitions = load_pickle(pkl_fname)
        trainval_im_names = old_partitions['trainval_im_names']
        ## Update
        # Train Val
        trainval_im_names = trainval_im_names + list(img_fnames)
        trainval_im_names.sort()
        trainval_ids = list(set([parse_new_im_name(n, 'id')
            for n in trainval_im_names]))
        trainval_ids.sort()
        trainval_ids2labels = dict(zip(trainval_ids, range(len(trainval_ids))))
        partitions = partition_train_val_set(
            trainval_im_names, parse_new_im_name, num_val_ids=100)
        # Train
        train_im_names = partitions['train_im_names']
        train_ids = list(set([parse_new_im_name(n, 'id')
            for n in partitions['train_im_names']]))
        train_ids.sort()
        train_ids2labels = dict(zip(train_ids, range(len(train_ids))))
        # Val
        val_marks = [0, ] * len(partitions['val_query_im_names']) \
            + [1, ] * len(partitions['val_gallery_im_names'])
        val_im_names = list(partitions['val_query_im_names']) \
            + list(partitions['val_gallery_im_names'])
        ## Save
        new_partitions = {'trainval_im_names': trainval_im_names,
                          'trainval_ids2labels': trainval_ids2labels,
                          'train_im_names': train_im_names,
                          'train_ids2labels': train_ids2labels,
                          'val_im_names': val_im_names,
                          'val_marks': val_marks,
                          'test_im_names': old_partitions['test_im_names'],
                          'test_marks': old_partitions['test_marks']}
        pkl_fpath_new = osp.join(self.update_trainingset_path, specific_dir, \
            'new')
        pkl_fname = osp.join(pkl_fpath_new, 'partitions.pkl')
        save_pickle(new_partitions, pkl_fname)
        return pkl_fname

    def _renameImgFName(self, fnames, labels):
        name_pattern = re.compile(r'([-\d]+)_c(\d)')
        name_format = '{:08d}_{:04d}_{:08d}.jpg'
        cnt = defaultdict(int)
        new_fnames = []
        for fname, pid in zip(fnames, labels):
            _, cam = name_pattern.search(fname).groups()
            # Avoid the same id with original data
            if self.dataset_nm == 'market1501':
                flag = 30000
            elif self.dataset_nm == 'duke':
                flag = 40000
            else:
                flag = 50000
            pid = flag + pid
            cnt[(pid, cam)] += 1
            # Rename
            new_fname = name_format.format(pid, int(cam), cnt[(pid, cam)]-1)
            new_fnames.append(new_fname)
        return new_fnames

    def _getKMeansLabels(self, feats, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters).fit(feats)
        labels = kmeans.labels_
        # select centers
        distances = kmeans.transform(feats)
        center_idx = np.argmin(distances, axis=0)
        #centers = [features[i] for i in center_idx]
        centers = feats[center_idx, :]
        return labels, centers

    def _updateFiles(self, src_fnames, dst_fnames):
        src_dir = osp.join('./dataset/real', self.dataset_nm)
        if self.dataset_nm == 'duke':
            dst_dir = './dataset/synthetic/duke'
        elif self.dataset_nm == 'market1501':
            dst_dir = './dataset/synthetic/market'
        else:
            print('ERROR: Bad dataset name!')
            exit(-1)
        # Copy the old images and rename them and move them to images dir.
        for ofname, nfname in zip(src_fnames, dst_fnames):
            src_file = osp.join(src_dir, ofname)
            dst_file = osp.join(dst_dir, 'images', nfname)
            shutil.copyfile(src_file, dst_file)
        print('INFO: loop %d - new generated training samples are saved in %s' \
            % (self.loop, osp.join(dst_dir, 'images')))
        # Updata pkl data
        new_partition_file = self._updatePartition(dst_fnames)
        #new_partition_file = self._updatePartitionOnlyClusteringRes(dst_fnames)
        print('INFO: loop %d - new partitions file is saved in %s' % \
            (self.loop, new_partition_file))
        shutil.copyfile(new_partition_file, osp.join(dst_dir, 'partitions.pkl'))
        print('INFO: loop %d - the partitions file is updated' % self.loop)

    def _clusteringGlobal_v2(self, features):
        # Get the cam info of original training samples:
        # 1. Get all the image names
        dname = osp.join('./dataset/real',  self.dataset_nm)
        ### test
        #dname = 'testbed_dataset2/market1501'
        ### end
        filelist = os.listdir(dname)
        filelist = np.asarray(filelist)
        filelist = np.sort(filelist)
        # 2. Get cams
        cams = []
        name_pattern = re.compile(r'([-\d]+)_c(\d)')
        for fname in filelist:
            _, cam = name_pattern.search(fname).groups()
            cams.append(int(cam))
        cams = np.asarray(cams)
        # KMean:
        NUM_CLUSTER = self.num_clusters
        LAMBDA = self.simlarity_thresh
        labels, centers = self._getKMeansLabels(features, NUM_CLUSTER)
        similarities = np.dot(centers, np.transpose(features))
        # select reliable images
        reliable_image_idx = np.unique(np.argwhere(similarities > LAMBDA)[:,1])
        print('INFO: loop %d - %d reliable images are got' % (self.loop, \
            reliable_image_idx.shape[0]))
        reliable_img_names = filelist[reliable_image_idx]
        reliable_img_labels= labels[reliable_image_idx]
        reliable_img_cams  = cams[reliable_image_idx]
        # Choose the samples under multiple cameras
        labels_unique, freq = np.unique(reliable_img_labels, return_counts=True)
        mul_labels_positions, = np.where(freq>=2)
        mul_labels = labels_unique[mul_labels_positions]
        new_samples = []
        new_labels = []
        for label in mul_labels:
            current_positions, = np.where(reliable_img_labels == label)
            current_cams = reliable_img_cams[current_positions]
            current_cams_unique = np.unique(current_cams)
            if current_cams_unique.shape[0] <= 1:
                continue
            else:
                fnames = reliable_img_names[current_positions]
                new_samples.append(fnames)
                num_fnames = fnames.shape[0]
                assert num_fnames >= 2
                nlabels = [label]*num_fnames
                new_labels.append(nlabels)
        new_samples = np.hstack(new_samples)
        new_labels = np.hstack(new_labels)
        # rename image filenames
        new_fnames = self._renameImgFName(new_samples, new_labels)
        # Save the mining results
        save_fname = 'generated_samples_cross_cam_' + str(self.loop) + '.txt'
        save_fpath = osp.join('models', self.dataset_nm, save_fname)
        self._saveMiningRes(save_fpath, new_samples, new_labels)
        print('INFO: loop %d - %d reliable images are chosen' % (self.loop, \
            len(new_samples)))
        # Update training dataset
        self._updateFiles(new_samples, new_fnames)

    def _clusteringGlobal(self, features):
        NUM_CLUSTER = self.num_clusters
        LAMBDA = self.simlarity_thresh
        kmeans = KMeans(n_clusters=NUM_CLUSTER).fit(features)

        # select centers
        distances = kmeans.transform(features)
        center_idx = np.argmin(distances, axis=0)
        centers = features[center_idx, :]
        # similarity
        similarities = np.dot(centers, np.transpose(features))

        # image names
        filelist = os.listdir(self.real_training_imgs_dir)
        filelist = np.asarray(filelist)
        filelist = np.sort(filelist)

        # Get the threshold of similarities
        labels = kmeans.labels_
        #LAMBDA = self._calSimilarityThreshold(similarities, filelist, labels)

        # select reliable images
        reliable_image_idx = np.unique(np.argwhere(similarities > LAMBDA)[:,1])
        print('INFO: loop %d - %d reliable images are got' % (self.loop, \
            reliable_image_idx.shape[0]))

        # Reliable image names
        reliable_img_names = filelist[reliable_image_idx]
        reliable_img_labels= kmeans.labels_[reliable_image_idx]

        # save new generated traing data
        # rename image filenames
        new_fnames = self._renameImgFName(reliable_img_names, \
            reliable_img_labels)
        # Save the mining results
        save_fname = 'generated_samples_base_auto_' + str(self.loop) + '.txt'
        save_fpath = osp.join('models', self.dataset_nm, save_fname)
        self._saveMiningRes(save_fpath, reliable_img_names, reliable_img_labels)
        # Update training dataset
        self._updateFiles(reliable_img_names, new_fnames)

    def _clusteringLocal(self):
        pass

    def train(self):
        # Extract features of real training dataset
        print('INFO: ==== Extract feature of real training dataset ...')
        feats_global, feat_local_list = self._extractFeaturesTrainingSet()
        # Clustering
        print('INFO: ==== KMean and choose reliable samples ...')
        # updata the generated training set with reliable clustering result.
        #self._clusteringGlobal_v2(feats_global)
        self._clusteringGlobal(feats_global)
        print('INFO: ==== Train PCB ...')
        # Updata data
        print('INFO: the data is update ...')
        self._loadDataset()
        self.num_classes_dict[self.dataset_nm] = len(self.train_set.ids2labels)
        print('INFO: the model is re-initialized ...')
        # Re-initialize the model
        self._initModel()
        # Train ...
        self._trainPCB()
        self._testPCB()

    def test(self):
        self._testPCB()

    def trainPCB_only(self):
        self._trainPCB()
        self._testPCB()

    def setReliableSamples(self, reliable_samples):
        self.reliable_samples = reliable_samples

    def getReliableSamples(self):
        return self.reliable_samples

def main():
    num_classes_dict = {'market1501': 702, 'duke': 751} # To load the init model
    num_clusters = {'market1501': 750, 'duke': 700}
    update_trainingset_path = './dataset/update'
    args = {'gpus': (0,), 'only_test': False, 'stripe_feats_dims': 256, \
        'dataset': 'market1501', \
        'update_trainingset_path': update_trainingset_path, \
        'root_save_dir': './models', 'lambda': 0.75, \
        'num_clusters': num_clusters}
    loop_upbound = 26
    for loop in range(1, loop_upbound):
        print('INFO: ******** Loop %d ********' % loop)
        ureider = UReID(args, loop, num_classes_dict)
        ureider.train()
        del ureider

if __name__ == "__main__":
    main()
