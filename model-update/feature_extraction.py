import sys
import os
import cv2
import os.path as osp

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel

import numpy as np
from PIL import Image

### Test
#from checking.metrics import cmc, mean_ap
### End

class UReIDFeatureExtraction(object):

    def __init__(self, model, TVT, cfg):
        self.TVT = TVT
        # Dataset settings
        self.scale_im = cfg.scale_im # Whether to scale by 1/255
        self.im_mean = cfg.im_mean
        self.im_std = cfg.im_std
        self.data_dims = 'NCHW'
        self.resize_h_w = cfg.resize_h_w
        self.model_w = model
        self.dataset_nm = cfg.dataset
        self.num_stripes = cfg.num_stripes
        print('============= Extract Features =============')
        print('INFO: UReIDFeatureExtraction initialized successfully!')

    def _normalize(self, nparray, order=2, axis=0):
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    def _compute_dist(self, array1, array2, type='euclidean'):
        """Compute the euclidean or cosine distance of all pairs.
        Args:
          array1: numpy array with shape [m1, n]
          array2: numpy array with shape [m2, n]
          type: one of ['cosine', 'euclidean']
        Returns:
          numpy array with shape [m1, m2]
        """
        assert type in ['cosine', 'euclidean']
        if type == 'cosine':
            array1 = normalize(array1, axis=1)
            array2 = normalize(array2, axis=1)
            dist = np.matmul(array1, array2.T)
            return dist
        else:
            # shape [m1, 1]
            square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
            # shape [1, m2]
            square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
            squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
            squared_dist[squared_dist < 0] = 0
            dist = np.sqrt(squared_dist)
            return dist

    def _compute_score(self, dist_mat, query_ids, gallery_ids, query_cams, \
        gallery_cams):
        # Compute mean AP
        mAP = mean_ap(distmat=dist_mat, query_ids=query_ids, \
            gallery_ids=gallery_ids, query_cams=query_cams, \
            gallery_cams=gallery_cams)
        # Compute CMC scores
        cmc_scores = cmc(distmat=dist_mat, query_ids=query_ids, \
            gallery_ids=gallery_ids, query_cams=query_cams, \
            gallery_cams=gallery_cams, separate_camera_set=False, \
            single_gallery_shot=False, first_match_break=True, topk=20)
        return mAP, cmc_scores

    def _print_scores(self, mAP, cmc_scores):
        print('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}], [cmc20: {:5.2%}]'.format(mAP, *cmc_scores[[0, 4, 9, 19]]))

    def _getImageFNames(self, dirname, batch_size):
        print('INFO: The images will be load from %s with batch size is %d' % \
            (dirname, batch_size))
        filelist = os.listdir(dirname)
        total_num = len(filelist)
        filelist = np.asarray(filelist)
        filelist = np.sort(filelist)
        print('INFO: %d files will be loaded ...' % total_num)
        # Batch num
        if total_num % batch_size == 0:
            num_batches = total_num / batch_size
        else:
            num_batches = total_num // batch_size + 1
        image_batch_list = []
        image_batch = []
        cnt = 0
        for item in filelist:
            img_path = osp.join(dirname, item)
            if len(image_batch) < batch_size:
                image_batch.append(img_path)
            else:
                image_batch_list.append(image_batch)
                image_batch = []
                image_batch.append(img_path)
            cnt = cnt + 1
            if cnt == total_num and len(image_batch_list) < num_batches:
                image_batch_list.append(image_batch)
                
        print('INFO: %d batches are generated!' % len(image_batch_list))
        return image_batch_list

    def _getImageFNamesBatch(self, dirname, img_names, batch_size):
        print('INFO: %d files will be loaded ...' % len(img_names))
        total_num = len(img_names)
        # Batch num
        if total_num % batch_size == 0:
            num_batches = total_num / batch_size
        else:
            num_batches = total_num // batch_size + 1
        image_batch_list = []
        image_batch = []
        cnt = 0
        for item in img_names:
            img_path = osp.join(dirname, item)
            if len(image_batch) < batch_size:
                image_batch.append(img_path)
            else:
                image_batch_list.append(image_batch)
                image_batch = []
                image_batch.append(img_path)
            cnt = cnt + 1
            if cnt == total_num and len(image_batch_list) < num_batches:
                image_batch_list.append(image_batch)

        print('INFO: %d batches are generated!' % len(image_batch_list))
        return image_batch_list

    def _getSamples(self, img_name):
        im = np.asarray(Image.open(img_name))
        im = self._preprocess(im)
        return im

    def _preprocess(self, img):
        # Parameters to preprocess.
        resize_h_w = self.resize_h_w
        scale = self.scale_im
        im_mean = self.im_mean
        im_std = self.im_std
        data_dims = self.data_dims
        # Process
        # Resize
        if (resize_h_w is not None) \
            and (resize_h_w != (img.shape[0], img.shape[1])):
            img = cv2.resize(img, resize_h_w[::-1], \
                interpolation=cv2.INTER_LINEAR)
        # Scaled by 1/255
        if scale:
            img = img / 255.
        #print(img)
        # Subtract mean and scaled by std
        if im_mean is not None:
            img = img - np.asarray(im_mean)
        if im_mean is not None and im_std is not None:
            img = img / np.asarray(im_std).astype(float)
        # The original image has dims 'HWC', transform it to 'CHW'.
        if data_dims == 'NCHW':
            img = img.transpose(2, 0, 1)
        return img

    def _extract(self, imgs):
        TVT = self.TVT
        model = self.model_w
        old_train_eval_model = model.training
        # Set eval mode.
        model.eval()
        imgs = Variable(TVT(torch.from_numpy(imgs).float()))
        try:
            local_feat_list, logits_list = model(imgs)
        except:
            local_feat_list = model(imgs)
        feat_list= [lf.data.cpu().numpy() for lf in local_feat_list]
        feat = np.concatenate(feat_list, axis=1)
        # Restore the model to its old train/eval mode.
        model.train(old_train_eval_model)
        return feat, feat_list

    def extractFeatures(self, imgs_root_dir, batch_size=32):
        # Load images
        dataset_nm = self.dataset_nm
        #dataset_path = osp.join(imgs_root_dir, dataset_nm)
        dataset_path = imgs_root_dir
        img_names_batch_list = self._getImageFNames(dataset_path, batch_size)
        # Extraction
        feat_global = []
        feat_local_list = []
        num_stripes = self.num_stripes
        for i in range(num_stripes):
            feat_local_list.append([])
        cnt_batch = 0
        num_batches = len(img_names_batch_list)
        for img_names_batch in img_names_batch_list:
            imgs_list = []
            for image_name in img_names_batch:
                #print(image_name)
                img = self._getSamples(image_name)
                imgs_list.append(img)
            imgs_array = np.stack(imgs_list, axis=0)
            #print(imgs_array)
            feat, feat_list = self._extract(imgs_array)
            #print(feat)
            feat_global.append(feat)
            # Updata value of loacal feature
            for i in range(num_stripes):
                feat_local_list[i].append(feat_list[i])
            cnt_batch += 1
            print('Batch: %d/%d' % (cnt_batch, num_batches))
        feat_global = np.vstack(feat_global)
        #print(feat_global.shape)
        feat_global = self._normalize(feat_global, axis=1)
        #print(feat_global)
        for i in range(num_stripes):
            feat_local_list[i] = \
                self._normalize(np.vstack(feat_local_list[i]), axis=1)
        return feat_global, feat_local_list

    def extractTest(self, imgs_root_dir, img_fnames, marks, batch_size=32):
        """It is used for checking if the features are correctly extracted
        """
        img_names_batch_list = self._getImageFNamesBatch(imgs_root_dir, \
            img_fnames, batch_size)
        feats = []
        cnt_batch = 0
        num_batches = len(img_names_batch_list)
        for img_names_batch in img_names_batch_list:
            imgs_list = []
            for image_name in img_names_batch:
                img = self._getSamples(image_name)
                imgs_list.append(img)
            imgs_array = np.stack(imgs_list, axis=0)
            feat, _ = self._extract(imgs_array)
            feats.append(feat)
            cnt_batch += 1
            print('Batch: %d/%d' % (cnt_batch, num_batches))
        feats = np.vstack(feats)
        feats = self._normalize(feats, axis=1)
        print(feats.shape)

        # Get cams and ids
        ids = []
        cams = []
        for img_fname in img_fnames:
            id = int(img_fname[:8])
            cam= int(img_fname[9:13])
            ids.append(id)
            cams.append(cam)
        ids = np.asarray(ids)
        cams= np.asarray(cams)

        # Calculate
        marks = np.asarray(marks)
        q_inds = marks == 0
        g_inds = marks == 1

        query_ids=ids[q_inds]
        gallery_ids=ids[g_inds]
        query_cams=cams[q_inds]
        gallery_cams=cams[g_inds]
        q_g_dist = self._compute_dist(feats[q_inds], feats[g_inds], \
            type='euclidean')
        mAP, cmc_scores = self._compute_score(q_g_dist, query_ids, gallery_ids,\
            query_cams, gallery_cams)
        self._print_scores(mAP, cmc_scores)

"""
# Old Version:
def main(args):
    # model_path, gpu_ids, dataset_nm, part_feature_dims
    model_path = args['model_path']
    gpu_ids = args['gpus']
    dataset_nm = args['dataset']
    stripe_feats_dims = args['stripe_feats_dims']
    # Create object
    model_path = osp.join(model_path, dataset_nm)
    extractor = UReIDFeatureExtraction(model_path, gpu_ids, dataset_nm, \
        stripe_feats_dims)
    # Extract features
    imgs_root_dir = args['data_dir']
    batchsize = args['batch_size']
    extractor.extractFeatures(imgs_root_dir, batchsize)

if __name__ == '__main__':
    args = {'model_path': './initial_models', \
            'gpus': (0,), \
            'dataset': 'market1501', \
            'stripe_feats_dims': 2048, \
            #'data_dir': '../DomainAdaptation/dataset/real_training_dataset'}
            'data_dir': './testbed_dataset',\
            'batch_size': 32}
    main(args)
"""
