# cross_domain_uReID
[Unspervised Cross-Domain Person Re-Identification: A New Framework](https://ieeexplore.ieee.org/document/8804418).
## Acknowledgements
- The codes are based on the project of [beyond-part-models](https://github.com/huanghoujing/beyond-part-models) by Houjing Huang.
- The framework is mainly based on the papers of [PCB](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yifan_Sun_Beyond_Part_Models_ECCV_2018_paper.pdf) and [SPGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_Image-Image_Domain_Adaptation_CVPR_2018_paper.pdf).
- The translation images of Duke and Market1501 are obtained from [here](https://github.com/Simon4Yan/Learning-via-Translation/tree/master/SPGAN). 

## Dependencies

- python=2.7.15
- pytorch=0.3.1
- py-opencv=3.4.2
- numpy=1.16.2
- scipy=1.1.0
- h5py=2.7.1
- scikit-learn=0.20.0
- pillow=5.3.0

(The requirements are similar with the project of [beyond-part-models](https://github.com/huanghoujing/beyond-part-models). And we recommend to prepare the environment using Anocanda.)

## Framework

## Train

### Prepare training data
- Put the tranlated images to the folder of dataset/synthetic .
- Run `python rename.py imgs_dname flag` under dataset/synthetic (imgs_dname: the folder storing tanslated training data; flag: 1 for Duke, 2 for Market).
- `zip -r duke.zip duke` or `zip -r market.zip market` under dataset/synthetic.
- Transform the dataset based on the style of beyond-part-models (e.g., run `python script/dataset/transform_duke.py --zip_file ../dataset/synthetic/duke.zip --save_dir ../dataset/synthetic/duke` under the folder of beyond-part-models).
- Download the datasets of Duke and Market1501, and put their traing images to the folders named dataset/real/duke and dataset/real/market1501 (Just use the original image filename, tranform is NO need here!).
- Before training the initial model, copy the partitions.pkl and train_test_split.pkl to the folder of dataset/update/duke(market1501)/original. And do NOT modify them!

(The procedure for training data preperation is complex; and we will modify them in a easily way in the update version. One can also get explaination from the README under the sub-folders in dataset).

### Begin to train
- Train the initial model: `sh train_init_model.sh market1501/duke`.
- Update model: `python model-update/ureid_update.py` (to add the constrain of cross-cam., use the function `_clusteringGlobal_v2()` in the line 733 of ureid_update.py).

## Results (without cross-cam.)
| Dataset | mAP | R1 | R5 | R10|
| :------: | :------: | :------: | :------: | :------: | :------: |
| Duke2Market | 54.2 | 78.7 | 87.0 | 90.7 |
| Market2Duke | 49.3 | 68.1 | 75.8 | 81.7 |

```text
@inproceedings{li2019unsupervised,
  title={Unsupervised Cross-Domain Person Re-Identification: A New Framework},
  author={Li, Da and Li, Dangwei and Zhang, Zhang and Wang, Liang and Tan, Tieniu},
  booktitle={2019 IEEE International Conference on Image Processing (ICIP)},
  pages={1222--1226},
  year={2019},
  organization={IEEE}
}
```
