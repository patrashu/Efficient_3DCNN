# Efficient-3DCNNs
PyTorch Implementation of the article "[Resource Efficient 3D Convolutional Neural Networks](https://arxiv.org/pdf/1904.02422.pdf)", codes and pretrained models.

This repository is forked by https://github.com/okankop/Efficient-3DCNNs


## Requirements

* [Pytorch 1.12 + CUDA 11.3](https://pytorch.org/get-started/locally/)
* OpenCV
* FFmpeg, FFprobe
* Python 3.8


## Run Train
- Training environment: Window11, NVIDIA A4000(16GB)*2
- training details in run-kinetics.sh
```bash
python main.py \
	--video_path <DATASET_PATH> \
	--annotation_path <JSON_PATH> \
	--result_path <SAVE_WEIGHT_PATH> \
```
## Run Test
- Enter opts.py, and Set test 'True'
```bash
python main.py \
	--video_path <DATASET_PATH> \
	--annotation_path <JSON_PATH> \
	--result_path <SAVE_WEIGHT_PATH> \
	--pretrain_path <PRETRAIN_PATH>
```
## Inference
- 
```
python video_inference.py \
	--video_path <DATASET_PATH> \
	--annotation_path <JSON_PATH> \
	--result_path <RESULT_PATH> \
	--pretrain_path <PRETRAIN_PATH>
```


### Augmentations

There are several augmentation techniques available. Please check spatial_transforms.py and temporal_transforms.py for the details of the augmentation methods.


### Calculating FLOPs

In order to calculate FLOPs, run the file 'calculate_FLOP.py'. You need to fist uncomment the desired model in the file. 

## Citation

Please cite the following article if you use this code or pre-trained models:

```bibtex
@inproceedings{kopuklu2019resource,
  title={Resource efficient 3d convolutional neural networks},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Kose, Neslihan and Gunduz, Ahmet and Rigoll, Gerhard},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)},
  pages={1910--1919},
  year={2019},
  organization={IEEE}
}
```

## Acknowledgement
We thank Kensho Hara for releasing his [codebase](https://github.com/kenshohara/3D-ResNets-PyTorch), which we build our work on top.
