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
python main.py --root_path <ROOT_PATH> \
	--video_path <DATASET_PATH> \
	--annotation_path <JSON_PATH> \
	--result_path <SAVE_WEIGHT_PATH> \
	--dataset <DATASETS> \
	# --n_classes <NUM_CLASSES> \
	# --sample_size 224 \
	# --model "resnet"
	# --width_mult 0.5 \
	# --train_crop random \
	# --sample_duration 16 \
	# --downsample 8 \
	# --batch_size 16 \
	# --n_epochs 300 \
	# --n_thread 8 \
	# --checkpoint 1 \
	# --n_val_samples 1 \
	# --norm_value 255 \
	--resume_path (if you want resuming train)
```
## Run Test
- Enter opts.py, and Set test 'True'
```bash
python main.py --root_path <ROOT_PATH> \
	--video_path <DATASET_PATH> \
	--annotation_path <JSON_PATH> \
	--result_path <SAVE_WEIGHT_PATH> \
	--dataset <DATASETS> \
	# --n_classes <NUM_CLASSES> \
	# --model <TRAIN_MODELS> \
	# --width_mult 0.5 \
	# --train_crop random \
	# --learning_rate 0.04 \
	# --sample_duration 16 \
	# --downsample 8 \
	# --batch_size 16 \
	# --n_threads 8 \
	# --checkpoint 1 \
	# --n_val_samples 1 \
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
