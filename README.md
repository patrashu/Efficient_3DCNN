# Efficient-3DCNNs
PyTorch Implementation of the article "[Resource Efficient 3D Convolutional Neural Networks](https://arxiv.org/pdf/1904.02422.pdf)", codes and pretrained models.

This repository is forked by https://github.com/okankop/Efficient-3DCNNs


## Requirements

* [PyTorch 1.0.1.post2](http://pytorch.org/)
* OpenCV
* FFmpeg, FFprobe
* Python 3


## Dataset Preparation

### Kinetics

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).
  * Locate test set in ```video_directory/test```.
* Different from the other datasets, we did not extract frames from the videos. Insted, we read the frames directly from videos using OpenCV throughout the training. If you want to extract the frames for Kinetics dataset, please follow the preperation steps in [Kensho Hara's codebase](https://github.com/kenshohara/3D-ResNets-PyTorch). You also need to modify the kinetics.py file in the datasets folder.

* Generate annotation file in json format similar to ActivityNet using ```utils/kinetics_json.py```
  * The CSV files (kinetics_{train, val, test}.csv) are included in the crawler.

```bash
python utils/kinetics_json.py train_csv_path val_csv_path video_dataset_path dst_json_path
```


## Running the code

- training details in run-kinetics.sh
- Baseline to training python code:
```bash
python main.py --root_path <ROOT_PATH> \
	--video_path <DATASET_PATH> \
	--annotation_path <JSON_PATH> \
	--result_path <SAVE_WEIGHT_PATH> \
	--dataset <DATASETS> \
	--n_classes <NUM_CLASSES> \
	--model <TRAIN_MODELS> \
	--width_mult 0.5 \
	--train_crop random \
	--learning_rate 0.1 \
	--sample_duration 16 \
	--downsample 2 \
	--batch_size 64 \
	--n_threads 16 \
	--checkpoint 1 \
	--n_val_samples 1 \
	# --resume_path (if you want resuming train)
```


### Augmentations

There are several augmentation techniques available. Please check spatial_transforms.py and temporal_transforms.py for the details of the augmentation methods.


### Calculating Video Accuracy

In order to calculate viceo accuracy, you should first run the models with '--test' mode in order to create 'val.json'. Then, you need to run 'video_accuracy.py' in utils folder to calculate video accuracies. 

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
