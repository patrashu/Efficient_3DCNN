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
### Custom Dataset
- If you want to train with Custom Dataset, set your dataset like kinetics dataset format
- First, make [csv_file](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics/data) like this
- Second, run this code, and you can get video clip like "video_name_000xxx_000xxx.mp4" format
```
python hongkong\download.py --input_csv <CSV_FILE> --output_dir <OUTPUT_DIR> --trim-format <TRIM_FORMAT> --tmp_dir <RAW_DATASET_DIR>
```
- Finally, run this code. If you set incorrect path, video total frames are not extract. So, Keep in mind check your path
```
python utils/kinetics_json.py train_csv_path val_csv_path video_dataset_path dst_json_path
```
- To run train.py, Set your dataset directory like this
```bash
├── datasets
│   ├── train
│   │   ├── CLASS_A
│   │   │   └── train_video1.mp4
│   │   │   └── train_video2.mp4
│   │   │   └── train_video3.mp4
│   │   │    ....
│   │   └── CLASS_B
│   │   └── CLASS_C
│   │   └── CLASS_D
│   │   ....
│   └── val
│   │   ├── CLASS_A
│   │   │   └── val_video1.mp4
│   │   │   └── val_video2.mp4
│   │   │   └── val_video3.mp4
│   │   │    ....
│   │   └── CLASS_B
│   │   └── CLASS_C
│   │   └── CLASS_D
│   └── train.csv
│   └── val.csv
│   └── test.csv
│   └── annot.json
```

## Run Train

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
## Run Test
- Enter opts.py, and Set test 'True'
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
