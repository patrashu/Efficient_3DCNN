from __future__ import print_function, division
import os
import cv2
import sys
import json
import pandas as pd

def convert_csv_to_dict(csv_path, dataset_path, subset):
    data = pd.read_csv(csv_path)
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.iloc[i, :]
        basename = '%s_%s_%s.mp4' % (row['youtube_id'],
                                 '%06d' % row['time_start'],
                                 '%06d' % row['time_end'])
        keys.append(basename)
        if subset != 'testing':
            key_labels.append(row['label'])

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        if subset != 'testing':
            label = key_labels[i]
            database[key]['annotations'] = {'label': label}
        else:
            database[key]['annotations'] = {}
        # Add n_frames
        sub = 'train' if subset == 'training' else 'validation'
        video_path = os.path.join(dataset_path, sub, label, key)
        cap = cv2.VideoCapture(video_path)
        print(video_path)
        print(cap.get(7))
        database[key]['n_frames'] = int(cap.get(7)) # Returns the number of frames in the video
        cap.release()

    cap.release()
    return database

def load_labels(train_csv_path):
    data = pd.read_csv(train_csv_path)
    return data['label'].unique().tolist()

def convert_kinetics_csv_to_activitynet_json(train_csv_path, val_csv_path, dataset_path, dst_json_path):
    labels = load_labels(val_csv_path)
    val_database = convert_csv_to_dict(val_csv_path, dataset_path, 'validation')
    train_database = convert_csv_to_dict(train_csv_path, dataset_path, 'training')

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {"training": train_database, "validation": val_database}

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)

if __name__=="__main__":
  train_csv_path = sys.argv[1]
  val_csv_path = sys.argv[2]
  dataset_path = sys.argv[3]
  dst_json_path = sys.argv[4]

  convert_kinetics_csv_to_activitynet_json(
    train_csv_path, val_csv_path, dataset_path, dst_json_path)
