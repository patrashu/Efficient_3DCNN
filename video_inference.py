from random import sample
from re import L
import cv2
import torch
import os
from torch import nn
import torch.nn.functional as F

import ffmpeg
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, writers
from celluloid import Camera

from opts import parse_opts
from lib.models.select_model import generate_model
from lib.utils.utils import *
from lib.transforms.spatial_transforms import *
from lib.transforms.temporal_transforms import *

def cal_acc(queue):
    sorted_scores, locs = torch.topk(queue, k=8)

    return sorted_scores, locs


class Video(animation.FuncAnimation):
    def __init__(
        self, device=0, fig=None, frames=None,
        interval=80, repeat_delay=5, blit=False, **kwargs    
    ) -> None:

        if fig is None:
            self.fig = plt.figure(figsize=(10, 4), dpi=100)
            spec = self.fig.add_gridspec(nrows=1, ncols=4)
            
            self.ax0 = self.fig.add_subplot(spec[0, :2])
            self.ax0.axis('off')
            self.ax1 = self.fig.add_subplot(spec[0, 3])


        super(Video, self).__init__(
                self.fig, self.updateFrame, init_func=self.start, frames=frames, 
                interval=interval, blit=blit,repeat_delay=repeat_delay, save_count=200, **kwargs
            )

        self.cap = cv2.VideoCapture(device)
        self.queue = []
        self.cnt = 0
        self.y_label = [v for k, v in class_names.items()]
        self.x_label = [0 for _ in range(len(class_names))]
        self.temporal_clip = opt.temporal_clip * 30
        self.split = int(self.temporal_clip // 2)
        self.odd = True if self.split%2 != 0 else False
        self.cnt_flag = True

    def start(self):
        ret, self.frame = self.cap.read()
        if ret:
            self.im0 = self.ax0.imshow(
                cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), aspect='auto'
            )
            self.im1 = self.ax1.barh(self.y_label, self.x_label)
    
    def updateFrame(self, k):
        label = ''
        # load and save Video
        total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        checkpoint = torch.load(model_path)
        model, _ = generate_model(opt)
        model.load_state_dict(checkpoint['state_dict'])

        ret, self.frame = self.cap.read()
        colors = sns.color_palette('hls',len(class_names))

        if ret:
            if self.cnt >= total_frame-self.temporal_clip-1:
                self.close()
            self.cnt += 1

            pil_frame = Image.fromarray(self.frame)
            self.queue.append(pil_frame)

            if len(self.queue) == self.temporal_clip:
                buffer = [transform(image) for image in self.queue]

                buffer = torch.stack(buffer, 0).unsqueeze(0).permute(0, 2, 1, 3, 4)
                output = model(buffer)
                output = F.softmax(output, dim=1)
                sorted_scores, locs = cal_acc(output)
                
                result_label = class_names[int(locs[0][0])]
                score = float(sorted_scores[0][0])
                label += result_label

                for i, num in enumerate(locs[0]):
                    self.x_label[num] = float(sorted_scores[0][i])
                
            if len(self.queue) > self.temporal_clip-1:
                if self.odd and self.cnt_flag:
                    for _ in range(self.split+1):
                        self.queue.pop(0)
                        print('Delete', len(self.queue))
                    self.cnt_flag=False
                else:
                    for _ in range(self.split):
                        self.queue.pop(0)
                        print('Delete', len(self.queue))
                    self.cnt_flag=True
                label = f'{label} ({score})'

            cv2.putText(
                self.frame, label, (50, 700), cv2.FONT_HERSHEY_COMPLEX, 
                1, (0, 0, 0), 1
            )

            self.im0.set_array(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

            self.ax1.cla()
            self.im1 = self.ax1.barh(self.y_label, self.x_label, color=colors)
            self.ax1.set_xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40])
            for p in self.ax1.patches:
                x, y, w, h = p.get_bbox().bounds
                self.ax1.text(w*1.07, y+h/2, "%.1f %%"%(w*100), va='center', fontsize=7)

    
    def close(self):
        if self.cap.isOpened():
            self.cap.release()
            return
        print('finish capture')


if __name__ == '__main__':
    opt = parse_opts()
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.width_mult) + 'x',
                               'downsample_' + str(opt.sample_duration)])
    

    base_path = './hongkong/hongkong/'
    video_path = "validation/drilling/IMG_0120_000063_000071.mp4"
    model_path = 'results\\kinetics_resnet_0.5x_downsample_16_checkpoint179_acc.pth'
        
    transform = Compose([
        Resize(opt.resize_h, opt.resize_h),
        CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value),
        # Normalize(opt.mean, opt.std),        
    ])

    # temporal terms in second
    # temporal_clip = opt.temporal_clip

    class_names = {
        0: 'travelling', 
        1: 'lifting brick', 
        2: 'lifting rebar', 
        3: 'measuring rebar', 
        4: 'tying rebar', 
        5: 'hammering', 
        6: 'drilling', 
        7: 'idle'
    }
    file_name = video_path.split('/')[1]
    
    camera = Video(device=base_path+video_path)
    plt.show()
    Writer = FFMpegWriter(fps=30, bitrate=1800)
    camera.save(f'{file_name}.avi', writer=Writer)
    # plt.show()
    camera.close()
    