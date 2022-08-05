import argparse


def parse_opts():
    parser = argparse.ArgumentParser()


    # input opt
    parser.add_argument('--video_path', default='video_kinetics_jpg', type=str, help='Directory path of Videos')
    parser.add_argument('--annotation_path', default='kinetics.json', type=str, help='Annotation file path')
    parser.add_argument('--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--dataset', default='kinetics', type=str, help='Used dataset (kinetics)')
    parser.add_argument('--n_classes', default=8, type=int, help='Number of classes (kinetics: 400, custom: your_dataset_class)')
    parser.add_argument('--n_finetune_classes', default=8, type=int, help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    
    # spatial and temporal transform
    parser.add_argument('--sample_size', default=224, type=int, help='Height and width of inputs')
    parser.add_argument('--resize_w', default=320, type=int, help='Height and width of inputs')
    parser.add_argument('--resize_h', default=240, type=int, help='Height and width of inputs')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of inputs')
    parser.add_argument('--downsample', default=8, type=int, help='How many frames skip in temporal duration, if 2=> idx = (0, 2, .....)')
    parser.add_argument('--train_crop', default='random', type=str, help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')

    # optimizer and lr
    parser.add_argument('--optimizer', default='sgd', type=str, help='Currently only support SGD')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--mean_dataset', default='kinetics', type=str, help='dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--learning_rate', default=0.04, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--lr_steps', default=[40, 55, 65, 70, 200, 250], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')

    # training 
    parser.add_argument('--batch_size', default=16, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=8, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--n_epochs', default=300, type=int, help='Number of total epochs to run')
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--n_val_samples', default=1, type=int, help='Number of validation samples for each activity')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument('--norm_value', default=255, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--model', default='resnet', type=str, help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth', default=50, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--ft_portion', default='complete', type=str, help='The portion of the model to apply fine tuning, either complete or last_layer')
    parser.add_argument('--test_subset', default='val', type=str, help='Used subset in test (val | test)')
    parser.add_argument('--scale_in_test', default=1.0, type=float, help='Spatial scale in test')
    parser.add_argument('--crop_position_in_test', default='c', type=str, help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument('--no_softmax_in_test', action='store_true', help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--checkpoint', default=1, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--no_hflip', action='store_true', help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument('--version', default=1.1, type=float, help='Version of the model')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--groups', default=3, type=int, help='The number of groups at group convolutions at conv layers')
    parser.add_argument('--width_mult', default=0.5, type=float, help='The applied width multiplier to scale number of filters')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')

    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    

    parser.set_defaults(no_train=False)
    parser.set_defaults(no_val=False)
    parser.set_defaults(test=False)
   
    

    args = parser.parse_args()

    return args
