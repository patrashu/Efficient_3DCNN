## train 
python main.py \
	--video_path "hongkong/hongkong/" \
	--annotation_path "hongkong/annot.json" \
	--result_path "results" \
	# --dataset "kinetics" \
	# --n_classes 8 \
	# --sample_size 224 \
	# --model "resnet" \
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

## resume train with weight
python main.py \
	--video_path "hongkong/hongkong/" \
	--annotation_path "hongkong/annot.json" \
	--result_path "results" \
	--resume_path <RESUME_PATH>
	# --dataset "kinetics" \
	# --n_classes 8 \
	# --sample_size 224 \
	# --model "resnet" \
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
	

## test 
## if you want to test, check opt.py (no_train, no_val, test)
python main.py \
	--video_path "hongkong/hongkong/" \
	--annotation_path "hongkong/annot.json" \
	--result_path "results" \
	--pretrain_path <PRETRAIN_PATH>
	# --dataset "kinetics" \
	# --n_classes 8 \
	# --sample_size 224 \
	# --model "resnet" \
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


## inference with parameter(second)
python video_inference_ffmg.py \
	--video_path "hongkong/hongkong/train/" \
	--annotation_path "hongkong/annot.json" \
	--result_path "results" \
	# --dataset "kinetics" \
	# --n_classes 8 \
	# --sample_size 240 \
	# --model "resnet" \
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