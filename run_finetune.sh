


python train.py \
	--gpu 0 \
	--dataset 'KITTI' \
	--data_path './data_scene_flow' \
	--batch_size 2 \
	--max_epoch 100 \
	--learning_rate 0.0001 \
	--best_checkpoint False \
	--pretrained_model './model/DeepLiDARFlow-FT3D'
     

