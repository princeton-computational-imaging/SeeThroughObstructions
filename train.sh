PRETRAINED_G=None
PRETRAINED_DOE=None
RESULT_DIR=ckpt/E2E_MV2400
PARAM=config/param_MV_2400.py
OBSTRUCTION=dirt_raindrop

conda activate SeeThroughObstruction

python train.py --train_optics --result_path $RESULT_DIR --param_file $PARAM --obstruction $OBSTRUCTION --pretrained_DOE $PRETRAINED_DOE --pretrained_G $PRETRAINED_G