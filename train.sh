PRETRAINED_G=
PRETRAINED_DOE=
RESULT_DIR=
PARAM=config/param_MV_1600.py
OBSTRUCTION=dirt_raindrop

python train.py --train_optics --result_path $RESULT_DIR --param_file $PARAM --obstruction $OBSTRUCTION --pretrained_DOE $PRETRAINED_DOE --pretrained_G $PRETRAINED_G