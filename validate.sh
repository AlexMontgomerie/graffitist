#!/bin/sh

#VARIABLE DEFINITIONS
#model_name=vgg
#data_dir=/data/imagenet/validation
#graph=models/vgg16_slim_pretrained/vgg16_slim_pretrained.pb
#ckpt=models/vgg16_slim_pretrained/vgg16_slim_pretrained.ckpt

model_name=mobilenet_v1_static_4_4
data_dir=/data/imagenet/validation
graph=models/mobilenet_v1_static_4_4/mobilenet_v1_slim_pretrained_quant.pb
ckpt=models/mobilenet_v1_static_4_4/mobilenet_v1_slim_pretrained_graffitist.ckpt



# RUN VALIDATION SCRIPT
python run_inference_dataset.py \
    --model_name $model_name \
    --data_dir $data_dir \
    --graph_path $graph \
    --ckpt_path $ckpt

# EVALUATE TOP1 AND TOP5
python eval.py outputs/$model_name 

