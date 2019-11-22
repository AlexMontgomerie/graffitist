#!/bin/sh
model_name=mobilenet_v1_static_4_4
image_path=/data/imagenet/validation/n01986214/ILSVRC2012_val_00023951.JPEG
graph=models/mobilenet_v1_static_4_4/mobilenet_v1_slim_pretrained_quant.pb
ckpt=models/mobilenet_v1_static_4_4/mobilenet_v1_slim_pretrained_graffitist.ckpt
input_node=input
output_node=MobilenetV1/Predictions/Softmax

# RUN SINGLE IMAGE SCRIPT
python run_single_image.py \
    --model_name $model_name \
    --image_path $image_path \
    --graph_path $graph \
    --ckpt_path $ckpt \
    -i $input_node \
    -o $output_node

