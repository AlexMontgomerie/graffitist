#!/bin/sh

#VARIABLE DEFINITIONS
mdir=models/mobilenet_v1_quantise_retrain
in_metagraph=$mdir/mobilenet_v1_slim_pretrained.ckpt.meta
in_graph=$mdir/mobilenet_v1_slim_pretrained.pb
opt_graph=$mdir/mobilenet_v1_slim_pretrained_opt.pb
trainquant_graph=$mdir/mobilenet_v1_slim_pretrained_trainquant.pb
infquant_graph=$mdir/mobilenet_v1_slim_pretrained_infquant.pb

input_node=input
output_node=MobilenetV1/Predictions/Softmax
input_shape=224,224,3
[ "$INT4_MODE" = 1 ] && wb=-4 || wb=-8; ab=-8; lb=-16; rb=8; pb=8; prb=8;
first_layer=MobilenetV1/MobilenetV1/Conv2d_0/Conv2D
last_layer=MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D

# Limit GPUs Used
export CUDA_VISIBLE_DEVICES="3"

# Generate quantised training graph
python graffitist/graffitize.pyc \
    --in_graph $in_metagraph \
    --out_graph $trainquant_graph \
    --inputs $input_node \
    --outputs $output_node \
    --input_shape $input_shape \
    --transforms 'fix_input_shape' \
                 'fold_batch_norms(is_training=True)' \
                 'preprocess_layers' \
                 'quantize(is_training=True, weight_bits='$wb', activation_bits='$ab', layer_bits='$lb', relu_bits='$rb', avgpool_bits='$pb', avgpool_reciprocal_bits='$prb', first_layer='$first_layer', last_layer='$last_layer')'

# Retrain Network
python scripts/train_imagenet_tf.py \
          --train_dir       /data/imagenet/train \
          --val_dir         /data/imagenet/validation \
          --ckpt_dir        $mdir/mobilenet_v1_slim_pretrained_graffitist.ckpt \
          --image_size      224 

# Generate inference graph
#python graffitist/graffitize.pyc \
#    --in_graph $in_graph \
#    --out_graph $quant_graph \
#    --inputs $input_node \
#    --outputs $output_node \
#    --input_shape $input_shape \
#    --transforms 'fix_input_shape' \
#                 'fold_batch_norms' \
#                 'remove_training_nodes' \
#                 'strip_unused_nodes' \
#                 'preprocess_layers' \
#                 'quantize(weight_bits='$wb', activation_bits='$ab', layer_bits='$lb', relu_bits='$rb', avgpool_bits='$pb', avgpool_reciprocal_bits='$prb')'
