#!/bin/sh

#VARIABLE DEFINITIONS
#mdir=models/vgg16_slim_pretrained
#in_graph=$mdir/vgg16_slim_pretrained.pb
#opt_graph=$mdir/vgg16_slim_pretrained_opt.pb
#quant_graph=$mdir/vgg16_slim_pretrained_quant.pb
#input_node=input
#output_node=vgg_16/fc8/squeezed
#input_shape=224,224,3
#wb=-8; ab=-8; lb=-16; rb=8; pb=8; prb=8;

#VARIABLE DEFINITIONS
mdir=models/mobilenet_v1_static_4_4
in_graph=$mdir/mobilenet_v1_slim_pretrained.pb
opt_graph=$mdir/mobilenet_v1_slim_pretrained_opt.pb
quant_graph=$mdir/mobilenet_v1_slim_pretrained_quant.pb
input_node=input
output_node=MobilenetV1/Predictions/Softmax
input_shape=224,224,3
wb=-4; ab=-8; lb=-8; rb=8; pb=8; prb=8;


# Limit GPUs Used
export CUDA_VISIBLE_DEVICES="0,1"

# Optimise Inference Graph
python graffitist/graffitize.pyc \
    --in_graph $in_graph \
    --out_graph $opt_graph \
    --inputs $input_node \
    --outputs $output_node \
    --input_shape $input_shape \
    --transforms 'fix_input_shape' \
                 'fold_batch_norms' \
                 'remove_training_nodes' \
                 'strip_unused_nodes' \
                 'preprocess_layers'

# Quantise Inference Graph (Static)
python graffitist/graffitize.pyc \
    --in_graph $in_graph \
    --out_graph $quant_graph \
    --inputs $input_node \
    --outputs $output_node \
    --input_shape $input_shape \
    --transforms 'fix_input_shape' \
                 'fold_batch_norms' \
                 'remove_training_nodes' \
                 'strip_unused_nodes' \
                 'preprocess_layers' \
                 'quantize(weight_bits='$wb', activation_bits='$ab', layer_bits='$lb', relu_bits='$rb', avgpool_bits='$pb', avgpool_reciprocal_bits='$prb')'
