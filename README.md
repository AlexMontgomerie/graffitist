# Graffitist for iDSL

This repo contains scripts for investigating graffitist's quantisation method.

## Static Quantisation

The static quantisation method requires the following information.

 - Model Directory
 - Input Graph
 - Optimised Graph
 - Quantised Graph
 - Input Node
 - Output Node
 - Input Shape
 - Quantisation Widths

To run static quantisation, please reference the following script. More details can be found in `README_GRAFFITIST.md`.
```
quantise_static.sh
```

## Retrained Quantisation (TODO)


## Run Validation

To run validation, use the `run_inference_dataset.py` script. 
```
python run_inference_dataset.py \
    --data_dir        <path_to_imagenet_validation_dataset_dir> \
    --model_name      <cnn_model_name> \
    --graph_path      <path_to_.pb_file> \
    --ckpt_path       <path_to_.ckpt_file> \
    --batch_size      <N>
```
Example usage can be found in the `validate.sh` script.


The class probabilities are saved to multiple files in the `outputs/$model_name` directory. Each batch is saved to an individual randomly named csv file, whose contents are as follows:

| Correct Label | Class 1 | Class 2 | ... | Class N |
| ------------- |:-------:|:-------:|:---:|:-------:|
| Class X       | Prob 1  | Prob 2  | ... | Prob N  |

The `eval.py` script can be used to find the top1 and top5 score, as shown in `validate.sh`.
