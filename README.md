# Graffitist for iDSL

This repo contains scripts for investigating graffitist's quantisation method.

To run, please use the graffitist conda environment. There is also some environment setup required.
```
conda activate graffitist
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.1/bin:$PATH
rm -rf ~/.nv/
```

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
usage: run_inference_dataset.py [-h] --data_dir PATH --model_name MODEL_NAME
                                --graph_path PATH --ckpt_path PATH [-b N] -i
                                STRING -o STRING

Validation Script

optional arguments:
  -h, --help            show this help message and exit
  --data_dir PATH       path to imagenet validation dataset dir
  --model_name MODEL_NAME
                        model name, to be used for outputs
  --graph_path PATH     path to the frozen model.pb
  --ckpt_path PATH      path to the .ckpt files
  -b N, --batch_size N  mini-batch size (default: 100)
  -i STRING, --input_node STRING
                        input node
  -o STRING, --output_node STRING
                        output node
```
Example usage can be found in the `validate.sh` script.


The class probabilities are saved to multiple files in the `outputs/$model_name` directory. Each batch is saved to an individual randomly named csv file, whose contents are as follows:

| Correct Label | Class 1 | Class 2 | ... | Class N |
| ------------- |:-------:|:-------:|:---:|:-------:|
| Class X       | Prob 1  | Prob 2  | ... | Prob N  |

The `eval.py` script can be used to find the top1 and top5 score, as shown in `validate.sh`.

## Run Single Image

An example for running a single image is given in the `run_single_image.py` script. It's usage is as follows:

```
usage: run_single_image.py [-h] --image_path PATH --model_name MODEL_NAME
                           --graph_path PATH --ckpt_path PATH -i STRING -o
                           STRING

Single Image Script

optional arguments:
  -h, --help            show this help message and exit
  --image_path PATH     path to image
  --model_name MODEL_NAME
                        model name, to be used for outputs
  --graph_path PATH     path to the frozen model.pb
  --ckpt_path PATH      path to the .ckpt files
  -i STRING, --input_node STRING
                        input node
  -o STRING, --output_node STRING
                        output node
```
