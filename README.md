# Graffitist for iDSL



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

In general, the Input Shape and Quantisation Widths are the same are across networks and don't need to be changed. An example for running static quantisation can be found in the `quantise_static.sh` script.

```
./quantise_static.sh
```

## Retrained Quantisation (TODO)


## Run Validation
