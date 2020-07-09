Train a single shot objectdetector (SSD) on custom data/classes and convert to tflite, tflite full int8, and tpu.

Copy adapted from https://github.com/ChunML/ssd-tf2

The main difference is the model is defined using tensorflow's functional API (not as a subclass). This makes the resulting trained model compatible wtih the TPU.
