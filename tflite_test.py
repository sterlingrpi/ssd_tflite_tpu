import numpy as np
import tensorflow as tf
import time

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="SSD.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

while(True):
    # get single data sample
    print(input_shape)
    data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    print('data_set shape =', np.shape(data))
    print('input_details =', input_details[0]['index'])
    print('output_details shape =', np.shape(output_details))

    # Test model on data
    start = time.time()
    print('here')
    interpreter.set_tensor(input_details[0]['index'], data)
    print('here2')
    interpreter.invoke()
    print('here3')
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print('here4')
    end = time.time()

    print("test time = ", (end-start)*1000, "ms")
    print("result = ", output_data[0])