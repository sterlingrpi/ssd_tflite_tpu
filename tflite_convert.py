import tensorflow as tf
import numpy as np
import time

def representative_data_gen():
    for i in range(100):
        input_value = np.array(np.random.random_sample([1, 300, 300, 3]), dtype=np.float32)
        yield [input_value]

ssd = tf.keras.models.load_model('./models/ssd.h5')
data = np.array(np.random.random_sample([1,300,300,3]), dtype=np.float32)
confs, locs = ssd.predict(data)
#print(np.shape(confs))
#print(np.shape(locs))

converter = tf.lite.TFLiteConverter.from_keras_model(ssd)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_data_gen

tflite_model = converter.convert()
open('./models/SSD.tflite', "wb").write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="./models/SSD.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
while(True):
    data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    print('data_set shape =', np.shape(data))

    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    outputs = []
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        outputs.append(output_data)
    end = time.time()
    print("test time = ", (end-start)*1000, "ms")

    for i, output in enumerate(outputs):
        print('output shape', i, '=', np.shape(output))
