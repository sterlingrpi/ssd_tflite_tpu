from tensorflow.keras.applications import VGG16, MobileNet, mobilenet_v2
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np

def get_VGG16_SSD(image_size, num_classes):
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], image_size[2]))
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2, 2, padding='same')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2, 2, padding='same')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2, 2, padding='same')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    features_1 = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(2, 2, padding='same')(features_1)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    output_vgg16_conv4 = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    vgg16_conv4 = tf.keras.Model(inputs=inputs, outputs=output_vgg16_conv4)

    x = layers.BatchNormalization(beta_initializer='glorot_uniform', gamma_initializer='glorot_uniform')(features_1)
    conf4 = layers.Conv2D(4 * num_classes, kernel_size=3, padding='same')(x)  # conf 4th block
    conf4 = layers.Reshape((conf4.shape[1]*conf4.shape[2]*conf4.shape[3]//num_classes, num_classes))(conf4)
    loc4 = layers.Conv2D(4 * 4, kernel_size=3, padding='same')(x)  # loc 4th block
    loc4 = layers.Reshape((loc4.shape[1]*loc4.shape[2]*loc4.shape[3]//4, 4))(loc4)

    x = layers.MaxPool2D(3, 1, padding='same')(output_vgg16_conv4)
    x = layers.Conv2D(1024, 3, padding='same', dilation_rate=6, activation='relu')(x)
    x = layers.Conv2D(1024, 1, padding='same', activation='relu')(x)
    conf7 = layers.Conv2D(6 * num_classes, kernel_size=3, padding='same')(x)  # conf 7th block
    conf7 = layers.Reshape((conf7.shape[1] * conf7.shape[2] * conf7.shape[3] // num_classes, num_classes))(conf7)
    loc7 = layers.Conv2D(6 * 4, kernel_size=3, padding='same')(x)  # loc 7th block
    loc7 = layers.Reshape((loc7.shape[1]*loc7.shape[2]*loc7.shape[3]//4, 4))(loc7)

    x = layers.Conv2D(256, 1, activation='relu')(x)  # extra_layers 8th block output shape: B, 512, 10, 10
    x = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu')(x)
    conf8 = layers.Conv2D(6 * num_classes, kernel_size=3, padding='same')(x)  # conf 8th block
    conf8 = layers.Reshape((conf8.shape[1] * conf8.shape[2] * conf8.shape[3] // num_classes, num_classes))(conf8)
    loc8 = layers.Conv2D(6 * 4, kernel_size=3, padding='same')(x)  # loc 8th block
    loc8 = layers.Reshape((loc8.shape[1] * loc8.shape[2] * loc8.shape[3] // 4, 4))(loc8)

    x = layers.Conv2D(128, 1, activation='relu')(x)  # extra_layers 9th block output shape: B, 256, 5, 5
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    conf9 = layers.Conv2D(6 * num_classes, kernel_size=3, padding='same')(x)  # conf 9th block
    conf9 = layers.Reshape((conf9.shape[1] * conf9.shape[2] * conf9.shape[3] // num_classes, num_classes))(conf9)
    loc9 = layers.Conv2D(6 * 4, kernel_size=3, padding='same')(x)  # loc 9th block
    loc9 = layers.Reshape((loc9.shape[1] * loc9.shape[2] * loc9.shape[3] // 4, 4))(loc9)

    x = layers.Conv2D(128, 1, activation='relu')(x)  # extra_layers 10th block output shape: B, 256, 3, 3
    x = layers.Conv2D(256, 3, activation='relu')(x)
    conf10 = layers.Conv2D(4 * num_classes, kernel_size=3, padding='same')(x)  # conf 10th block
    conf10 = layers.Reshape((conf10.shape[1] * conf10.shape[2] * conf10.shape[3] // num_classes, num_classes))(conf10)
    loc10 = layers.Conv2D(4 * 4, kernel_size=3, padding='same')(x)  # loc 10th block
    loc10 = layers.Reshape((loc10.shape[1] * loc10.shape[2] * loc10.shape[3] // 4, 4))(loc10)

    x = layers.Conv2D(128, 1, activation='relu')(x)  # extra_layers 11th block output shape: B, 256, 1, 1
    x = layers.Conv2D(256, 3, activation='relu')(x)
    conf11 = layers.Conv2D(4 * num_classes, kernel_size=3, padding='same')(x)  # conf 11th block
    conf11 = layers.Reshape((conf11.shape[1] * conf11.shape[2] * conf11.shape[3] // num_classes, num_classes))(conf11)
    loc11 = layers.Conv2D(4 * 4, kernel_size=3, padding='same')(x)  # loc 11th block
    loc11 = layers.Reshape((loc11.shape[1] * loc11.shape[2] * loc11.shape[3] // 4, 4))(loc11)

    x = layers.Conv2D(128, 1, activation='relu')(x)  # extra_layers 12th block output shape: B, 256, 1, 1
    x = layers.Conv2D(256, 4, activation='relu')(x)
    conf12 = layers.Conv2D(4 * num_classes, kernel_size=1)(x)  # conf 12th block
    conf12 = layers.Reshape((conf12.shape[1] * conf12.shape[2] * conf12.shape[3] // num_classes, num_classes))(conf12)
    loc12 = layers.Conv2D(4 * 4, kernel_size=1)(x)  # loc 12th block
    loc12 = layers.Reshape((loc12.shape[1] * loc12.shape[2] * loc12.shape[3] // 4, 4))(loc12)

    confs = layers.concatenate([conf4, conf7, conf8, conf9, conf10, conf11, conf12], axis=1)
    locs = layers.concatenate([loc4, loc7, loc8, loc9, loc10, loc11, loc12], axis=1)
    model = tf.keras.Model(inputs=inputs, outputs=[confs, locs])

    model_VGG16 = VGG16(weights='imagenet')
    for i in range(18):
        model.get_layer(index=i).set_weights(model_VGG16.get_layer(index=i).get_weights())

    return model

def get_VGG16_SSD_300(image_size, num_classes):
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], image_size[2]), name='input_1_base')
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2d_1_base')(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2d_2_base')(x)
    x = layers.MaxPool2D(2, 2, padding='same', name='max_pooling2d_1_base')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2d_3_base')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2d_4_base')(x)
    x = layers.MaxPool2D(2, 2, padding='same', name='max_pooling2d_2_base')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name='conv2d_5_base')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name='conv2d_6_base')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', name='conv2d_7_base')(x)
    x = layers.MaxPool2D(2, 2, padding='same', name='max_pooling2d_3_base')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv2d_8_base')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv2d_9_base')(x)
    block_4_features = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv2d_10_base')(x)
    x = layers.MaxPool2D(2, 2, padding='same', name='max_pooling2d_4_base')(block_4_features)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv2d_11_base')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv2d_12_base')(x)
    output_vgg16 = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv2d_13_base')(x)

    x = layers.BatchNormalization(beta_initializer='glorot_uniform', gamma_initializer='glorot_uniform')(block_4_features)
    conf4 = layers.Conv2D(4*num_classes, kernel_size=3, padding='same')(x)  # conf 4th block0. Originally had 4x the num of filters but TPU compiler crashes
    conf4 = layers.Reshape((conf4.shape[1]*conf4.shape[2]*conf4.shape[3]//num_classes, num_classes))(conf4)
    loc4 = layers.Conv2D(4*4, kernel_size=3, padding='same')(x)  # loc 4th block
    loc4 = layers.Reshape((loc4.shape[1]*loc4.shape[2]*loc4.shape[3]//4, 4))(loc4)

    x = layers.MaxPool2D(3, 1, padding='same')(output_vgg16)
    x = layers.Conv2D(1024, 3, padding='same', dilation_rate=6, activation='relu')(x)
    x = layers.Conv2D(1024, 1, padding='same', activation='relu')(x)
    conf7 = layers.Conv2D(6 * num_classes, kernel_size=3, padding='same')(x)  # conf 7th block
    conf7 = layers.Reshape((conf7.shape[1] * conf7.shape[2] * conf7.shape[3] // num_classes, num_classes))(conf7)
    loc7 = layers.Conv2D(6 * 4, kernel_size=3, padding='same')(x)  # loc 7th block
    loc7 = layers.Reshape((loc7.shape[1]*loc7.shape[2]*loc7.shape[3]//4, 4))(loc7)

    x = layers.Conv2D(256, 1, activation='relu')(x)  # extra_layers 8th block output shape: B, 512, 10, 10
    x = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu')(x)
    conf8 = layers.Conv2D(6 * num_classes, kernel_size=3, padding='same')(x)  # conf 8th block
    conf8 = layers.Reshape((conf8.shape[1] * conf8.shape[2] * conf8.shape[3] // num_classes, num_classes))(conf8)
    loc8 = layers.Conv2D(6 * 4, kernel_size=3, padding='same')(x)  # loc 8th block
    loc8 = layers.Reshape((loc8.shape[1] * loc8.shape[2] * loc8.shape[3] // 4, 4))(loc8)

    x = layers.Conv2D(128, 1, activation='relu')(x)  # extra_layers 9th block output shape: B, 256, 5, 5
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    conf9 = layers.Conv2D(6 * num_classes, kernel_size=3, padding='same')(x)  # conf 9th block
    conf9 = layers.Reshape((conf9.shape[1] * conf9.shape[2] * conf9.shape[3] // num_classes, num_classes))(conf9)
    loc9 = layers.Conv2D(6 * 4, kernel_size=3, padding='same')(x)  # loc 9th block
    loc9 = layers.Reshape((loc9.shape[1] * loc9.shape[2] * loc9.shape[3] // 4, 4))(loc9)

    x = layers.Conv2D(128, 1, activation='relu')(x)  # extra_layers 10th block output shape: B, 256, 3, 3
    x = layers.Conv2D(256, 3, activation='relu')(x)
    conf10 = layers.Conv2D(4 * num_classes, kernel_size=3, padding='same')(x)  # conf 10th block
    conf10 = layers.Reshape((conf10.shape[1] * conf10.shape[2] * conf10.shape[3] // num_classes, num_classes))(conf10)
    loc10 = layers.Conv2D(4 * 4, kernel_size=3, padding='same')(x)  # loc 10th block
    loc10 = layers.Reshape((loc10.shape[1] * loc10.shape[2] * loc10.shape[3] // 4, 4))(loc10)

    x = layers.Conv2D(128, 1, activation='relu')(x)  # extra_layers 11th block output shape: B, 256, 1, 1
    x = layers.Conv2D(256, 3, activation='relu')(x)
    conf11 = layers.Conv2D(4 * num_classes, kernel_size=3, padding='same')(x)  # conf 11th block
    conf11 = layers.Reshape((conf11.shape[1] * conf11.shape[2] * conf11.shape[3] // num_classes, num_classes))(conf11)
    loc11 = layers.Conv2D(4 * 4, kernel_size=3, padding='same')(x)  # loc 11th block
    loc11 = layers.Reshape((loc11.shape[1] * loc11.shape[2] * loc11.shape[3] // 4, 4))(loc11)

    confs = layers.concatenate([conf4, conf7, conf8, conf9, conf10, conf11], axis=1)
    locs = layers.concatenate([loc4, loc7, loc8, loc9, loc10, loc11], axis=1)
    model = tf.keras.Model(inputs=inputs, outputs=[confs, locs])

    model_VGG16 = VGG16(weights='imagenet')
    for i in range(18):
        model.get_layer(index=i).set_weights(model_VGG16.get_layer(index=i).get_weights())

    return model

def get_mobilenet_SSD(image_size, num_classes):
    mobilenet = MobileNet(input_shape=image_size, include_top=False, weights="imagenet")
    for layer in mobilenet.layers:
        layer._name = layer.name + '_base'

    x = layers.BatchNormalization(beta_initializer='glorot_uniform', gamma_initializer='glorot_uniform')(mobilenet.get_layer(name='conv_pad_6_base').output)
    conf1 = layers.Conv2D(4*num_classes, kernel_size=3, padding='same')(x)
    conf1 = layers.Reshape((conf1.shape[1]*conf1.shape[2]*conf1.shape[3]//num_classes, num_classes))(conf1)
    loc1 = layers.Conv2D(4*4, kernel_size=3, padding='same')(x)
    loc1 = layers.Reshape((loc1.shape[1]*loc1.shape[2]*loc1.shape[3]//4, 4))(loc1)

    x = layers.MaxPool2D(3, 1, padding='same')(mobilenet.get_layer(name='conv_pad_12_base').output)
    x = layers.Conv2D(1024, 3, padding='same', dilation_rate=6, activation='relu')(x)
    x = layers.Conv2D(1024, 1, padding='same', activation='relu')(x)
    conf2 = layers.Conv2D(6 * num_classes, kernel_size=3, padding='same')(x)
    conf2 = layers.Reshape((conf2.shape[1] * conf2.shape[2] * conf2.shape[3] // num_classes, num_classes))(conf2)
    loc2 = layers.Conv2D(6 * 4, kernel_size=3, padding='same')(x)
    loc2 = layers.Reshape((loc2.shape[1]*loc2.shape[2]*loc2.shape[3]//4, 4))(loc2)

    x = layers.Conv2D(256, 1, activation='relu')(x)
    x = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu')(x)
    conf3 = layers.Conv2D(6 * num_classes, kernel_size=3, padding='same')(x)
    conf3 = layers.Reshape((conf3.shape[1] * conf3.shape[2] * conf3.shape[3] // num_classes, num_classes))(conf3)
    loc3 = layers.Conv2D(6 * 4, kernel_size=3, padding='same')(x)
    loc3 = layers.Reshape((loc3.shape[1] * loc3.shape[2] * loc3.shape[3] // 4, 4))(loc3)

    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    conf4 = layers.Conv2D(6 * num_classes, kernel_size=3, padding='same')(x)
    conf4 = layers.Reshape((conf4.shape[1] * conf4.shape[2] * conf4.shape[3] // num_classes, num_classes))(conf4)
    loc4 = layers.Conv2D(6 * 4, kernel_size=3, padding='same')(x)
    loc4 = layers.Reshape((loc4.shape[1] * loc4.shape[2] * loc4.shape[3] // 4, 4))(loc4)

    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, activation='relu')(x)
    conf5 = layers.Conv2D(4 * num_classes, kernel_size=3, padding='same')(x)
    conf5 = layers.Reshape((conf5.shape[1] * conf5.shape[2] * conf5.shape[3] // num_classes, num_classes))(conf5)
    loc5 = layers.Conv2D(4 * 4, kernel_size=3, padding='same')(x)
    loc5 = layers.Reshape((loc5.shape[1] * loc5.shape[2] * loc5.shape[3] // 4, 4))(loc5)

    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, activation='relu')(x)
    conf6 = layers.Conv2D(4 * num_classes, kernel_size=3, padding='same')(x)
    conf6 = layers.Reshape((conf6.shape[1] * conf6.shape[2] * conf6.shape[3] // num_classes, num_classes))(conf6)
    loc6 = layers.Conv2D(4 * 4, kernel_size=3, padding='same')(x)
    loc6 = layers.Reshape((loc6.shape[1] * loc6.shape[2] * loc6.shape[3] // 4, 4))(loc6)

    confs = layers.concatenate([conf1, conf2, conf3, conf4, conf5, conf6], axis=1)
    locs = layers.concatenate([loc1, loc2, loc3, loc4, loc5, loc6], axis=1)
    model = tf.keras.Model(inputs=mobilenet.layers[0].output, outputs=[confs, locs])

    return model

if __name__ == '__main__':
    num_classes = 21
    #model = get_VGG16_SSD(image_size=(512, 512, 3), num_classes=num_classes)
    #model = get_VGG16_SSD_300(image_size=(300, 300, 3), num_classes=num_classes)
    model = get_mobilenet_SSD(image_size=(300, 300, 3), num_classes=num_classes)
    model.summary()

    image = np.random.rand(1, 300, 300, 3)
    confs, locs = model.predict(image)
    print('confs shape =', np.shape(confs))
    print('locs shape =', np.shape(locs))