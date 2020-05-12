import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import Model

def main():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train/255.0 , x_test/255.0
    #对训练集的图片进行初步处理

    class ConvBNRelu(Model):
        def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
            super(ConvBNRelu, self).__init__()
            self.model = tf.keras.models.Sequential([
                Conv2D(ch, kernelsz, strides=strides, padding=padding),
                BatchNormalization(),
                Activation('relu')
            ])

        def call(self, x):
            x = self.model(x,training=False)  # 在training=False时，BN通过整个训练集计算均值、方差去做批归一化，training=True时，通过当前batch的均值、方差去做批归一化。推理时 training=False效果好
            return x

    class InceptionBlk(Model):
        def __init__(self, ch, strides=1):
            super(InceptionBlk, self).__init__()
            self.ch = ch
            self.strides = strides
            self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
            self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
            self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
            self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
            self.c3_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
            self.c3_3 = ConvBNRelu(ch, kernelsz=3, strides=1)
            self.p4_1 = MaxPool2D(3, strides=1, padding='same')
            self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides)

        def call(self, x):
            x1 = self.c1(x)
            x2_1 = self.c2_1(x)
            x2_2 = self.c2_2(x2_1)
            x3_1 = self.c3_1(x)
            x3_2 = self.c3_2(x3_1)
            x3_3 = self.c3_3(x3_2)
            x4_1 = self.p4_1(x)
            x4_2 = self.c4_2(x4_1)
            # concat along axis=channel
            x = tf.concat([x1, x2_2, x3_3, x4_2], axis=3)
            return x

    class Inception10(Model):
        def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
            super(Inception10, self).__init__(**kwargs)
            self.in_channels = init_ch
            self.out_channels = init_ch
            self.num_blocks = num_blocks
            self.init_ch = init_ch
            self.c1 = ConvBNRelu(init_ch)
            self.blocks = tf.keras.models.Sequential()
            for block_id in range(num_blocks):
                for layer_id in range(2):
                    if layer_id == 0:
                        block = InceptionBlk(self.out_channels, strides=2)
                    else:
                        block = InceptionBlk(self.out_channels, strides=1)
                    self.blocks.add(block)
                # enlarger out_channels per block
                self.out_channels *= 2
            self.p1 = GlobalAveragePooling2D()
            self.f1 = Dense(num_classes, activation='softmax')

        def call(self, x):
            x = self.c1(x)
            x = self.blocks(x)
            x = self.p1(x)
            y = self.f1(x)
            return y
    model = Inception10(num_blocks=2, num_classes=10)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    # 训练
    # 断点续训
    model_save_path = "./checkpoint/cifar10.ckpt"
    if os.path.exists(model_save_path + '.index'):
        print('_____________________model load____________________')
        model.load_weights(model_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
         filepath=model_save_path,
         save_best_only=True,
         save_weights_only=True
    )

    history = model.fit(x_train, y_train, batch_size=32, epochs=5,
            validation_data=(x_test, y_test),
            validation_freq=1,
            callbacks=[cp_callback])
    model.summary()
    # 保存权重
    file = open('./weight.txt', 'w')
    for v in model.trainable_variables:
        file.write(str(v.name)+'\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')

    # 数据可视化
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='training accuracy')
    plt.plot(val_acc, label='val acc')
    plt.title('train and val acc')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='training loss')
    plt.plot(val_loss, label='val loss')
    plt.title('train and val loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()