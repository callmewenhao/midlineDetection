import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers.advanced_activations import ReLU

class Identity(keras.Model):
    def __init__(self):
        super().__init__()
    
    @tf.function
    def call(self, x):
        return x

class block(keras.Model):
    def __init__(self, in_dim, out_dim, stride=2):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(
            out_dim,
            kernel_size=3,
            strides=stride,
            padding='same'
        )
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        self.conv2 = keras.layers.Conv2D(
            out_dim,
            kernel_size=3,
            strides=1,
            padding='same'
        )
        self.bn2 = keras.layers.BatchNormalization()
        # 使用 1*1 卷积核进行下采样
        if stride == 2 and in_dim != out_dim:  # 只有 stride = 2 同时 in_dim != out_dim 时
            self.downsampling = keras.Sequential([
                keras.layers.Conv2D(out_dim, 1, strides=stride, padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU()
            ])
        else:
            self.downsampling = Identity()
    @tf.function
    def call(self, input):
        h = input
        x = self.conv1(input)
        x = self.bn1(x, training=True)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=True)
        x = self.relu(x)
        identity = self.downsampling(h)
        output = x + identity
        return output


class ResNet18(keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        # head of model
        self.conv1 = keras.layers.Conv2D(64, 7, strides=2, padding='same')
        self.MaxPool = keras.layers.MaxPool2D(3, 2, padding='same')
        # body of model
        self.blocks1 = self.blocks(64, 64, 1, 2)
        self.blocks2 = self.blocks(64, 128, 2, 2)
        self.blocks3 = self.blocks(128, 256, 2, 2)
        self.blocks4 = self.blocks(256, 512, 2, 2)
        # classifier of model
        self.avg = keras.layers.AveragePooling2D(1)
        self.flatten = keras.layers.Flatten()
        self.fc = keras.layers.Dense(num_classes)

    def blocks(self, in_dim, out_dim, stride, num_blocks=2):
        blocks_list = []
        blocks_list.append(block(in_dim, out_dim, stride))
        for i in range(num_blocks-1):
            blocks_list.append(block(out_dim, out_dim, 1))
        return keras.Sequential(blocks_list)

    @tf.function
    def call(self, input):
        x = self.conv1(input)
        x = self.MaxPool(x)
        # print(x.shape)

        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        # print(x.shape)

        x = self.avg(x)
        x = self.flatten(x)
        output = self.fc(x)
        return output

# 使用示例
def main():
    x = np.random.randn(4, 64, 64, 1)
    x = tf.constant(x)
    print(x.shape)
    model = ResNet18(num_classes=10)
    print(model)
    out = model(x)
    print(out.shape)


if __name__ == "__main__":
    main()