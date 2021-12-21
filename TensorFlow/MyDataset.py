import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.cond(
        # 判断是否为 jpg 格式图片。如果是，就使用 decode_jpeg，不是则使用 decode_png
        # 针对 png 格式的图片，请使用 decode_png
        tf.image.is_jpeg(image_string),
        lambda: tf.image.decode_jpeg(image_string),  # 注意修改 channel!
        lambda: tf.image.decode_png(image_string))
    image_resized = tf.image.resize(image_decoded, [64, 64]) / 255.0
    return image_resized, label


def get_dataset(filename="data\\train\\label.txt", image_dir="data\\train", batch_size=4, mode="train"):
    inages_path_list = []
    landmarks_frame = pd.read_csv(filename, header=None)  # txt文件
    image_names = landmarks_frame.iloc[:, 0]
    for image_name in image_names:
        # print(image_name)
        image_path = image_dir + "\\" + image_name
        # print(image_path)
        inages_path_list.append(image_path)
        # break

    landmarks = landmarks_frame.iloc[:, 1:]
    # print(len(landmarks))
    landmarks = np.array(landmarks)
    landmarks = landmarks.astype('float32').reshape(-1, 5, 2)
    landmarks[..., 0] /= 94  # 注意修改 size !
    landmarks[..., 1] /= 60
    landmarks = tf.constant(landmarks)
    # print(landmarks)
    dataset = tf.data.Dataset.from_tensor_slices((inages_path_list, landmarks))
    # for name, label in train_dataset:
    #     print(name, label)
    #     break
    if mode == "train":
        dataset = dataset.shuffle(buffer_size=30).map(decode_and_resize).batch(batch_size)
    else:
        dataset = dataset.map(decode_and_resize).batch(batch_size)  # test 就不打乱顺序
    return dataset

def main():
    train_dataset = get_dataset()
    for images, labels in train_dataset:
        """ 
        展示一个 batch(4) 中的图片
        """
        fig, axs = plt.subplots(1, 4)
        for i in range(4):
            axs[i].set_title(labels.numpy()[i])
            axs[i].imshow(images.numpy()[i, :, :, 0])
        plt.show()
        break


if __name__ == "__main__":
    main()



















