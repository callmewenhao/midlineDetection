import tensorflow as tf
import tensorflow.keras as keras
from model.resnet18 import ResNet18
from MyDataset import get_dataset
from utils import show_landmarks
import matplotlib.pyplot as plt
from skimage import io


def main():
    img_path = "data\\test\\"
    save_path = "outputs\\test_outputs\\"
    test_dataset = get_dataset(filename="data\\test\\label.txt", image_dir="data\\test", batch_size=1, mode="test")
    model = ResNet18()
    model.load_weights('outputs\\my_checkpoint')
    mse = keras.metrics.MeanSquaredError()
    coordinates = []
    for (i, data) in enumerate(test_dataset):
            images = data[0]
            labels = data[1]
            y_pred = model.call(images)  # 有无 .call() 都可以
            y_pred = tf.reshape(y_pred, [-1, 5, 2])
            mse.update_state(y_true=labels, y_pred=y_pred)
            y_pred = y_pred.numpy()
            y_pred[..., 0] *= 94
            y_pred[..., 1] *= 60
            print(y_pred)
            coordinates.append(y_pred)
    print("test MSE loss: %f" % mse.result())  # 0.144016
    
    for i in range(len(test_dataset)):
        # print(img_path + f"{i:0>3}.png")
        image = io.imread(img_path + f"{i:0>3}.png")
        plt.figure("test")
        show_landmarks(image, coordinates[i])
        plt.savefig(save_path+f'pred_outputs_{i:0>3}.png')
        print("saved " + f'pred_outputs_{i:0>3}.png')
        plt.close()

if __name__ == "__main__":
    main()


