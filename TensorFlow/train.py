import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from model.resnet18 import ResNet18
from MyDataset import get_dataset
from utils import plot_loss, get_logger


def main():

    epochs = 25
    batch_size = 4
    learning_rate = 5e-4

    train_dataset = get_dataset(
        filename="data\\train\\label.txt", 
        image_dir="data\\train",
        batch_size=batch_size,
        mode="train"
    )

    model = ResNet18(10)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    losses = {'train':[], 'test':[]}
    # log 开始
    logger = get_logger('train.log')
    logger.info('start training!')

    for epoch in range(epochs):
        # train one epoch
        for (i, data) in enumerate(train_dataset):
            images = data[0]
            labels = data[1]
            with tf.GradientTape() as tape:
                y_pead = model(images)
                y_pead = tf.reshape(y_pead, [-1, 5, 2])
                loss = keras.losses.MSE(y_true=labels, y_pred=y_pead)
                loss = tf.reduce_mean(loss)
                if i % 2 == 0:
                    print(f"epoch:{epoch}/{epochs}, {i+1}th, loss:{loss}")
                    losses["train"].append(loss)
                    
            grads = tape.gradient(loss, model.trainable_variables)  # 因为用了 BN， 所以要用.trainable_variables()否则报 warning
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    # Save the weights
    model.save_weights(".\\outputs\\my_checkpoint")
    # 打印 loss
    plot_loss(losses, save_path=f".\\outputs\\losses_{epochs}.png")
    # log 结束
    logger.info('finish training!')


if __name__ == "__main__":
    main()















