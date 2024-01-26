import os
import uuid

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


def make_gen():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16 * 16 * 1024, input_shape=(256,)))
    model.add(layers.ReLU())

    model.add(layers.Reshape((16, 16, 1024)))

    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh'))

    return model


gen = make_gen()

ckpt_dir = './training_checkpoints'
ckpt_prefix = os.path.join(ckpt_dir, 'ckpt')
ckpt = tf.train.Checkpoint(step=tf.Variable(1), generator=gen,)
ckpt = ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()

num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, 256])


def generate_and_save_images(_model, _seed):
    predictions = _model(_seed, training=False)

    fig = plt.figure(figsize=(16, 16))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = (predictions[i] + 1) / 2.0
        plt.imshow(img)
        plt.axis('off')

    rand_id = uuid.uuid4()
    rand_id = str(rand_id)[:8]

    plt.savefig(f'cats_{rand_id}.png')
    plt.close(fig)


generate_and_save_images(gen, seed)
