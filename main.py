""" Based mainly on arXiv:1511.06434. """
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython import display
from PIL import Image
from tensorflow.keras import layers

# Pictures' size
imgs = 256

directory = 'datasets/cats'

# Hyperparameters
n = 256  # Latent space size
batch_size = 4
epochs = 10000
gen_lr = 1e-3
disc_lr = 1e-6
beta1 = 0.5


# Due to lack of memory this model uses hard drive for the training
# which significantly increases the training time.
def image_generator():
    """ Generate resized and normalised images from the dataset. """
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(directory, filename))
            img = img.resize((imgs, imgs))
            img = np.array(img).astype('float32')
            img = (img - 127.5) / 127.5
            yield img


output_signature = tf.TensorSpec(shape=(imgs, imgs, 3), dtype=tf.float32)
train_dataset = tf.data.Dataset.from_generator(image_generator, output_signature=output_signature)

train_dataset = train_dataset.batch(batch_size)


def make_gen():
    """ Make the generator model.

    The structure of the generator model is very similar to the
    original DCGAN with slight modifications.

    The original DCGAN was generating 64x64 pictures from latent space of size 100,
    this model generates 256x256 pictures from latent space of size 256.
    Another difference is kernel size - 4 instead of 5.
    To generate images without blurs, the convolutional representation with
    feature maps was made bigger - 16x16x1024 instead of 4x4x1024.
    """
    model = tf.keras.Sequential()
    model.add(layers.Dense(16 * 16 * 1024, input_shape=(n,)))
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


def make_disc():
    """ Make the discriminator model.

    In the paper the discriminator should have about
    the same amount of layers as the generator.
    """
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', input_shape=(imgs, imgs, 3)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(1024, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Dense(1, activation='sigmoid'))

    return model


# The loss function is going to be binary cross entropy.
bincent = tf.keras.losses.BinaryCrossentropy()


def gen_loss(_fake):
    """ Get generator's loss. """
    return bincent(tf.ones_like(_fake), _fake)


def disc_loss(_real, _fake):
    """ Get discriminator's loss. """
    real_loss = bincent(tf.ones_like(_real), _real)
    fake_loss = bincent(tf.zeros_like(_fake), _fake)
    total_loss = real_loss + fake_loss
    return total_loss


# Create models.
gen = make_gen()
disc = make_disc()

# Optimiser used is adam with set learning rate and beta1.
g_optimiser = tf.keras.optimizers.Adam(learning_rate=gen_lr, beta_1=beta1)
d_optimiser = tf.keras.optimizers.Adam(learning_rate=disc_lr, beta_1=beta1)

# Checkpoint handling.
ckpt_dir = './training_checkpoints'
ckpt_prefix = os.path.join(ckpt_dir, 'ckpt')
ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                           generator_optimizer=g_optimiser,
                           discriminator_optimizer=d_optimiser,
                           generator=gen,
                           discriminator=disc)
ckpt.restore(tf.train.latest_checkpoint(ckpt_dir))

num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, n])


@tf.function
def train_step(_images):
    noise = tf.random.normal([batch_size, n])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(noise, training=True)

        real_output = disc(_images, training=True)
        fake_output = disc(generated_images, training=True)

        _gen_loss = gen_loss(fake_output)
        _disc_loss = disc_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(_gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(_disc_loss, disc.trainable_variables)

    g_optimiser.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    d_optimiser.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))

    # print('step - Generator Loss: {}, Discriminator Loss: {}'.format(_gen_loss, _disc_loss))

    return _gen_loss, _disc_loss


def train(_dataset, _epochs):
    for epoch in range(_epochs):
        start = time.time()
        gen_losses = []
        disc_losses = []

        for image_batch in _dataset:
            _gen_loss, _disc_loss = train_step(image_batch)
            gen_losses.append(_gen_loss)
            disc_losses.append(_disc_loss)

        avg_gen_loss = tf.reduce_mean(gen_losses)
        avg_disc_loss = tf.reduce_mean(disc_losses)

        print('Time for epoch {} is {} sec'.format(ckpt.step + 0, time.time() - start))
        print('avg - Generator Loss: {}, Discriminator Loss: {}'.format(avg_gen_loss, avg_disc_loss))
        print('----------------------------------------------------------------------------')

        if ckpt.step % 1 == 0:
            ckpt.save(file_prefix=ckpt_prefix)
            display.clear_output(wait=True)
            generate_and_save_images(gen, seed)

        ckpt.step.assign_add(1)


def generate_and_save_images(_model, _seed):
    predictions = _model(_seed, training=False)

    fig = plt.figure(figsize=(16, 16))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        img = (predictions[i] + 1) / 2.0
        plt.imshow(img)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(ckpt.step + 0))
    plt.close(fig)


train(train_dataset, epochs)
