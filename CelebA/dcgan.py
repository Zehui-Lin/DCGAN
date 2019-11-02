###Import TensorFlow and other libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import time
import os
import matplotlib.pyplot as plt
import PIL
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE


###Load and prepare the dataset
IMG_WIDTH = 128 #origin:178
IMG_HEIGHT = 128 #origin:218

data_dir = './celeba/img_align_celeba/img_align_celeba/'
data_dir = pathlib.Path(data_dir)
list_ds = tf.data.Dataset.list_files(str(data_dir/'*'))

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = (img - 0.5) / 0.5 # Normalize the images to [-1, 1]
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH]) # [IMG_HEIGHT, IMG_WIDTH]

def process_path(file_path):
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
decoded_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

BUFFER_SIZE = 202599
BATCH_SIZE = 256
default_timeit_steps = BUFFER_SIZE//BATCH_SIZE

# Shuffled and batched the dataset
def prepare_for_training(ds, cache=True, shuffle_buff_size=BUFFER_SIZE):
    #use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    #ds = ds.shuffle(buffer_size=shuffle_buff_size) #not necessary

    #Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

train_ds = prepare_for_training(decoded_ds)


###Create the models

#The Generator
#Input a seed (random noise). Start with a Dense layer, then upsample by ConvTranspose.
#input shape = batch_size * 500
#output shape = 128 * 128 * 3
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16*16*256, use_bias=False, input_shape=(500, )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 256))) # (bs, 16, 16, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)) # (bs, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)) # (bs, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)) # (bs, 64, 64, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')) # (bs, 128, 128, 3)

    return model

#The Discriminator
#A CNN-based image classifier
#input shape = batch_size * 128 * 128 * 3
#output shape = batch_size * 1
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 3])) # (bs, 64, 64, 64)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')) # (bs, 32, 32, 128)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')) # (bs, 16, 16, 256)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten()) # (bs, 16*16*256)
    model.add(layers.Dense(1))

    return model

###Define the loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


#Discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

#Generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

#Two optimizers because of two networks separately
generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

###Checkpoints saving
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,"ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

###Define the training loop
EPOCHS = 30
noise_dim = 500
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(image):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(noise, training=True)

        real_output = discriminator(image, training=True)
        fake_output = discriminator(generated_image, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        it = iter(dataset)
        for i in range(default_timeit_steps):
            batch = next(it)
            train_step(batch)

        # Produce images for the GIF as we go
        generate_and_save_images(generator, epoch + 1, seed)

        #Save the model when finish (the model is big)
        if (epoch + 1) % 30 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)

###Visualize the results
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :] * 127.5 + 127.5) #cmap shouble be ignored
        plt.axis('off')

    if not os.path.exists('./result'):
        os.mkdir('./result')
    plt.savefig('./result/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()



#Train the model
if __name__ == '__main__':
    train(train_ds, EPOCHS)

    # print(__name__)

