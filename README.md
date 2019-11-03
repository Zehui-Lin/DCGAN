# DCGAN


## The implementation of Deep Convolutional Generative Adversarial Network (DCGAN)

This is a basic implementation of [Deep Convolutional Generative Adversarial Network (DCGAN)](https://arxiv.org/pdf/1511.06434.pdf). The code mainly refers to the [TensorFlow Tutorials](https://tensorflow.google.cn/tutorials/generative/dcgan) in [`tensorflow/docs`](https://github.com/tensorflow/docs). Two datasets mnist and Large-scale Celeb Faces Attributes ([CelebA](https://www.kaggle.com/jessicali9530/celeba-dataset)) were used.

## Requirements

* python3
* tensorflow-gpu == 2.0.0

The CelebA dataset can be downloaded in kaggle by the above link.

## Getting start

* Download the dataset CelebA in kaggle
* Modify the dataset path in `dcgan.py`
* Run the script `dcgan.py` (ensure you have at least one GPU device) and `image_at_epoch_xxxx.png` will be generated in `result` folder
* Run the script `generate_visual_result.py` and `ground_true_image.png`, `dcgan.gif`, `final.png` will be generated.
* Enjoy it!

When apply to the mnist dataset, the first and second step can be ignored.

## There are some differences in the newotk architecture that implements these two data sets

The detail can be seem in the code.

* Generator

  * mnist

    ```python
    #The Generator
    #Input a seed (random noise). Start with a Dense layer, then upsample by ConvTranspose.
    #input shape = batch_size * 100
    #output shape = 28 * 28 * 1
    def make_generator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100, )))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256))) # (bs, 7, 7, 256)

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)) # (bs, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)) # (bs, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')) # (bs, 28, 28, 1)

        return model

    ```

  * CelebA

    ```python
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

    ```

* Discriminator

  * mnist

    ```python
    #The Discriminator
    #A CNN-based image classifier
    #input shape = batch_size * 28 * 28 * 1
    #output shape = batch_size * 1
    def make_discriminator_model():
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1])) # (bs, 14, 14, 64)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')) # (bs, 7, 7, 128)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten()) # (bs, 7*7*128)
        model.add(layers.Dense(1))

        return model

    ```

  * CelebA

    ```python
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

    ```


## Visual result

### mnist
#### ground true
![avatar](https://github.com/Zehui-Lin/DCGAN/blob/master/mnist/ground_true_image.png)

#### generating process
![avatar](https://github.com/Zehui-Lin/DCGAN/blob/master/mnist/dcgan.gif)

#### final generated result
![avatar](https://github.com/Zehui-Lin/DCGAN/blob/master/mnist/final.png)


### CelebA
#### ground true
![avatar](https://github.com/Zehui-Lin/DCGAN/blob/master/CelebA/ground_true_image.png)

#### generating process
![avatar](https://github.com/Zehui-Lin/DCGAN/blob/master/CelebA/dcgan.gif)

#### final generated result
![avatar](https://github.com/Zehui-Lin/DCGAN/blob/master/CelebA/final.png)



## LICENSE

The code is under Apache-2.0 License.
