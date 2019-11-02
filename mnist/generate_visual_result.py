import os
import imageio
import glob
gif_name = 'dcgan.gif'
final_name = 'final.png'

def creat_gif(result_path, gif_name, duration=0.2):
    filenames = glob.glob(os.path.join(result_path,'image*.png'))
    filenames = sorted(filenames)
    # generate final image
    final_image = filenames[-1]
    final_image = imageio.imread(final_image)
    imageio.imwrite(final_name,final_image)
    # generate gif image
    frames = []

    for image_name in filenames:
        frames.append(imageio.imread(image_name))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

    return

creat_gif('./result', gif_name)


#generate ground true image
import tensorflow as tf
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(4, 4))
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(train_images[i, :, :] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')
plt.savefig('ground_true_image')


