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
import glob
import pathlib
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_WIDTH = 128 #origin:178
IMG_HEIGHT = 128 #origin:218
sample_dir = './samples/'
data_root = pathlib.Path(sample_dir)
all_image_paths = list(data_root.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)


fig = plt.figure(figsize=(4, 4))

for n, image in enumerate(image_ds.take(16)):
  plt.subplot(4,4,n+1)
  plt.imshow(image)
  plt.axis('off')
# plt.show()
plt.savefig('ground_true_image')


