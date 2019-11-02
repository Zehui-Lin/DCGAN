import os
import imageio
import glob
gif_name = 'dcgan.gif'

def creat_gif(result_path, gif_name, duration=0.2):
    filenames = glob.glob(os.path.join(result_path,'image*.png'))
    filenames = sorted(filenames)
    frames = []

    for image_name in filenames:
        frames.append(imageio.imread(image_name))

    # 保存为gif格式的图
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

    return

creat_gif('./result', gif_name)


