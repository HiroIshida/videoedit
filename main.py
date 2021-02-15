import os
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

def get_workspace_dir(filename):
    rawname, ext = os.path.splitext(
            os.path.basename(filename))
    assert ext==".mp4", "only accept mp4 type data"
    home = os.path.expanduser("~")
    workspace_dir = "{0}/.videoedit/{1}/".format(home, rawname)
    return workspace_dir

def prepare(filename):
    workspace_dir = get_workspace_dir(filename)
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
        cmd="ffmpeg -i {0} {1}%04d.png".format(
                filename, workspace_dir)
        subprocess.call(cmd, shell=True)
    else:
        raise Exception("the same dir name already exists")

def crop_image(np_image, v_crop=None, h_crop=None):
    shape = np_image.shape[0:2]
    slice_spec_lst = []
    for crop, size in zip([v_crop, h_crop], shape):
        slice_spec = (0, -1) if crop is None \
                else (int(round(size * crop[0])), int(round(size*crop[1])))
        slice_spec_lst.append(slice_spec)

    vs, ve = slice_spec_lst[0]
    hs, he = slice_spec_lst[1]
    return np_image[vs:ve, hs:he]

def transposed_indices(tile_shape):
    n, m = tile_shape
    for i in range(n):
        for j in range(m):
            yield n * j + i

class ChunkManager(object):
    def __init__(self, filename):
        self.fps = 30
        self.ws_dir = get_workspace_dir(filename)

    def get_frame(self, sec):
        n_frame = int((sec * self.fps)) + 1
        print(n_frame)
        number4 = "{:0=4}".format(n_frame)
        image_name = self.ws_dir + number4 + ".png"
        img = Image.open(image_name)
        return np.array(img)

    def plot_time_sequence(self, time_seq, base_time=0, tile_shape=None, v_crop=None, h_crop=None, transpose=False, filename="hogehoge", dpi=300):
        if tile_shape is not None:
            assert len(time_seq) == np.prod(tile_shape)
        else:
            tile_shape = (len(time_seq), 1)

        print(list(transposed_indices(tile_shape)))
        if transpose:
            time_seq = np.array(time_seq)[list(transposed_indices(tile_shape))]

        img_seq = [
                crop_image(self.get_frame(t+base_time), v_crop, h_crop)
                for t in time_seq]
        if tile_shape is None:
            tile_shape = (1, len(img_seq))

        height, width, _ = img_seq[0].shape
        image_ratio = height/(width*1.0)
        tile_ratio = tile_shape[0]/(tile_shape[1]*1.0)

        fig = plt.figure(figsize=tile_shape)
        fig.set_size_inches(6, 6*image_ratio*tile_ratio)
        fig.tight_layout()
        gspec = gridspec.GridSpec(*tile_shape)
        gspec.update(wspace=0.02, hspace=0.02)
        axes = [plt.subplot(gspec[i]) for i in range(len(img_seq))]

        for ax, img in zip(axes, img_seq):
            ax.imshow(img)
            ax.axis("off")
        plt.imshow(img)
        #plt.show()
        plt.savefig(filename, format="png", dpi=dpi)
        plt.show()

if __name__=='__main__':

    # oven 1
    oven_before_detail = False
    if oven_before_detail:
        n = 6
        time_seq_base = np.array([0 + 1.0 * i for i in range(n)])
        filename = "/home/hiro/catkin_ws/src/oven/detail.mp4"
        try:
            prepare(filename)
        except:
            pass
        cm = ChunkManager(filename)
        #base_time = 1.0
        #time_seq = time_seq_base + base_time
        #cm.plot_time_sequence(time_seq, tile_shape=(1, n), filename="before_30_detail.png", v_crop=None)

    oven_after_1 = False
    if oven_after_1:
        n = 6
        time_seq_base = np.array([0 + 0.5 * i for i in range(n)])
        filename = "/home/hiro/catkin_ws/src/oven/after_learning.mp4"
        try:
            prepare(filename)
        except:
            pass
        cm = ChunkManager(filename)
        base_time = 60.0 * 4.0 + 27.5
        time_seq = time_seq_base + base_time
        cm.plot_time_sequence(time_seq, tile_shape=(1, n), filename="after_05.png", v_crop=None)

    oven_before = False
    if oven_before:
        n = 6
        time_seq_base = np.array([0 + 0.5 * i for i in range(n)])
        filename = "/home/hiro/catkin_ws/src/oven/before_learning.mp4"
        try:
            prepare(filename)
        except:
            pass
        cm = ChunkManager(filename)
        base_time = 60 * 3.0 + 50.0
        time_seq = time_seq_base + base_time
        cm.plot_time_sequence(time_seq, tile_shape=(1, n), filename="before_30.png", v_crop=None)

    def create_seq_com1():
        filename = "/home/hiro/catkin_ws/src/magicgrasp/video/demo_com1.mp4"
        try:
            prepare(filename)
        except:
            pass
        cm = ChunkManager(filename)

        time_seq_base = np.array([0 + 0.8 * i for i in range(7)])

        h_crop = [0.0, 0.9]
        v_crop = [0.05, 0.8]

        base_time = 13.0
        time_seq = time_seq_base + base_time
        cm.plot_time_sequence(time_seq, tile_shape=(1, 7), filename="com1_after_seq.png", h_crop=h_crop, v_crop=v_crop)

        base_time = 35.0
        time_seq = time_seq_base + base_time
        cm.plot_time_sequence(time_seq, tile_shape=(1, 7), filename="com1_before_seq.png", h_crop=h_crop, v_crop=v_crop)

    def create_seq_com2():
        filename = "/home/hiro/catkin_ws/src/magicgrasp/video/demo_com2.mp4"
        try:
            prepare(filename)
        except:
            pass
        cm = ChunkManager(filename)

        h_crop = [0.0, 0.9]
        v_crop = [0.05, 0.8]
        time_seq_base = np.array([0 + 0.8 * i for i in range(7)])
        base_time = 7.0
        time_seq = time_seq_base + base_time
        #cm.plot_time_sequence(time_seq, tile_shape=(1, 7), filename="com2_after_seq.png", v_crop=[0.2, 0.8])
        cm.plot_time_sequence(time_seq, tile_shape=(1, 7), filename="com2_after_seq.png", h_crop=h_crop, v_crop=v_crop)

        base_time = 29.5
        time_seq = time_seq_base + base_time
        cm.plot_time_sequence(time_seq, tile_shape=(1, 7), filename="com2_before_seq.png", h_crop=h_crop, v_crop=v_crop)

    create_seq_com1()
    create_seq_com2()


    #cm.plot_time_sequence([0.0, 1.0, 2.0, 3.0, 4.0], base_time=42.0) # oven after -0.0
    #cm.plot_time_sequence([0.0, 1.0, 2.0, 3.0, 4.0], base_time=56.0) # oven after -0.04
    #cm.plot_time_sequence([0.0, 1.0, 2.0, 3.0, 4.0], base_time=69.0) # oven after 0.06

    #cm.plot_time_sequence([0.0, 1.0, 2.0, 3.0, 4.0], base_time=3.0) # oven before 0.0
    #cm.plot_time_sequence([0.0, 1.0, 2.0, 3.0, 4.0], base_time=16.0) # oven before -0.04
    #cm.plot_time_sequence([0.0, 1.0, 2.0, 3.0, 4.0], base_time=29.0) # oven before 0.06



