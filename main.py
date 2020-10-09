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

class ChunkManager(object):
    def __init__(self, filename):
        self.fps = 30
        self.ws_dir = get_workspace_dir(filename)

    def get_frame(self, sec):
        n_frame = int((sec * self.fps)) + 1
        number4 = "{:0=4}".format(n_frame)
        image_name = self.ws_dir + number4 + ".png"
        img = Image.open(image_name)
        return np.array(img)

def plot_sequence(img_seq, tile_shape=None):
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

    N = len(img_seq)
    t_lst_ = ["0.0", "1.0", "2.5", "5.0"]
    t_lst = t_lst_ + t_lst_
    for i in range(N):
        ax, img = axes[i], img_seq[i]
        ax.imshow(img)
        ax.axis("off")

        """
        pre = "(a)" if i < N/2 else "(b)"
        label = pre + " t = " +  t_lst[i] + " s"
        ax.text(0.05, 0.03, label, 
                transform=ax.transAxes, 
                fontsize=9)
        """
    plt.imshow(img)
    #plt.show()
    plt.savefig("tmp.pdf", format="pdf", dpi=300)

if __name__=='__main__':
    filename = "./icra_oven.mp4"
    try:
        prepare(filename)
    except:
        pass
    cm = ChunkManager(filename)
    frame = cm.get_frame(0)
    frames = [frame]*6
    plot_sequence(frames, (6, 1))

