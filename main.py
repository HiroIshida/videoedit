import os
import math
import subprocess
import numpy as np
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


class ChunkManager(object):
    def __init__(self, filename):
        self.fps = 30
        self.ws_dir = get_workspace_dir(filename)

    def get_frame(self, sec):
        n_frame = int((sec * self.fps)) + 1
        number4 = "{:0=4}".format(n_frame)
        image_name = self.ws_dir + number4 + ".png"
        img = Image.open(image_name)
        return np.asarray(img)

if __name__=='__main__':
    filename = "./icra_oven.mp4"
    try:
        prepare(filename)
    except:
        pass
    cm = ChunkManager(filename)
    frame = cm.get_frame(0)
    import matplotlib.pyplot as plt
    plt.imshow(frame)
    plt.show()

