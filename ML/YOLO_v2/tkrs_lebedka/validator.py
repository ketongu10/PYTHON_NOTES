import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkrs.tkrs_lebedka.methods_rect import PolsunokSides, PolsunokMass, TrosAndRails, Method
import io
from PIL import Image

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

class Validator:



    AHTUNG_SPEED = 15 #cm/sec
    SPEED_TRESHOLD_UP = 300 # cm/s - max speed

    def __init__(self, size=(1080, 1920)):
        self.size = size
        self.by_sides = PolsunokSides(size, self)
        self.by_mass = PolsunokMass(size, self)
        self.by_tros = TrosAndRails(size, self)
        self.now_speeds = []
        self.t = 0
        self.ts = []
        self.rail_size = None
        self.rail_sizes = []

        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        self.axs[0].set_ylabel("position")
        self.axs[1].set_ylabel("speed, cm/s")
        self.axs[1].set_xlabel("time, s")
        self.axs[1].set_ylim([-100, 100])
        self.has_legend = False



        #self.fig.patch.set_facecolor('black')




    def finalize(self):
        del self.fig
        del self.axs

    def update(self, labels, t):
        self.t = t
        self.ts.append(t)
        Method.update_rail_size(labels, self)
        self.by_sides.update(labels, t).calc_prec_speed()
        self.by_mass.update(labels, t).calc_prec_speed()
        self.by_tros.update(labels, t).calc_prec_speed()




    def render(self, image):

        image_ = self.by_sides.render(image)
        image_ = self.by_mass.render(image_)
        image_ = self.by_tros.render(image_)
        return image_

    def render_graphics(self):
        l1 = self.by_sides.render_graphics(self.axs, self.t, color='r')
        l2 = self.by_mass.render_graphics(self.axs, self.t, color='b')
        l3 = self.by_tros.render_graphics(self.axs, self.t, color='g')
        l4 = Method.render_ahtung_lines(self, self.t)

        good_speeds = []
        for sp in self.now_speeds:
            if np.abs(sp) < Validator.SPEED_TRESHOLD_UP:
                good_speeds.append(sp)

        self.axs[1].set_title(f"{np.mean(good_speeds):.01f} | by {len(good_speeds)} good method")
        self.now_speeds.clear()

        lbls = []
        ls = []
        if l1:
            ls.append(l1)
            lbls.append("by_sides")
        if l2:
            ls.append(l2)
            lbls.append("by_mass")
        if l3:
            ls.append(l3)
            lbls.append("by_tros")
        if l4:
            ls.append(l4)
            lbls.append("ahtung")
        if not self.has_legend and len(ls) == 4:
            self.fig.legend(ls, labels=lbls, loc="upper right")
            self.has_legend = True
        img = np.array(fig2img(self.fig))[:, :, :3]
        img = cv2.resize(img, (min(self.size), min(self.size)))
        return img

