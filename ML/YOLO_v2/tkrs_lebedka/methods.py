import cv2
import numpy as np

import matplotlib.pyplot as plt

MASK_SIZE = 640

R_Matrix = np.zeros(shape=(MASK_SIZE, MASK_SIZE, 2), dtype=int)

for x in range(MASK_SIZE):
    for y in range(MASK_SIZE):
        R_Matrix[x, y] = [y, x]


CONFS = {
    0: 0.5,
    1: 0.5,
    2: 0.5,
    3: 0.5,
    4: 0.5,
}
STR2CLS = {"box": 0, "tros": 1, "lebedka_polsunok": 2, "rails": 3, "width": 4}
TIMEOUT = 10000

def is_intersected(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    return  ((b2_x1 <= b1_x1 <= b2_x2 or b2_x1 <= b1_x2 <= b2_x2 or b1_x1 <= b2_x2 <= b1_x2)
             and (b2_y1 <= b1_y1 <= b2_y2 or b2_y1 <= b1_y2 <= b2_y2 or b1_y1 <= b2_y2 <= b1_y2))

def fit_line(xs, ys, xywh):
    x, y, w, h = xywh
    if w >= h:
        return np.polyfit(xs, ys, 1)
    else:
        alpha, betta = np.polyfit(ys, xs, 1)

        return 1/alpha, -betta/alpha


def render_cros(image, pos, size=10, color=(255, 255, 255), thickness=3):

    image_ = cv2.line(image, (int(pos[0]-size/2), int(pos[1])), (int(pos[0]+size/2), int(pos[1])), color, thickness)
    image_ = cv2.line(image_, (int(pos[0]), int(pos[1]-size/2)), (int(pos[0]), int(pos[1]+size/2)), color, thickness)
    return image_


def render_line_in_bbox(bbox, k, b):
    x, y, w, h = bbox
    x1, x2, y1, y2 = x - w/2, x + w/2 , y-h/2, y+h/2

def render_line_around_pos(image, pos, k, b, width=500, color=(255, 255, 255), thickness=2):

    x1, x2 = pos[0]-width/2, pos[0]+width/2
    y1, y2 = k*x1+b, k*x2+b
    image_ = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return image_



class Method:
    nothing2cm = 1600/2.5
    LAST_N = 10

    def __init__(self, size, validator):
        self.validator = validator
        self.pos = []
        self.speeds = []
        self.size = size
        self.t = 0

    def update(self, labels, t):
        return self

    def calc_speed(self):
        return self

    def calc_prec_speed(self):
        if len(self.pos) <= 4:
            return self
        x_t = np.array(self.pos[-5:])
        v_x, _ = np.polyfit(x_t[:, -1], x_t[:, -3], 1)
        v_y, _ = np.polyfit(x_t[:, -1], x_t[:, -2], 1)

        if self.pos[-1][-1] == self.t:
            self.speeds.append((np.sqrt(np.square([v_x, v_y]).sum())*np.sign(v_x), self.t))
        else:
            self.speeds.append((0, self.t))

        return self

    def filter_speeds(self, speds):
        arr = np.array(speds[-Method.LAST_N:][:])
        m = arr.mean()

        valid_values = np.where((np.abs((arr-m)/m) < 1.0) & (arr < self.validator.SPEED_TRESHOLD_UP/Method.nothing2cm))
        if len(valid_values) < 5:
            return speds
        return arr[valid_values]

        # return speds



    def render(self, image):
        return image

    @staticmethod
    def render_ahtung_lines(validator, t):
        ax_x, ax_vx = validator.axs
        l = None
        if validator.t == t and len(validator.ts) > 1:
            l = ax_vx.plot(validator.ts[-2:], [validator.AHTUNG_SPEED]*2, c='c', linestyle='dashed')
            l = ax_vx.plot(validator.ts[-2:], [-validator.AHTUNG_SPEED]*2, c='c', linestyle='dashed')
        return l

    def render_graphics(self, axs, t, color='r'):
        ax_x, ax_vx = axs
        l = None
        if len(self.pos) > 1 and self.pos[-1][-1] == t:
            l = ax_x.plot(np.array(self.pos[-2:])[:, -1], np.array(self.pos[-2:])[:, -3], c=color)


        # if len(self.speeds) > 1 and self.speeds[-1][-1] == t and self.validator.rail_size:
        #     l = ax_vx.plot(np.array(self.speeds[-2:])[:, -1], np.array(self.speeds[-2:])[:, 0]/self.validator.rail_size*Method.nothing2cm, c=color)
        #     ax_vx.set_title(f"{(self.speeds[-1][0]/self.validator.rail_size*Method.nothing2cm):.01f}")
        LAST_N = 10
        if len(self.speeds) > Method.LAST_N and self.speeds[-1][-1] == t and self.validator.rail_size:
            old_v = self.filter_speeds(np.array(self.speeds[-LAST_N-1:-1])[:, 0]).mean()

            new_v = self.filter_speeds(np.array(self.speeds[-LAST_N:])[:, 0]).mean()

            # print(old_v, np.array(self.speeds[-LAST_N-1:-1])[:, 0])
            # print(new_v, np.array(self.speeds[-LAST_N:])[:, 0])
            self.validator.now_speeds.append(new_v/self.validator.rail_size*Method.nothing2cm)
            l = ax_vx.plot(np.array(self.speeds[-2:])[:, -1], np.array([old_v, new_v])/self.validator.rail_size*Method.nothing2cm, c=color)
        return l

    @staticmethod
    def update_rail_size(labels, validator):
        clss = labels.boxes.cls.cpu().numpy().astype(dtype=int)
        confs = labels.boxes.conf.cpu().numpy()
        xywh = labels.boxes.xywh.cpu().numpy()

        width_bboxs = []
        width_confs = []

        for i, cls in enumerate(clss):
            if cls == STR2CLS["width"] and confs[i] >= CONFS[cls]:
                width_bboxs.append(xywh[i])
                width_confs.append(confs[i])

        if width_bboxs:
            width_bbox = width_bboxs[np.argmax(width_confs)]
            x, y, w, h = width_bbox
            validator.rail_sizes.append(np.sqrt(h**2+w**2))
            if len(validator.rail_sizes) >= 5:
                validator.rail_size = np.mean(validator.rail_sizes[-5:])/validator.size[0]

class PolsunokSides(Method):

    def update(self, labels, t):

        clss = labels.boxes.cls.cpu().numpy().astype(dtype=int)
        confs = labels.boxes.conf.cpu().numpy()
        xywh = labels.boxes.xywh.cpu().numpy()

        polsunoks = []
        polsunok_confs = []

        leb_boxs = []

        for i, cls in enumerate(clss):
            if cls == STR2CLS["lebedka_polsunok"] and confs[i] >= CONFS[cls]:
                polsunoks.append(xywh[i])
                polsunok_confs.append(confs[i])
            if cls == STR2CLS["box"] and confs[i] >= CONFS[cls]:
                leb_boxs.append(xywh[i])

        if polsunoks and leb_boxs:
            polsunok_xywh = polsunoks[np.argmax(polsunok_confs)]

            left = polsunok_xywh[0] - polsunok_xywh[2] / 2
            right = polsunok_xywh[0] + polsunok_xywh[2] / 2
            up = polsunok_xywh[1] - polsunok_xywh[3] / 2
            down = polsunok_xywh[1] + polsunok_xywh[3] / 2

            #print("BBOX", polsunok_xywh[0] / self.size[0], polsunok_xywh[1] / self.size[0])
            self.pos.append(np.array([left, right, up, down, polsunok_xywh[0]/ self.size[0], polsunok_xywh[1]/ self.size[0], t]))
        self.t = t

        return self

    def calc_speed(self):
        if len(self.pos) <= 1:
            return

        dx = self.pos[-1] - self.pos[-2]
        #print(self.pos[-1], self.pos[-2], dx)
        if dx[0]*dx[1] > 0 and dx[2]*dx[3] > 0 and dx[-1] < TIMEOUT:
            #print(np.square(dx[4:-1]).sum()/dx[-1])
            self.speeds.append((np.sqrt(np.square(dx[4:-1]).sum())/dx[-1]*np.sign(dx[4]), self.t)) # center_bbox vx, vy
        else:
            self.speeds.append((0, self.t))

        return self


    def calc_prec_speed(self):
        if len(self.pos) <= 4:
            return self
        x_t = np.array(self.pos[-5:])
        v_x, _ = np.polyfit(x_t[:, -1], x_t[:, -3], 1)
        v_y, _ = np.polyfit(x_t[:, -1], x_t[:, -2], 1)
        if self.pos[-1][-1] == self.t:
            self.speeds.append((np.sqrt(np.square([v_x, v_y]).sum())*np.sign(v_x), self.t))
        else:
            self.speeds.append((0, self.t))

        # v_l, _ = np.polyfit(x_t[:, -1], x_t[:, 0], 1)
        # v_r, _ = np.polyfit(x_t[:, -1], x_t[:, 1], 1)
        # v_u, _ = np.polyfit(x_t[:, -1], x_t[:, 2], 1)
        # v_d, _ = np.polyfit(x_t[:, -1], x_t[:, 3], 1)
        # v_x = (v_r+v_l)/2
        # v_y = (v_u+v_d)/2
        # if v_l*v_r > 0 and self.pos[-1][-1] == self.t:
        #     self.speeds.append((np.sqrt(np.square([v_x, v_y]).sum())*np.sign(v_x), self.t))
        # else:
        #     self.speeds.append((0, self.t))


        # if len(self.pos) <= 3:
        #     return self
        # f2 = self.pos[-1]
        # f1 = self.pos[-2]
        # f0 = self.pos[-3]
        #
        # df = (f0 - 4*f1+3*f2)
        # dt = (f2-f0)[-1]/4
        # df_dt = df/(2*dt)
        #
        # # print(self.pos[-1], self.pos[-2], df)
        # if df_dt[0]*df_dt[1] > 0 and df_dt[2]*df_dt[3] > 0 and df_dt[-1] < TIMEOUT:
        #     # print(np.square(df[4:-1]).sum()/df[-1])
        #     self.speeds.append((np.sqrt(np.square(df_dt[4:-1]).sum())*np.sign(df_dt[4]), self.t)) # center_bbox vx, vy
        # else:
        #     self.speeds.append((0, self.t))


        return self

    def render(self, image):
        if self.pos and self.pos[-1][-1] == self.t:
            return render_cros(image,pos=self.pos[-1][4:-1] * self.size[0],  color=(255, 255, 255), thickness=1, size=30)

        return image

    # def render_graphics(self, axs, t, color='r'):
    #     ax_x, ax_vx = axs
    #
    #     if len(self.pos) > 1 and self.pos[-1][-1] == t:
    #         ax_x.plot(np.array(self.pos[-2:])[:,-1], np.array(self.pos[-2:])[:,-3], c=color)
    #
    #
    #     if len(self.speeds) > 1 and self.speeds[-1][-1] == t:
    #         ax_vx.plot(np.array(self.speeds[-2:])[:,1], np.array(self.speeds[-2:])[:,0], c=color)



class PolsunokMass(Method):

    def update(self, labels, t):
        clss = labels.boxes.cls.cpu().numpy().astype(dtype=int)
        confs = labels.boxes.conf.cpu().numpy()
        xywh = labels.boxes.xywh.cpu().numpy()
        if labels.masks:
            mask = labels.masks.data.cpu().numpy().astype(int)

        polsunoks = []
        polsunok_confs = []

        leb_boxs = []

        for i, cls in enumerate(clss):
            if cls == STR2CLS["lebedka_polsunok"] and confs[i] >= CONFS[cls]:
                polsunoks.append(mask[i])
                polsunok_confs.append(confs[i])
            if cls == STR2CLS["box"] and confs[i] >= CONFS[cls]:
                leb_boxs.append(xywh[i])

        if polsunoks and leb_boxs:
            polsunok_mask = polsunoks[np.argmax(polsunok_confs)]
            pos_x = (polsunok_mask * R_Matrix[:, :, 0]).sum()/polsunok_mask.sum()
            pos_y = (polsunok_mask * R_Matrix[:, :, 1]).sum()/polsunok_mask.sum()
            #print("MASS", pos_x/MASK_SIZE, pos_y/MASK_SIZE)

            self.pos.append(np.array([pos_x/MASK_SIZE, pos_y/MASK_SIZE, t]))
        self.t = t

        return self

    def calc_speed(self):
        if len(self.pos) <= 1:
            return

        dx = self.pos[-1] - self.pos[-2]
        if dx[-1] < TIMEOUT:
            self.speeds.append((np.sqrt(np.square(dx[0:-1]).sum())/dx[-1]*np.sign(dx[0]), self.t))  # center_bbox vx, vy
        else:
            self.speeds.append((0, self.t))


        return self


    def render(self, image):
        if self.pos and self.pos[-1][-1] == self.t:
            return render_cros(image, pos=self.pos[-1][:-1]*self.size[0], color=(255, 255, 255), thickness=3, size=10)

        return image

class TrosAndRails(Method):

    k_tros = None
    k_rails = None
    b_tros = None
    b_rails = None


    def reset(self):
        self.k_tros = self.k_rails = self.b_tros = self.b_rails = None

    def update(self, labels, t):
        clss = labels.boxes.cls.cpu().numpy().astype(dtype=int)
        confs = labels.boxes.conf.cpu().numpy()
        xywh = labels.boxes.xywh.cpu().numpy()
        if labels.masks:
            mask = labels.masks.data.cpu().numpy().astype(int)

        tros = []
        tros_bboxs = []
        tros_confs = []
        rails = []
        rails_bboxs = []
        rails_confs = []

        for i, cls in enumerate(clss):
            if cls == STR2CLS["tros"] and confs[i] >= CONFS[cls]:
                tros.append(mask[i])
                tros_bboxs.append(xywh[i])
                tros_confs.append(confs[i])
            if cls == STR2CLS["rails"] and confs[i] >= CONFS[cls]:
                rails.append(mask[i])
                rails_bboxs.append(xywh[i])
                rails_confs.append(confs[i])

        self.t = t
        if tros and rails:
            tros_mask_inds = np.where(tros[np.argmax(tros_confs)] == 1 )
            rails_mask_inds = np.where(rails[np.argmax(rails_confs)] == 1 )
            tros_bbox = tros_bboxs[np.argmax(tros_confs)]
            rails_bbox = rails_bboxs[np.argmax(rails_confs)]

            k1, b1 = fit_line(tros_mask_inds[1], tros_mask_inds[0], tros_bbox) #np.polyfit(tros_mask_inds[1], tros_mask_inds[0], 1)
            k2, b2 = fit_line(rails_mask_inds[1], rails_mask_inds[0], rails_bbox)

            if k1 != k2:
                pos_x = (b1-b2)/(k2-k1)
                pos_y = pos_x*k1+b1
                self.k_tros = k1
                self.b_tros = b1
                self.k_rails = k2
                self.b_rails = b2
                #print("TROS", pos_x / MASK_SIZE, pos_y / MASK_SIZE, '\n')
                self.pos.append(np.array([pos_x/ MASK_SIZE, pos_y/ MASK_SIZE, t]))
                return self

        self.reset()
        return self

    def calc_speed(self):
        if len(self.pos) <= 1:
            return

        dx = self.pos[-1] - self.pos[-2]
        if dx[-1] < TIMEOUT:
            self.speeds.append((np.sqrt(np.square(dx[0:-1]).sum())/dx[-1]*np.sign(dx[0]), self.t))  # center_bbox vx, vy
        else:
            self.speeds.append((0, self.t))

        return self

    def render(self, image):
        image_ = image
        if self.k_tros and self.k_rails and self.b_tros and self.b_rails:
            """draw  lines"""
            image_ = render_line_around_pos(image_, self.pos[-1][:-1]*self.size[0], self.k_tros ,
                                            self.b_tros/MASK_SIZE*self.size[0], color=(5, 245, 230))

            image_ = render_line_around_pos(image_, self.pos[-1][:-1]*self.size[0], self.k_rails ,
                                            self.b_rails/MASK_SIZE*self.size[0], color=(255, 0,  255))

        if self.pos and self.pos[-1][-1] == self.t:
            """draw cros at crospoint"""
            image_ =  render_cros(image_, pos=self.pos[-1][:-1]*self.size[0], color=(240, 145, 86), thickness=2, size=30)

        if self.speeds and self.speeds[-1][-1] == self.t:
            """draw cros at crospoint"""
            ...

        return image_