import cv2
import numpy as np

CONFS = {
0: 0.2, #6,
1: 0.2, #0.75,
2: 0.6,
3: 0.7,
4: 0.6,
5: 0.75,
6: 0.75
}
INTERSECTION_PROB = 0.9
STR2CLS = {"pipe_1_end": 0, "ecn_tros": 1, "spider": 2, "kops_gate": 3, "kops_other":4, "wheel_on_stick": 5, "gis_tros": 6,}

def is_intersected(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    return  ((b2_x1 <= b1_x1 <= b2_x2 or b2_x1 <= b1_x2 <= b2_x2 or b1_x1 <= b2_x2 <= b1_x2)
             and (b2_y1 <= b1_y1 <= b2_y2 or b2_y1 <= b1_y2 <= b2_y2 or b1_y1 <= b2_y2 <= b1_y2))


def hardly_intersected(mask1, mask2):
    """
        means mask1 is almost inside mask2
        return float [0:1], where 1 is fully inside, 0 - are not intersected
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)) #31 if full hd
    expanded_mask = cv2.dilate(mask2, kernel, iterations=1)
    return (expanded_mask*mask1).sum()/(mask1).sum()

class P:
    ECN = "ECN"
    GIS = "GIS"
    KOPS = "KOPS"
    UNDEF = "OTHER"

    DICT = {"ECN": 0,
            "GIS": 1,
            "KOPS": 2}

    @staticmethod
    def by_ind(ind: int):
        if ind == 0:
            return P.ECN
        if ind == 1:
            return P.GIS
        if ind == 2:
            return P.KOPS
        return P.UNDEF

    @staticmethod
    def get_ind(proc: str):
        return P.DICT[proc]


class Queues:
    buflen = 100
    q_ecn = np.array([0 for _ in range(buflen)], dtype=np.float32)
    q_gis = np.array([0 for _ in range(buflen)], dtype=np.float32)
    q_kops = np.array([0 for _ in range(buflen)], dtype=np.float32)
    q_undef = np.array([0 for _ in range(buflen)], dtype=np.float32)

    @staticmethod
    def update(ecn, gis, kops):
        Queues.q_ecn[:-1] = Queues.q_ecn[1:]
        Queues.q_ecn[-1] = ecn
        Queues.q_gis[:-1] = Queues.q_gis[1:]
        Queues.q_gis[-1] = gis
        Queues.q_kops[:-1] = Queues.q_kops[1:]
        Queues.q_kops[-1] = kops

        Queues.q_undef[:-1] = Queues.q_undef[1:]
        Queues.q_undef[-1] = 0 #if ecn+gis+kops > 0 else 1

    @staticmethod
    def start():
        Queues.q_ecn[:] = 0
        Queues.q_gis[:] = 0
        Queues.q_kops[:] = 0
        #Queues.q_undef[:] = 0.8


class Validator:

    process: str = P.UNDEF
    last_proc_max: str = P.UNDEF
    min_aver = 0.8

    prob_intersect_pipes = 0.0

    min_vhod = 0.5
    min_vyhod = 0.2
    values: list
    save_img = False
    num = 0

    @staticmethod
    def update(labels):
        Queues.update(float(Validator.is_ecn(labels)),
                      float(Validator.is_gis(labels)),
                      float(Validator.is_kops(labels)))
        #print(labels.masks.data.cpu().numpy()) #.astype(dtype=int)
        Validator.values = [
            Queues.q_ecn.mean(),
            Queues.q_gis.mean(),
            Queues.q_kops.mean(),
            #Queues.q_undef.mean()
        ]
        # ind_max = np.argmax(Validator.values)
        # Validator.last_proc_max = P.by_ind(ind_max)
        # if Validator.last_proc_max != Validator.process and Validator.values[ind_max] >= Validator.min_aver and Validator.is_human_on_board():
        #     Validator.process = Validator.last_proc_max


        if Validator.process != P.UNDEF:
            if Validator.values[P.get_ind(Validator.process)] < Validator.min_vyhod:
                Validator.process = P.UNDEF
                for i in range(3):
                    if Validator.values[i] >= Validator.min_vhod:
                        Validator.process = P.by_ind(i)
        else:
            for i in range(3):
                if Validator.values[i] >= Validator.min_vhod:
                    Validator.process = P.by_ind(i)




    @staticmethod
    def is_gis(labels) -> bool:

        """
        -Основной варик:
            Если есть вертикальный гис-трос,
            который пересекает колесо или
            есть ветрикальный гис-трос и
            гис-трос в трубе - Тру


        -Запасной варик:
            Если есть колесо либо снизу,
            либо сверху - Тру.
            *Могут находиться просто
            *лежачие колеса
        """
        clss = labels.boxes.cls.cpu().numpy().astype(dtype=int)
        confs = labels.boxes.conf.cpu().numpy()
        xywh = labels.boxes.xywh.cpu().numpy()
        troses = []
        wheels = []
        for i, cls in enumerate(clss):
            if cls == STR2CLS["gis_tros"] and confs[i] >= CONFS[cls]:
                troses.append(xywh[i])
            if cls == STR2CLS["wheel_on_stick"] and confs[i] >= CONFS[cls]:
                wheels.append(xywh[i])
        for tros in troses:
            for wheel in wheels:
                if is_intersected(tros, wheel):
                    return True
        if 2 <= len(troses) <= 3:
            return True
        return False

    @staticmethod
    def is_kops(labels) -> bool:
        """
        Если есть копс на площадке ыыыыыы - Тру
        """
        clss = labels.boxes.cls.cpu().numpy().astype(dtype=int)
        confs = labels.boxes.conf.cpu().numpy()
        #xywh = labels.boxes.xywh.cpu().numpy()
        for i, cls in enumerate(clss):
            if cls == STR2CLS["kops_gate"] and confs[i] >= CONFS[cls] :
                return True
        #print(clss, confs, xywh)
        return False

    @staticmethod
    def is_ecn(labels) -> bool:
        """
        Если есть эцн-трос, который
        пересекает спайдер или
        фланец(надо сделать....) - Тру
        """
        clss = labels.boxes.cls.cpu().numpy().astype(dtype=int)
        confs = labels.boxes.conf.cpu().numpy()
        xywh = labels.boxes.xywh.cpu().numpy()
        masks = None
        if labels.masks:
            masks = labels.masks.data.cpu().numpy().astype(np.uint8)
        troses = []
        troses_masks = []
        spiders = []
        pipes_masks = []
        for i, cls in enumerate(clss):
            if cls == STR2CLS["ecn_tros"] and confs[i] >= CONFS[cls] and masks is not None:
                troses.append(xywh[i])
                troses_masks.append(masks[i])
            if cls == STR2CLS["spider"] and confs[i] >= CONFS[cls]:
                spiders.append(xywh[i])
            if cls == STR2CLS["pipe_1_end"] and confs[i] >= CONFS[cls] and masks is not None:
                pipes_masks.append(masks[i])

        Validator.prob_intersect_pipes = 0
        for tros in troses_masks:
            for pipe in pipes_masks:
                Validator.prob_intersect_pipes = hardly_intersected(tros, pipe)
                if Validator.prob_intersect_pipes > INTERSECTION_PROB:
                    return False

        for tros in troses:
            for spider in spiders:
                if is_intersected(tros, spider):
                    return True
        # if troses:
        #     return True


        return False

    @staticmethod
    def is_human_on_board() -> bool:
        return True

    @staticmethod
    def start():
        Queues.start()
        Validator.process = P.UNDEF
        Validator.last_proc_max = P.UNDEF
        Validator.values = []
        Validator.save_img = False
        Validator.num = 0

    @staticmethod
    def render(image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        h,w,_ = image.shape
        min_h = h-70
        min_w = w-100
        dh = 20

        # fontScale
        fontScale = 0.5

        # Blue color in BGR
        red = (0, 0, 255)
        green = (0, 255, 0)

        # Line thickness of 2 px
        thickness = 2
        #print(Validator.process)
        # for i in range(len(Validator.values)):
        #     image = cv2.putText(image, f'{P.by_ind(i)}: {Validator.values[i]:.02f}' + ('!' if Validator.last_proc_max == P.by_ind(i) else ''),
        #                         (min_w, min_h+dh*i), font, fontScale, (green if Validator.process == P.by_ind(i) else red),
        #                         thickness, cv2.LINE_AA)

        for i in range(len(Validator.values)):
            image = cv2.putText(image, f'{P.by_ind(i)}: {Validator.values[i]:.02f}',
                                (min_w, min_h+dh*i), font, fontScale, (green if Validator.process == P.by_ind(i) else red),
                                thickness, cv2.LINE_AA)

        image = cv2.putText(image, f'ECN-PIPE INTER: {Validator.prob_intersect_pipes:.02f}',
                            (min_w-80, min_h + dh * (-1)), font, fontScale,
                            (green if Validator.process == P.ECN else red),
                            thickness, cv2.LINE_AA)
        return image
