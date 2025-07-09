from multiprocessing import current_process, Pool, Process
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

from torch.utils.data import Sampler
import random
import numpy as np
import cv2
from time import time, sleep
from tqdm import tqdm
from dataloader_seg import CLS_KEYS

class RamLoader:

    def __init__(self, img_num=100, img_shape=(640, 640, 3), img_paths=None, workers_num=3):
        self.img_shape = img_shape
        print('AIMAGE SHAPE', self.img_shape)
        self.img_num = img_num
        self.img_paths = img_paths
        self.img_paths_len = len(self.img_paths)
        self.stride = 3
        sizE = img_num*self.stride*np.dtype(np.uint8).itemsize*img_shape[0]*img_shape[1]*img_shape[2]

        self.ram_imgs = SharedMemory(create=True, size=sizE, name='RamLoader')
        self.buffer_imgs = np.ndarray((img_num, self.stride, *img_shape), dtype=np.uint8, buffer=self.ram_imgs.buf)
        self.buffer_imgs.fill(0)


        self.ram_masks = SharedMemory(create=True, size=sizE, name='RamLoaderMasks')
        self.buffer_masks = np.ndarray((img_num, *img_shape[:-1]), dtype=np.uint8, buffer=self.ram_masks.buf)
        self.buffer_masks.fill(0)

        self.ram_inds = SharedMemory(create=True, size=self.img_paths_len*np.dtype(np.int32).itemsize, name='RamLoaderInds')
        self.buffer_inds = np.ndarray((self.img_paths_len, ), dtype=np.int32, buffer=self.ram_inds.buf)

        self.ram_progress = SharedMemory(create=True, size=self.img_num * np.dtype(np.uint8).itemsize,name='RamLoaderProgress')
        self.buffer_progress = np.ndarray((self.img_num,), dtype=np.uint8, buffer=self.ram_progress.buf)
        self.buffer_progress.fill(0)

        self.ram_used = SharedMemory(create=True, size=self.img_paths_len * np.dtype(np.uint8).itemsize,name='RamLoaderUsedPathes')
        self.buffer_used = np.ndarray((self.img_paths_len,), dtype=np.uint8, buffer=self.ram_used.buf)
        self.buffer_used.fill(0)

        self.indexes_logger = None
        self.i = 0
        self.workers_num = workers_num

        self.subproc = [] #Process(target=self.load_next_wave)
        self.was_started = False

        # self.ram.close()  # В каждом процессе после использования
        # self.ram.unlink()
        # self.ram_inds.close()  # В каждом процессе после использования
        # self.ram_inds.unlink()
        # exit()
        # for i in range(img_num):
        #     img = np.random.rand(*img_shape)
        #     self.buffer[i] = img


    def finalize(self):
        for pr in self.subproc:
            if pr.is_alive():
                pr.terminate()
        self.ram_imgs.close()
        self.ram_masks.close()
        self.ram_inds.close()
        self.ram_progress.close()
        self.ram_used.close()
        self.ram_imgs.unlink()
        self.ram_masks.unlink()
        self.ram_inds.unlink()
        self.ram_progress.unlink()
        self.ram_used.unlink()

    def on_iter(self):
        self.buffer_used.fill(0)
        self.buffer_progress.fill(0)
        print(f'do_smth {self.i}:', self.indexes_logger[:5])

        t0 = time()

        # for i in tqdm(range(self.img_num), desc="Загрузка изображений"):
        #     self.load_sample(self.indexes_logger[i], i, self.buffer_imgs, self.buffer_inds, self.buffer_progress)

        for pr in self.subproc:
            if pr.is_alive():
                pr.terminate()

        collected_args = [[(self.indexes_logger[i], i) for i in range(start, self.img_num, self.workers_num)] for start in range(self.workers_num)]
        self.subproc = [Process(target=self.pre_load, args=[collected_args_i]) for collected_args_i in collected_args]
        for pr in self.subproc:
            pr.start()
        for pr in self.subproc:
            pr.join()

        print('TOTAL RAM LOADING TIME =', time()-t0)
        print(self.buffer_inds)
        for pr in self.subproc:
            if pr.is_alive():
                pr.terminate()

        self.subproc = [Process(target=self.load_next_wave, args=(start,)) for start in range(self.workers_num)]
        for pr in self.subproc:
            pr.start()

        sleep(2)


    def pre_load(self, collected_args):
        ram_imgs = SharedMemory(name='RamLoader')
        buffer_imgs = np.ndarray((self.img_num, self.stride, *self.img_shape), dtype=np.uint8, buffer=ram_imgs.buf)

        ram_masks = SharedMemory(name='RamLoaderMasks')
        buffer_masks = np.ndarray((self.img_num, *self.img_shape[:-1]), dtype=np.uint8, buffer=ram_masks.buf)

        ram_inds = SharedMemory(name='RamLoaderInds')
        buffer_inds = np.ndarray((self.img_paths_len,), dtype=np.int32, buffer=ram_inds.buf)

        ram_progress = SharedMemory(name='RamLoaderProgress')
        buffer_progress = np.ndarray((self.img_num,), dtype=np.uint8, buffer=ram_progress.buf)
        for arg in collected_args:
            self.load_sample(*arg, buffer_imgs,buffer_masks, buffer_inds, buffer_progress)


    def load_sample(self, idx_from, idx_to, buffer_imgs, buffer_masks, buffer_inds, buffer_progress):
        buffer_inds[idx_from] = idx_to
        for i in range(self.stride):
            img = cv2.imread(self.img_paths[idx_from] + f"/{i}.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(self.img_shape[1], self.img_shape[0]))
            buffer_imgs[idx_to][i] = img

        packed_masks = np.zeros(shape=(*self.img_shape[:-1],), dtype=np.uint8)
        for cls_ind, cls in enumerate(CLS_KEYS):
            str_path = self.img_paths[idx_from].replace("images", f"mask_{cls}") + f".png"
            if Path(str_path).exists():
                mask = cv2.imread(str_path)[..., 0]
                mask = cv2.resize(mask, dsize=(self.img_shape[1], self.img_shape[0]))
            else:
                mask = np.zeros(shape=(*self.img_shape[:-1],), dtype=np.uint8)

            packed_masks |= (mask.astype(np.bool_).astype(np.uint8) << cls_ind)

        buffer_masks[idx_to] = packed_masks
        buffer_progress[idx_to] = 1



    # WORKS IN OTHER PROCESS
    def load_next_wave(self, start):

        print('STARTED WAVE LOADING')
        working = True
        sample_queue = self.img_num+start

        ram_imgs = SharedMemory(name='RamLoader')
        buffer_imgs = np.ndarray((self.img_num, self.stride, *self.img_shape), dtype=np.uint8, buffer=ram_imgs.buf)


        ram_masks = SharedMemory(name='RamLoaderMasks')
        buffer_masks = np.ndarray((self.img_num, *self.img_shape[:-1]), dtype=np.uint8, buffer=ram_masks.buf)

        ram_inds = SharedMemory(name='RamLoaderInds')
        buffer_inds = np.ndarray((self.img_paths_len,), dtype=np.int32, buffer=ram_inds.buf)

        ram_progress = SharedMemory(name='RamLoaderProgress')
        buffer_progress = np.ndarray((self.img_num,), dtype=np.uint8, buffer=ram_progress.buf)

        ram_used = SharedMemory(name='RamLoaderUsedPathes')
        buffer_used = np.ndarray((self.img_paths_len,), dtype=np.uint8, buffer=ram_used.buf)

        dt = max(int(self.img_num/100), 1)
        loaded = 0
        missed = 0

        times_done = 0
        t0 = time()
        while working:
            times_done+=1
            for i in range(start, self.img_num, self.workers_num):
                flag = buffer_progress[i]


                if sample_queue >= self.img_paths_len:
                    working = False
                    print('STOPPING LOADING WAVE')
                    break

                if start==0 and i%dt==0:
                    with open(Path(__file__).parent/'progress.txt', 'w') as f:
                        print(f'start {start} | t0 {t0:0.2f}', file=f)
                        print(f'missed {missed}', file=f)
                        print(f'loaded {loaded}', file=f)
                        print(f'missing speed: {missed/(time()-t0)}', file=f)
                        print(f'loading speed: {loaded/(time()-t0)}', file=f)
                        print(f'now progress {i}/{self.img_num}', file=f)
                        print(f'total progress {sample_queue}/{self.img_paths_len}', file=f)


                if flag == 0:
                    if buffer_used[self.indexes_logger[sample_queue]] == 0:
                        loaded+=1
                        self.load_sample(self.indexes_logger[sample_queue], i, buffer_imgs, buffer_masks, buffer_inds, buffer_progress)
                    else:
                        missed += 1
                    sample_queue+=self.workers_num

                # if flag == 2:
                #     buffer_progress[i] = 0
                #     sample_queue += self.workers_num
                #     missed+=1
            # sleep(0.5)



class TrackedRandomSampler(Sampler):

    def __init__(self, data_source, generator=None, ram_loader=None):
        self.data_source = data_source
        self.generator = generator
        self.ram_loader = ram_loader

    def __iter__(self):
        n = len(self.data_source)
        indices = list(range(n))
        random.shuffle(indices)
        self.ram_loader.indexes_logger = indices.copy()
        print('__iter__:', self.ram_loader.indexes_logger[:5])
        self.ram_loader.on_iter()

        yield from indices

    def __len__(self):
        return len(self.data_source)


if __name__=="__main__":
    lst = ['pasha_stats', 'RamLoader', 'RamLoaderMasks', 'RamLoaderInds', 'RamLoaderProgress', 'RamLoaderUsedPathes']
    for n in lst:
        try:
            ram = SharedMemory(name=n)
            ram.unlink()
        except Exception as e:
            print(e)
    exit()
    a = RamLoader(10000)
    sleep(10)
    a.finalize()


