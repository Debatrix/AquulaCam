import os.path as osp
import random
import pickle
import lmdb
import copy
from glob import glob
from numpy.core.fromnumeric import resize
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as vtf
import torch.nn.functional as nnf
from multiprocessing import Pool, cpu_count

DATA_IN_RAM = {}
STAY_IN_RAM = False


def rand(size=1, range=1):
    assert (isinstance(size, int)
            or isinstance(size, tuple)), 'size must be int or tuple'
    if isinstance(size, int):
        size = (size, )
    return (np.random.rand(*size) - 0.5) * 2 * range


def get_img(env, key, size):
    if key in DATA_IN_RAM:
        img = copy.deepcopy(DATA_IN_RAM[key])
    else:
        img = read_lmdb_img(env, key, size)
        if STAY_IN_RAM:
            DATA_IN_RAM[key] = copy.deepcopy(img)
    return img


def load_img_to_ram(img_list):
    for path in tqdm(img_list, ascii=True):
        img_name = osp.basename(path).split('.')[0]
        DATA_IN_RAM[img_name] = cv2.imread(
            path, cv2.IMREAD_GRAYSCALE)[np.newaxis, :, :]


def read_lmdb_img(env, key, size):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    if C == 1:
        img = img_flat.reshape(C, H, W)
    else:
        img = img_flat.reshape(H, W, C)
        img = img.mean(2)
        img = img[np.newaxis, :, :]
    return img.copy()


def img_lrud_move(img, x=None, y=None):
    '''
    left-right-up-down movement
    '''
    _, h, w = img.shape

    if x > 0:
        img = np.pad(img, ((0, 0), (0, 0), (0, x)), 'reflect')
        img = img[:, :, x:w + x]
    elif x < 0:
        img = np.pad(img, ((0, 0), (0, 0), (np.abs(x), 0)), 'reflect')
        img = img[:, :, :w]
    if y > 0:
        img = np.pad(img, ((0, 0), (0, y), (0, 0)), 'reflect')
        img = img[:, y:h + y, :]
    elif y < 0:
        img = np.pad(img, ((0, 0), (np.abs(y), 0), (0, 0)), 'reflect')
        img = img[:, :h, :]
    return img


def gaussian(array_like_hm, mean, sigma):
    """modifyed version normal distribution pdf, vector version"""
    array_like_hm -= mean
    x_term = array_like_hm[:, 0]**2
    y_term = array_like_hm[:, 1]**2
    exp_value = -(x_term + y_term) / 2 / pow(sigma, 2)
    return np.exp(exp_value)


def generate_heatmap(points, size):
    '''
    :param points:  [[x,y,sigma]]
    :return: heatmap
    '''
    x = np.arange(size[1], dtype=np.float)
    y = np.arange(size[0], dtype=np.float)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.c_[xx.ravel(), yy.ravel()]

    heatmap = []

    for joint_id in range(len(points)):
        mu_x = int(points[joint_id][0])
        mu_y = int(points[joint_id][1])
        sigma = points[joint_id][2]
        zz = gaussian(xxyy.copy(), (mu_x, mu_y), sigma)
        heatmap.append(zz.reshape(size))
    heatmap = np.stack(heatmap, 0)

    return heatmap


def generate_mask(points, size):
    '''
    :param points:  [[x,y,sigma]]
    :return: mask
    '''
    mask = np.zeros(size).astype(np.uint8)
    for point in points:
        point = np.round(point).astype(np.int)
        mask = cv2.circle(mask, (point[0], point[1]), point[2], (1, 1, 1), -1)
    return mask


# ##############################################################################
def hut_collate_wrapper(batch):
    envs, starts = list(zip(*batch))
    return HutEnvBatch(envs), torch.tensor(starts)


class HutEnv():
    def __init__(self,
                 lmdb_path,
                 seq,
                 topk=0.05,
                 fb_move='None',
                 lrud_move='None',
                 max_fb_move=0.075,
                 max_lrud_move=(100, 50),
                 max_frame=20,
                 mask_down=8):
        self.lmdb_env = None
        self.lmdb_path = lmdb_path

        self.mask_down = mask_down

        self.cur_frame = -1
        self.max_frame = max_frame
        max_lrud_move = np.array(max_lrud_move)
        self._init_move(fb_move, max_fb_move, lrud_move, max_lrud_move)

        # Bug: channel_num
        if isinstance(seq[0][1], str):
            img_shape = seq[0][1][1:-1].split(',')
            self.img_shape = (1, int(img_shape[0]), int(img_shape[1]))
        else:
            self.img_shape = (3, seq[0][1][0], seq[0][1][1])
        self.seq = {x[3]: x for x in seq}
        self.position = np.sort(np.array([x[3] for x in seq]))
        # self.topk = sorted([x[3] for x in seq],
        #                    key=lambda x: np.abs(x))[:int(len(seq) * topk)]

    def _init_move(self, fb_move, max_fb_move, lrud_move, max_lrud_move):
        if fb_move.lower() == 'random':
            fb_move = random.choice(
                ['linear', 'cos', 'rlinear', 'rcos', 'random', 'none'])
        if lrud_move.lower() == 'random':
            lrud_move = random.choice(
                ['linear', 'cos', 'rlinear', 'rcos', 'random', 'none'])

        if fb_move.lower() == 'linear':
            start, end = np.sort(rand(2))
            self.fb_move = np.linspace(start, end,
                                       self.max_frame) * max_fb_move
        elif fb_move.lower() == 'cos':
            start, end = np.sort(rand(2, 2 * np.pi))
            self.fb_move = np.cos(np.linspace(start, end,
                                              self.max_frame)) * max_fb_move
        elif fb_move.lower() == 'rlinear':
            start, end = np.sort(rand(2))
            self.fb_move = (
                np.sort(np.random.uniform(start, end, self.max_frame)) +
                np.random.normal(0, 0.5, self.max_frame) * 0.2) * max_fb_move
        elif fb_move.lower() == 'rcos':
            start, end = np.sort(rand(2, 2 * np.pi))
            self.fb_move = (
                np.cos(np.sort(np.random.uniform(start, end, self.max_frame)))
                + np.random.normal(0, 0.5, self.max_frame) * 0.2) * max_fb_move
        elif fb_move.lower() == 'random':
            self.fb_move = np.random.normal(0, 0.5,
                                            self.max_frame) * max_fb_move
        else:
            self.fb_move = None

        if lrud_move.lower() == 'linear':
            start, end = np.sort(rand((2, 2)), axis=0)
            self.lrud_move = np.stack(
                (np.linspace(start[0], end[0], self.max_frame),
                 np.linspace(start[1], end[1], self.max_frame)),
                axis=1) * max_lrud_move
        elif lrud_move.lower() == 'cos':
            start, end = np.sort(rand((2, 2), 2 * np.pi), axis=0)
            self.lrud_move = np.cos(
                np.stack((np.linspace(start[0], end[0], self.max_frame),
                          np.linspace(start[1], end[1], self.max_frame)),
                         axis=1)) * max_lrud_move
        elif lrud_move.lower() == 'rlinear':
            start, end = np.sort(rand((2, 2)), axis=0)
            self.lrud_move = (np.stack(
                (np.sort(np.random.uniform(start[0], end[0], self.max_frame)),
                 np.sort(np.random.uniform(start[1], end[1], self.max_frame))),
                axis=1) + np.random.normal(0, 0.5, (self.max_frame, 2)) *
                              0.2) * max_lrud_move
        elif lrud_move.lower() == 'rcos':
            start, end = np.sort(rand((2, 2), 2 * np.pi), axis=0)
            self.lrud_move = (np.cos(
                np.stack(
                    (np.sort(
                        np.random.uniform(start[0], end[0], self.max_frame)),
                     np.sort(
                         np.random.uniform(start[1], end[1], self.max_frame))),
                    axis=1)) + np.random.normal(0, 0.5, (self.max_frame, 2)) *
                              0.2) * max_lrud_move
        elif lrud_move.lower() == 'random':
            self.lrud_move = np.random.normal(
                0, 0.5, (self.max_frame, 2)) * max_lrud_move
        else:
            self.lrud_move = None
        if self.lrud_move is not None:
            self.lrud_move = self.lrud_move.astype(np.int)

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.lmdb_env = lmdb.open(self.lmdb_path,
                                  readonly=True,
                                  lock=False,
                                  readahead=False,
                                  meminit=False)

    @staticmethod
    def _find_nearest(target, array):
        return array.flat[np.abs(array - float(target)).argmin()]

    @staticmethod
    def _find_side(target, array):
        idx = np.abs(array - float(target)).argmin()
        p1 = array[idx]
        if p1 > target and idx - 1 > 0:
            p2 = array[idx - 1]
        elif p1 < target and idx + 1 < len(array):
            p2 = array[idx + 1]
        else:
            p2 = p1
        # np.abs(p1 - target), np.abs(p2 - target)
        return p1, p2

    # def topk_check(self, position):
    #     return position in self.topk

    def next(self, offset, cur=0):
        self.cur_frame += 1
        offset_x, offset_y, offset_z = 0, 0, 0
        if self.lmdb_env is None:
            self._init_lmdb()
        position = offset + cur
        if self.fb_move is not None:
            offset_z = self.fb_move[self.cur_frame % self.max_frame]
            position += offset_z
        # position = self._find_nearest(
        #     position, self.position) if position not in self.seq else position
        # img = get_img(self.lmdb_env, self.seq[position][0],
        #                     self.img_shape)
        # points = np.array(self.seq[position][-1]).reshape((2, 3))
        if position in self.seq:
            img = get_img(self.lmdb_env, self.seq[position][0], self.img_shape)
            points = np.array(self.seq[position][-1]).reshape((2, 3))
        else:
            p1, p2 = self._find_side(position, self.position)
            if p1 == p2:
                img = get_img(self.lmdb_env, self.seq[p1][0], self.img_shape)
                points = np.array(self.seq[p1][-1]).reshape((2, 3))
            else:
                img = (
                    (get_img(self.lmdb_env, self.seq[p1][0], self.img_shape) *
                     np.abs(position - p1) / np.abs(p2 - p1)) +
                    (get_img(self.lmdb_env, self.seq[p2][0], self.img_shape) *
                     np.abs(position - p2) / np.abs(p2 - p1)))
                points = (
                    (np.array(self.seq[p1][-1]) + np.array(self.seq[p2][-1])) /
                    2).reshape((2, 3))

        img = np.clip(img, 0, 255)
        # img = cv2.resize(img[0],
        #                  None,
        #                  fx=1 / self.mask_down,
        #                  fy=1 / self.mask_down)[np.newaxis, :, :]
        if self.lrud_move is not None:
            offset_x, offset_y = self.lrud_move[self.cur_frame %
                                                self.max_frame]
            img = img_lrud_move(img, offset_x, offset_y)
            points = points - np.array([offset_x, offset_y, 0])

        points = points / self.mask_down
        mask_shape = (self.img_shape[1] // self.mask_down,
                      self.img_shape[2] // self.mask_down)
        # mask = generate_heatmap(points, mask_shape).sum(0)
        mask = generate_mask(points, mask_shape)

        img = torch.from_numpy(img / 255.0).to(torch.float32)
        mask = torch.from_numpy(mask).to(torch.float32)
        position = torch.tensor(position, dtype=torch.float32)
        move = torch.tensor((offset_x, offset_y, offset_z),
                            dtype=torch.float32)
        img = nnf.interpolate(img.unsqueeze(0),
                              (self.img_shape[1] // self.mask_down,
                               self.img_shape[2] // self.mask_down),
                              mode='bilinear')[0]
        if self.lrud_move is not None:
            if torch.rand(1).item() > 0.5:
                img = torch.unsqueeze(img, 0)
                mask = torch.unsqueeze(torch.unsqueeze(mask, 0), 0)
                f = torch.rand(1).item() + 0.5
                img = nnf.interpolate(
                    img, (int(img.shape[-2] * f), int(img.shape[-1] * f)),
                    mode='bilinear')
                mask = nnf.interpolate(
                    mask, (int(mask.shape[-2] * f), int(mask.shape[-1] * f)),
                    mode='bilinear')
                if f > 1:
                    img = img[0, :, :self.img_shape[1] //
                              self.mask_down, :self.img_shape[2] //
                              self.mask_down]
                    mask = mask[0, 0, :self.img_shape[1] //
                                self.mask_down, :self.img_shape[2] //
                                self.mask_down]
                elif f < 1:
                    padx = (self.img_shape[1] // self.mask_down - img.shape[2])
                    pady = (self.img_shape[2] // self.mask_down - img.shape[3])
                    img = nnf.pad(img, (pady, 0, padx, 0),
                                  "replicate")[0, :, :, :]
                    mask = nnf.pad(
                        mask,
                        (pady, 0, padx, 0),
                        "replicate",
                    )[0, 0, :, :]
        mask = mask.to(torch.long)
        return img, mask, position, move


class HutEnvBatch():
    def __init__(self, env_list):
        self.env_list = env_list
        self.max_frame = env_list[0].max_frame
        self._pin_memory = False

    def pin_memory(self):
        self._pin_memory = True
        return self

    def __len__(self):
        return len(self.env_list)

    def __call__(self, offsets, curs=None):
        imgs, masks, positions, movement = [], [], [], []
        for idx, offset in enumerate(offsets):
            cur = 0 if curs is None else curs[idx].item()
            img, mask, position, move = self.env_list[idx].next(
                offset.item(), cur)
            imgs.append(img)
            masks.append(mask)
            positions.append(position)
            movement.append(move)
        imgs = torch.stack(imgs, dim=0)
        masks = torch.stack(masks, dim=0)
        positions = torch.stack(positions, dim=0)
        movement = torch.stack(movement, dim=0)
        if self._pin_memory:
            imgs = imgs.pin_memory()
            masks = masks.pin_memory()
            # positions = positions.pin_memory()
            # movement = movement.pin_memory()
        return imgs, masks, positions, movement


class HutSeqDataset(data.Dataset):
    def __init__(self,
                 path,
                 mode,
                 less_data=True,
                 topk=0.05,
                 fb_move='None',
                 lrud_move='None',
                 max_fb_move=0.15,
                 max_lrud_move=(600, 600),
                 max_frame=50,
                 mask_down=8,
                 load_to_ram=False):
        self.mode = mode
        self.path = path if path[-1] != '/' else path[:-1]
        self.env_setting = (topk, fb_move, lrud_move, max_fb_move,
                            max_lrud_move, max_frame + 1, mask_down)
        self.lmdb_path = osp.join(path,
                                  osp.basename(path).split('/')[-1] + '.lmdb')
        self.img_path = osp.join(path, 'png')

        with open(osp.join(path, 'meta_info.pkl'), "rb") as f:
            meta_info = pickle.load(f)
        self.all_seq_list = meta_info['label2']
        if mode == 'train':
            label_list = meta_info['split'][0]
        elif mode == 'val':
            label_list = meta_info['split'][1]
        else:
            label_list = meta_info['split'][2]
        self.seq_list = self._get_sequence_list(label_list)
        random.shuffle(self.seq_list)
        if less_data == True:
            # data_num = 1024 if mode == 'train' else 128
            self.seq_list = self.seq_list[:int(0.02 * len(self.seq_list))]
        elif isinstance(less_data, float):
            self.seq_list = self.seq_list[:int(less_data * len(self.seq_list))]
        if load_to_ram:
            task_num = cpu_count()
            img_list = []
            tasks = []
            for l in [x[0] for x in self.seq_list]:
                img_list += [
                    osp.join(self.img_path, y[0] + '.png')
                    for y in self.all_seq_list[l]
                ]
            img_list = list(set(img_list))
            task_len = int(len(img_list) / task_num) + 1
            tasks = [
                img_list[i * task_len:(i + 1) * task_len]
                for i in range(task_num)
            ]
            # for path in tqdm(img_list, ascii=True, desc='Loading data'):
            #     img_name =  osp.basename(path).split('.')[0]
            #     DATA_IN_RAM[img_name] = cv2.imread(
            #         path, cv2.IMREAD_GRAYSCALE)[np.newaxis, :, :]
            p = Pool(task_num)
            for i in tasks:
                p.apply_async(func=load_img_to_ram, args=(i, ))
            p.close()
            p.join()

    def _get_sequence_list(self, label_list):
        seq_list = []
        for l in label_list:
            imglist = sorted(self.all_seq_list[l])
            thr = int(0.2 * len(imglist))
            for idx in range(len(imglist) - 2 * thr):
                seq_list.append((l, self.all_seq_list[l][thr + idx][3]))
        return seq_list

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, item):
        task = self.seq_list[item]
        env = HutEnv(self.lmdb_path, self.all_seq_list[task[0]],
                     *self.env_setting)
        start = task[1]
        return env, start


# ##############################################################################
class HUTDataset(data.Dataset):
    def __init__(self,
                 path,
                 mode,
                 less_data=True,
                 mask_down=8,
                 lrud_move=False):
        super(HUTDataset, self).__init__()
        self.lmdb_env = None
        self.mode = mode
        self.path = path
        self.lmdb_path = osp.join(path,
                                  osp.basename(path).split('/')[-1] + '.lmdb')
        self.img_path = osp.join(path, 'png')
        self.mask_down = mask_down
        self.lrud_move = lrud_move

        with open(osp.join(path, 'meta_info.pkl'), "rb") as f:
            meta_info = pickle.load(f)
        img_list = meta_info['label2']

        if isinstance(img_list['0001'][0][1], str):
            img_shape = img_list['0001'][0][1][1:-1].split(',')
            self.img_shape = (1, int(img_shape[0]), int(img_shape[1]))
        else:
            self.img_shape = (3, img_list['0001'][0][1][0],
                              img_list['0001'][0][1][1])

        if mode == 'train':
            label_list = meta_info['split'][0]
        elif mode == 'val':
            label_list = meta_info['split'][1]
        else:
            label_list = meta_info['split'][2]
        self.img_info_list = []
        for l in label_list:
            self.img_info_list += img_list[l]
        random.shuffle(self.img_info_list)
        if less_data:
            # data_num = 1024 if mode == 'train' else 128
            self.img_info_list = self.img_info_list[:int(0.1 *
                                                         len(self.img_info_list
                                                             ))]

    def __len__(self):
        return len(self.img_info_list)

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.lmdb_env = lmdb.open(self.lmdb_path,
                                  readonly=True,
                                  lock=False,
                                  readahead=False,
                                  meminit=False)

    def __getitem__(self, item):
        if self.lmdb_env is None:
            self._init_lmdb()
        info = self.img_info_list[item]
        img = read_lmdb_img(self.lmdb_env, info[0], self.img_shape)
        img = cv2.resize(img[0],
                         None,
                         fx=1 / self.mask_down,
                         fy=1 / self.mask_down)
        points = np.array(info[-1]).reshape((2, 3))

        if self.lrud_move and torch.rand(1).item() > 0.5:
            ksize = np.random.randint(1, 5) * 2 + 1
            sigma = np.random.rand(1)[0] * 10
            img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
            offset_x = np.random.randint(int(0.2 * self.img_shape[2]))
            offset_y = np.random.randint(int(0.2 * self.img_shape[1]))
            img = img[np.newaxis, :, :]
            img = img_lrud_move(img, offset_x, offset_y)
            points = points - np.array([offset_x, offset_y, 0])
        else:
            img = img[np.newaxis, :, :]
        points = points // self.mask_down
        mask_shape = (self.img_shape[1] // self.mask_down,
                      self.img_shape[2] // self.mask_down)
        mask = generate_heatmap(points, mask_shape).sum(0)
        img = torch.from_numpy(img / 255.0).to(torch.float32)
        mask = torch.from_numpy(mask).to(torch.float32)
        if self.lrud_move:
            if torch.rand(1).item() > 0.5:
                img = torch.unsqueeze(img, 0)
                mask = torch.unsqueeze(torch.unsqueeze(mask, 0), 0)
                f = torch.rand(1).item() + 0.5
                img = nnf.interpolate(
                    img, (int(img.shape[-2] * f), int(img.shape[-1] * f)))
                mask = nnf.interpolate(
                    mask, (int(mask.shape[-2] * f), int(mask.shape[-1] * f)))
                if f > 1:
                    img = img[0, :, :self.img_shape[1] //
                              self.mask_down, :self.img_shape[2] //
                              self.mask_down]
                    mask = mask[0, 0, :self.img_shape[1] //
                                self.mask_down, :self.img_shape[2] //
                                self.mask_down]
                elif f < 1:
                    padx = (self.img_shape[1] // self.mask_down - img.shape[2])
                    pady = (self.img_shape[2] // self.mask_down - img.shape[3])
                    img = nnf.pad(img, (pady, 0, padx, 0), "constant",
                                  0)[0, :, :, :]
                    mask = nnf.pad(mask, (pady, 0, padx, 0), "constant",
                                   0)[0, 0, :, :]
        positions = torch.tensor(info[3])
        return img, mask, positions


if __name__ == "__main__":
    dataset = HutSeqDataset('dataset/HutIris-Blur',
                            'test',
                            max_frame=9,
                            less_data=False,
                            fb_move='random',
                            lrud_move='random',
                            max_fb_move=0.5,
                            max_lrud_move=(1000, 1000),
                            mask_down=8)
    env, start = dataset[0]
    for i in tqdm(range(10)):
        img, mask, position, move = env.next(start)
        # print(img.shape, mask.shape, start, position, move)
        cv2.imwrite('checkpoints/move/img_{}.png'.format(i),
                    (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))
        img = img[0].numpy()
        mask = mask.numpy()
        # mask = cv2.resize(mask, None, fx=8, fy=8)
        img = np.clip(np.stack(
            (img, img + mask * 0.5, img), axis=2), 0, 1) * 255
        cv2.imwrite('checkpoints/move/mask_{}.png'.format(i),
                    img.astype(np.uint8))
    # dataset = HUTDataset('dataset/HutIris-Blur',
    #                      'test',
    #                      less_data=True,
    #                      mask_down=8,
    #                      lrud_move=True)
    # for img, mask, positions in dataset:
    #     print(img.shape, mask.shape, positions)
