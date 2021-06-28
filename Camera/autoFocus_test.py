import os
import cv2
import time
import pickle
import random
import os.path as osp
from tqdm import tqdm

import torch
import numpy as np

from Camera.BaseCamera import BaseCamera, fmeasure


# #######################################################################
def get_sequence_list(img_list, label_list):
    seq_list = []
    _seq_list = []
    for l in label_list:
        imglist = sorted(img_list[l])
        idx = 0
        while idx < 54:
            num = np.random.randint(int(0.5 * len(imglist)), len(imglist))
            start = np.random.randint(0, len(imglist) - num)
            seq = imglist[start:start + num]
            m = np.abs(np.array([x[3] for x in seq])).min()
            if m > 0.01:
                continue
            _seq = '{}{}{}'.format(seq[0][0].split('_')[0], start, num)
            if _seq not in _seq_list:
                seq_list.append(seq)
                _seq_list.append(_seq)
                idx += 1
    np.random.shuffle(seq_list)
    return seq_list


def rand(size=1, range=1):
    assert (isinstance(size, int)
            or isinstance(size, tuple)), 'size must be int or tuple'
    if isinstance(size, int):
        size = (size, )
    return (np.random.rand(*size) - 0.5) * 2 * range


def img_lrud_move(img, x=0, y=0):
    '''
    left-right-up-down movement
    '''
    h, w = img.shape

    if x > 0:
        img = np.pad(img, ((0, 0), (0, x)), 'edge')
        img = img[:, x:w + x]
    elif x < 0:
        img = np.pad(img, ((0, 0), (np.abs(x), 0)), 'edge')
        img = img[:, :w]
    if y > 0:
        img = np.pad(img, ((0, y), (0, 0)), 'edge')
        img = img[y:h + y, :]
    elif y < 0:
        img = np.pad(img, ((np.abs(y), 0), (0, 0)), 'edge')
        img = img[:h, :]
    return img


def init_move(fb_move,
              lrud_move,
              max_fb_move=0.25,
              max_lrud_move=0.15,
              max_frame=50):
    lrud_move = str(lrud_move)
    fb_move = str(fb_move)
    if lrud_move.lower() == 'random':
        lrud_move = random.choice(
            ['linear', 'cos', 'rlinear', 'rcos', 'random', 'none'])
    if fb_move.lower() == 'random':
        fb_move = random.choice(
            ['linear', 'cos', 'rlinear', 'rcos', 'random', 'none'])

    if fb_move.lower() == 'linear':
        start, end = np.sort(rand(2))
        fb_move = np.linspace(start, end, max_frame) * max_fb_move
    elif fb_move.lower() == 'cos':
        start, end = np.sort(rand(2, 4 * np.pi))
        fb_move = np.cos(np.linspace(start, end, max_frame)) * max_fb_move
    elif fb_move.lower() == 'rlinear':
        start, end = np.sort(rand(2))
        fb_move = (np.sort(np.random.uniform(start, end, max_frame)) +
                   np.random.normal(0, 0.5, max_frame) * 0.2) * max_fb_move
    elif fb_move.lower() == 'rcos':
        start, end = np.sort(rand(2, 4 * np.pi))
        fb_move = (np.cos(np.sort(np.random.uniform(start, end, max_frame))) +
                   np.random.normal(0, 0.5, max_frame) * 0.2) * max_fb_move
    elif fb_move.lower() == 'random':
        fb_move = np.random.normal(0, 0.5, max_frame) * max_fb_move
    else:
        fb_move = None

    if lrud_move.lower() == 'linear':
        start, end = np.sort(rand((2, 2)), axis=0)
        lrud_move = np.stack(
            (np.linspace(start[0], end[0],
                         max_frame), np.linspace(start[1], end[1], max_frame)),
            axis=1) * max_lrud_move
    elif lrud_move.lower() == 'cos':
        start, end = np.sort(rand((2, 2), 4 * np.pi), axis=0)
        lrud_move = np.cos(
            np.stack((np.linspace(start[0], end[0], max_frame),
                      np.linspace(start[1], end[1], max_frame)),
                     axis=1)) * max_lrud_move
    elif lrud_move.lower() == 'rlinear':
        start, end = np.sort(rand((2, 2)), axis=0)
        lrud_move = (np.stack(
            (np.sort(np.random.uniform(start[0], end[0], max_frame)),
             np.sort(np.random.uniform(start[1], end[1], max_frame))),
            axis=1) + np.random.normal(0, 0.5,
                                       (max_frame, 2)) * 0.2) * max_lrud_move
    elif lrud_move.lower() == 'rcos':
        start, end = np.sort(rand((2, 2), 4 * np.pi), axis=0)
        lrud_move = (np.cos(
            np.stack((np.sort(np.random.uniform(start[0], end[0], max_frame)),
                      np.sort(np.random.uniform(start[1], end[1], max_frame))),
                     axis=1)) +
                     np.random.normal(0, 0.5,
                                      (max_frame, 2)) * 0.2) * max_lrud_move
    elif lrud_move.lower() == 'random':
        lrud_move = np.random.normal(0, 0.5, (max_frame, 2)) * max_lrud_move
    else:
        lrud_move = None
    if lrud_move is not None:
        lrud_move = lrud_move
    return fb_move, lrud_move


# #######################################################################
class VirtualCamera(BaseCamera):
    def __init__(self,
                 seq,
                 mode='norm',
                 max_frame=30,
                 min_ele=0,
                 max_ele=530,
                 move_type=(None, None),
                 is_save_result=True,
                 is_save_img=True,
                 is_display=False,
                 is_display_info=True) -> None:
        super().__init__(min_ele, max_ele, is_save_result, is_save_img,
                         is_display, is_display_info)
        self.seq = seq
        self.mode = mode
        self.dis_mode = mode
        self.max_frame = max_frame

        self.basepath = 'dataset/HutIris-Blur/png'
        self.len_range = np.array([int(x[0].split('_')[2]) for x in seq])
        self.min_ele = self.len_range.min()
        self.max_ele = self.len_range.max()

        self.seq = {int(x[0].split('_')[2]): x for x in seq}

        self.move_type = move_type
        self.fb_move, self.lrud_move = init_move(move_type[0],
                                                 move_type[1],
                                                 max_frame=max_frame)

        self._config()

    def _find_side(self, target):
        idx = np.abs(self.len_range - int(target)).argmin()
        p1 = self.len_range[idx]
        if p1 > target and idx - 1 > 0:
            p2 = self.len_range[idx - 1]
        elif p1 < target and idx + 1 < len(self.len_range):
            p2 = self.len_range[idx + 1]
        else:
            p2 = p1
        return p1, p2

    def _getimg(self, pos=None, loc=True):
        # fb move
        if self.fb_move is not None:
            offset_z = self.fb_move[self.frame_num % self.max_frame]
            offset_z = int(offset_z * (self.max_ele - self.min_ele))
            pos += offset_z

        if pos in self.len_range:
            img, roi = self._load_img(self.seq[pos])
        else:
            pos1, pos2 = self._find_side(pos)
            if np.abs(pos1 - pos2) < 1:
                img, roi = self._load_img(self.seq[pos1])
            else:
                img1, roi1 = self._load_img(self.seq[pos1])
                img2, roi2 = self._load_img(self.seq[pos2])
                alpha1 = np.abs(pos1 - pos) / np.abs(pos1 - pos2)
                alpha2 = np.abs(pos2 - pos) / np.abs(pos1 - pos2)
                img = alpha1 * img1.astype(np.double) + alpha2 * img2.astype(
                    np.double)
                img = np.clip(np.round(img), 0, 255).astype(np.uint8)
                roi = np.round(alpha1 * np.array(roi1) +
                               alpha2 * np.array(roi2)).astype(np.int)
                assert np.all(roi > 0)

        # lrud move
        if self.lrud_move is not None:
            offset_x, offset_y = self.lrud_move[self.frame_num %
                                                self.max_frame]
            offset_x = int(offset_x * img.shape[1])
            offset_y = int(offset_y * img.shape[0])
            img = img_lrud_move(img, offset_x, offset_y)
            roi = roi - np.array([offset_x, offset_y, 0])

        return img, roi.tolist()

    def _load_img(self, info, resize=True):
        _path = osp.join(self.basepath, info[0] + '.png')
        img = cv2.imread(_path, cv2.IMREAD_GRAYSCALE)
        roi = np.array([
            [
                int((info[-1][0])),
                int((info[-1][1])),
                int((info[-1][2])),
            ],
            [
                int((info[-1][3])),
                int((info[-1][4])),
                int((info[-1][5])),
            ],
        ])
        if resize:
            img = cv2.resize(img, None, fx=0.125, fy=0.125)
            roi = np.round(roi * 0.125).astype(np.int)
        return img, roi

    def _loop(self):
        if self.mode == 'fib':
            self.fibonacci_search()
            while self.frame_num < self.max_frame:
                self.get_img()
                if self.key == ord('q'):
                    break
        elif self.mode == 'fast':
            self.fast_tracking(self.max_frame)
        else:
            if self.is_display:
                if self.move_type[0] is not None or self.move_type[
                        1] is not None:
                    while self.frame_num < self.max_frame:
                        self.get_img()
                        if self.key == ord('q'):
                            break
                else:
                    while self.cur_ele < self.max_ele:
                        self.get_img(self.cur_ele)
                        self.cur_ele += 1
                        if self.key == ord('q'):
                            break
        if self.is_display:
            cv2.destroyAllWindows()
        return 0

    def _final(self):
        if self.is_save_result:
            img_num = len(self.saved_result['img'])
            ele = np.array(self.saved_result['ele'])
            ele = (ele - ele.min()) / (ele.max() - ele.min())
            self.saved_result['ele_norm'] = ele

            if len(self.saved_result['gt_roi']) > 0:
                roi = self.saved_result['gt_roi']
            elif len(self.saved_result['roi']) > 0:
                roi = self.saved_result['roi']
            else:
                roi = [None for _ in range(img_num)]
            fm = [
                fmeasure(self.saved_result['img'][idx], roi[idx])
                for idx in range(img_num)
            ]
            self.saved_result['focus'] = np.array(fm)
            self.saved_result['seq'] = self.seq
            self.saved_result['move'] = (self.fb_move, self.lrud_move)
            if not self.is_save_img:
                del self.saved_result['img']


# #######################################################################
def debug():
    with open('dataset/HutIris-Blur/meta_info.pkl', "rb") as f:
        meta_info = pickle.load(f)
    img_list = meta_info['label2']
    label_list = meta_info['split'][2]
    seq_list = get_sequence_list(img_list, label_list)

    cam = VirtualCamera(seq_list[5],
                        max_frame=30,
                        mode='fast',
                        move_type=(None, None),
                        is_save_img=False,
                        is_save_result=True)
    cam.run()


def test():
    with open('dataset/HutIris-Blur/meta_info.pkl', "rb") as f:
        meta_info = pickle.load(f)
    img_list = meta_info['label2']
    label_list = meta_info['split'][2]
    seq_list = get_sequence_list(img_list, label_list)

    print('1/4 fib stay')
    result = []
    # for seq in tqdm(seq_list[:-5]):
    # cam = VirtualCamera(
    #     seq,
    #     max_frame=30,
    #     mode='fib',
    #     move_type=(None, None),
    #     is_save_img=False,
    #     is_save_result=True,
    #     is_display=False,
    # )
    # cam.run()
    # result.append(cam.saved_result)
    for seq in tqdm(seq_list[-5:]):
        cam = VirtualCamera(
            seq,
            max_frame=30,
            mode='fib',
            move_type=(None, None),
            is_save_img=True,
            is_save_result=True,
            is_display=False,
        )
        cam.run()
        result.append(cam.saved_result)
    torch.save(result, 'Camera/result/fibstay.pth')

    print('2/4 fib move')
    result = []
    # for seq in tqdm(seq_list[:-5]):
    #     cam = VirtualCamera(
    #         seq,
    #         max_frame=30,
    #         mode='fib',
    #         move_type=('cos', 'cos'),
    #         is_save_img=False,
    #         is_save_result=True,
    #         is_display=False,
    #     )
    #     cam.run()
    #     result.append(cam.saved_result)
    for seq in tqdm(seq_list[-5:]):
        cam = VirtualCamera(
            seq,
            max_frame=30,
            mode='fib',
            move_type=('cos', 'cos'),
            is_save_img=True,
            is_save_result=True,
            is_display=False,
        )
        cam.run()
        result.append(cam.saved_result)
    torch.save(result, 'Camera/result/fibmove.pth')

    print('3/4 fast stay')
    result = []
    # for seq in tqdm(seq_list[:-5]):
    #     cam = VirtualCamera(
    #         seq,
    #         max_frame=30,
    #         mode='fast',
    #         move_type=(None, None),
    #         is_save_img=False,
    #         is_save_result=True,
    #         is_display=False,
    #     )
    #     cam.run()
    #     result.append(cam.saved_result)
    for seq in tqdm(seq_list[-5:]):
        cam = VirtualCamera(
            seq,
            max_frame=30,
            mode='fast',
            move_type=(None, None),
            is_save_img=True,
            is_save_result=True,
            is_display=False,
        )
        cam.run()
        result.append(cam.saved_result)
    torch.save(result, 'Camera/result/faststay.pth')

    print('4/4 fast move')
    result = []
    # for seq in tqdm(seq_list[:-5]):
    #     cam = VirtualCamera(
    #         seq,
    #         max_frame=30,
    #         mode='fast',
    #         move_type=('cos', 'cos'),
    #         is_save_img=False,
    #         is_save_result=True,
    #         is_display=False,
    #     )
    #     cam.run()
    #     result.append(cam.saved_result)
    for seq in tqdm(seq_list[-5:]):
        cam = VirtualCamera(
            seq,
            max_frame=30,
            mode='fib',
            move_type=('cos', 'cos'),
            is_save_img=True,
            is_save_result=True,
            is_display=False,
        )
        cam.run()
        result.append(cam.saved_result)
    torch.save(result, 'Camera/result/fastmove.pth')


def show():
    with open('dataset/HutIris-Blur/meta_info.pkl', "rb") as f:
        meta_info = pickle.load(f)
    img_list = meta_info['label2']
    label_list = meta_info['split'][2]
    seq_list = get_sequence_list(img_list, label_list)
    seq_list = np.random.choice(seq_list, 5)

    for seq in tqdm(seq_list):
        cam = VirtualCamera(
            seq,
            max_frame=np.random.randint(45, 75),
            mode='fast',
            move_type=('none', 'none'),
            is_save_img=True,
            is_save_result=True,
            is_display=True,
        )
        cam.run()
        result = cam.saved_result
        dst = 'D:/Code/FocusTracker/Camera/result/fast_stay_{}'.format(
            int(time.time()))
        os.makedirs(dst)
        for idx in range(len(result['ele'])):
            frame = result['img'][idx]
            text = 'frame:{} fm:{} ele:{}'.format(idx,
                                                  int(result['focus'][idx]),
                                                  int(result['ele'][idx]))
            cv2.putText(frame, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite('{}/{:0>2d}.png'.format(dst, idx), frame)


# #######################################################################

if __name__ == "__main__":
    show()