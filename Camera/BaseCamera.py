import warnings
warnings.filterwarnings('ignore')

import cv2
import time

import torch
import numpy as np

from model.network import UNetLike

# #######################################################################


def load_checkpoint(model, cp_path):
    if cp_path:
        cp_data = torch.load(cp_path, map_location=torch.device('cpu'))
        try:
            model.load_state_dict(cp_data['model'])
        except Exception as e:
            model.load_state_dict(cp_data['model'], strict=False)
            print(e)
    return model


def make_fibonacci(N):
    assert N > 2

    fibonacci_list = [0, 1]
    i = 2
    while fibonacci_list[-1] < N:
        fib = fibonacci_list[i - 1] + fibonacci_list[i - 2]
        fibonacci_list.append(fib)
        i += 1
    return fibonacci_list


def generate_mask(roi, size):
    if isinstance(roi, list):
        mask = np.zeros(size).astype(np.double)
        for r in roi:
            r = np.round(r).astype(np.int)
            mask = cv2.circle(mask, (r[0], r[1]), r[2], (1, 1, 1), -1)
    elif isinstance(roi, (np.ndarray, np.generic)):
        mask = roi
    else:
        mask = np.ones(size).astype(np.double)
    return mask


def fmeasure(img, roi=None):
    """
    Implements the Tenengrad (TENG) focus measure operator.
    Based on the gradient of the image.
    """
    img = img.astype(np.double)
    mask = generate_mask(roi, img.shape)

    gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gaussianY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    fm = ((gaussianX**2 + gaussianY**2) * mask).sum() / mask.sum()
    return fm


# #######################################################################


class BaseCamera(object):
    def __init__(self,
                 min_ele=530,
                 max_ele=0,
                 is_save_result=True,
                 is_save_img=False,
                 is_display=False,
                 is_display_info=True) -> None:
        super().__init__()

        self.min_ele = min_ele
        self.max_ele = max_ele

        self.frame_num = 0

        self.is_save_result = is_save_result
        self.is_save_img = is_save_img
        self.is_display = is_display
        self.is_display_info = is_display_info

    def _config(self):
        self.key = -1
        self.dis_mode = 'norm'
        self.model = None
        self.cur_ele = np.random.randint(self.min_ele, self.max_ele)
        if self.is_save_result:
            self.saved_result = {
                'ele': [],
                'gt_roi': [],
                'img': [],
                'roi': [],
            }
        if self.is_display:
            self.timer = time.time()
            self.cur_frame = None
            self.cur_gt_roi = None
            self.cur_roi = None
            cv2.namedWindow('AquulaCam', cv2.WINDOW_AUTOSIZE)

    def run(self):
        self._loop()
        self._final()

    def _init_deep_model(self):
        cp_path = "checkpoints/0308_160732_UNetLike/90_1.8029e-01.pth"
        self.device = 'cuda:0'

        self.model = UNetLike(dct=False, channel=16, use_lstm=False)
        cp_data = torch.load(cp_path, map_location=torch.device('cpu'))
        try:
            self.model.load_state_dict(cp_data['model'])
        except Exception as e:
            self.model.load_state_dict(cp_data['model'], strict=False)
            print(e)
        self.model.to(device=self.device)
        self.model.eval()
        return self.model

    def _display(self):
        if self.cur_gt_roi is not None:
            gt = generate_mask(self.cur_gt_roi,
                               self.cur_frame.shape) * 255 + self.cur_frame
        else:
            gt = self.cur_frame
        if self.cur_roi is not None:
            pred = generate_mask(self.cur_roi,
                                 self.cur_frame.shape) * 255 + self.cur_frame
        else:
            pred = self.cur_frame
        frame = np.stack((self.cur_frame, gt, pred), -1)
        frame = np.clip(np.round(frame), 0, 255).astype(np.uint8)

        if self.is_display_info:
            fps = 1 / (time.time() - self.timer)
            self.timer = time.time()
            if self.cur_roi is not None:
                fm = fmeasure(self.cur_frame, self.cur_roi)
            elif self.cur_gt_roi is not None:
                fm = fmeasure(self.cur_frame, self.cur_gt_roi)
            else:
                fm = fmeasure(self.cur_frame)
            info = 'ele:{} focus:{}, fps:{}'.format(int(self.cur_ele), int(fm),
                                                    int(fps))
            cv2.putText(frame, info, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('AquulaCam', frame)
        self.key = cv2.waitKey(1)
        return self.key

    def _loop(self):
        raise NotImplementedError

    def _final(self):
        raise NotImplementedError

    def _getimg(self, pos=None, loc=True):
        raise NotImplementedError

    def get_img(self, pos=None, loc=True):
        if pos is None:
            pos = self.cur_ele
        if pos != self.cur_ele:
            self.cur_ele = pos
        img, roi = self._getimg(pos, loc)
        if self.is_save_result:
            self.saved_result['img'].append(img)
            self.saved_result['ele'].append(pos)
            self.saved_result['gt_roi'].append(roi)
        if self.is_display:
            self.cur_frame = img
            self.cur_gt_roi = roi
            if self.dis_mode != 'fast':
                self._display()
        self.frame_num += 1
        return img, roi

    def fibonacci_search(self, a=None, b=None, thr=0.5):
        self.dis_mode = 'fib'
        if a is None or b is None:
            a, b = int(self.min_ele), int(self.max_ele)

        fibonacci_list = make_fibonacci(b - a)

        N = len(fibonacci_list)
        x, y = a, b
        dst_x, roi_x = self.get_img(x)
        dst_y, roi_y = self.get_img(y)
        fx = fmeasure(dst_x, roi_x)
        fy = fmeasure(dst_y, roi_y)
        error = abs(fx - fy)

        for i in range(N - 2):
            L = y - x
            if error < thr:
                i = i - 1
                break

            x1 = int(x +
                     (fibonacci_list[N - i - 3] / fibonacci_list[N - i - 1]) *
                     L)
            y1 = int(y -
                     (fibonacci_list[N - i - 3] / fibonacci_list[N - i - 1]) *
                     L)
            if x1 == x:
                x1 = x
                fx1 = fx
                dst_x1 = dst_x
                roi_x1 = roi_x
            else:
                dst_x1, roi_x1 = self.get_img(x1)
                fx1 = fmeasure(dst_x1, roi_x1)
            if y1 == y:
                y1 = y
                fy1 = fy
                dst_y1 = dst_y
                roi_y1 = roi_y
            else:
                dst_y1, roi_y1 = self.get_img(y1)
                fy1 = fmeasure(dst_y1, roi_y1)

            if fx1 < fy1:
                x = x1
                fx = fx1
                dst_x = dst_x1
                roi_x = roi_x1

            else:
                y = y1
                fy = fy1
                dst_y = dst_y1
                roi_y = roi_y1

            error = abs(fx - fy)

        if fx > fy:
            ele = x
        else:
            ele = y

        dst, roi = self.get_img(ele)
        fm = fmeasure(dst, roi)
        return ele, fm, i + 2

    def fast_tracking(self, max=2e4):
        self.dis_mode = 'fast'
        if self.model is None:
            self._init_deep_model()

        step = 0
        hidden = None
        for _ in range(max):
            img, _ = self.get_img(self.cur_ele + step, True)
            img = torch.from_numpy(img) / 255.0
            img = img.unsqueeze(0).unsqueeze(0).to().to(self.device)
            with torch.no_grad():
                step, roi, hidden = self.model(img, hidden)
            step = (step.cpu().item() * 530)
            roi = torch.softmax(roi[0], dim=0)[1, :, :].cpu().numpy()
            if self.is_save_result:
                self.saved_result['roi'].append(roi)
            if self.is_display:
                self.cur_roi = roi
                self._display()
