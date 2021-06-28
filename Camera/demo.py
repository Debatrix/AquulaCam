import cv2
import time
import serial
import mvsdk
import torch

import numpy as np

from Camera.BaseCamera import BaseCamera


class optoLens(object):
    def __init__(self, port='COM7'):
        self.port = port
        self.Ic = 293.0
        self.table = []
        self.max_mA = 100  #292
        self.min_mA = -100  #-249.5
        self.ele = 0

        self._init_table()
        self.ser = None
        self._open()

    def _open(self):
        self.ser = serial.Serial(self.port, 115200, timeout=None)

    def _close(self):
        assert self.ser is not None
        self.ser.close()

    def _init_table(self):
        self.table = []
        for i in range(256):
            value = 0
            temp = i
            for j in range(8):
                if (((value ^ temp) & 0x0001) != 0):
                    value = (value >> 1) ^ 0xA001
                else:
                    value = value >> 1
                temp = temp >> 1
            self.table.append(value)

    def _computerCRC(self, data):
        crc = 0
        for i in data:
            index = (crc ^ i) % 2**8
            crc = ((crc >> 8) ^ self.table[index]) % 2**16
        return crc

    def elec2cmd(self, electricity):
        xi = round((electricity / 293.0) * 4096)

        # 负数转化仅适用于本例
        xi = 65535 + xi if xi < 0 else xi
        command = b'\x41\x77' + xi.to_bytes(2, 'big')
        crc = self._computerCRC(command)
        command = command + ((crc & 0xFF) % 256).to_bytes(1, 'big') + (
            (crc >> 8) % 256).to_bytes(1, 'big')
        assert self._computerCRC(command) == 0
        return command

    def triangular(self, low_elec, high_elec, frequency=50, step=1):
        assert self.ser is not None
        for i in range(low_elec, high_elec, step):
            self.ser.write(self.elec2cmd(i))
            time.sleep(1 / frequency)

    def send_cmd(self, elec=0):
        assert self.ser is not None
        if self.min_mA <= elec <= self.max_mA:
            if self.ele != elec:
                self.ser.write(self.elec2cmd(elec))
                self.ele = elec
        else:
            if self.min_mA > elec:
                print('{:.2f} is smaller than min_mA'.format(elec))
            else:
                print('{:.2f} is bigger than max_mA'.format(elec))
        return self.ele


class DemoCamera(BaseCamera):
    def __init__(self, is_save_result, is_save_img, is_display,
                 is_display_info) -> None:
        super().__init__(min_ele=-240,
                         max_ele=290,
                         is_save_result=is_save_result,
                         is_save_img=is_save_img,
                         is_display=is_display,
                         is_display_info=is_display_info)

        self.opto = optoLens('COM5')
        self.min_ele = self.opto.min_mA
        self.max_ele = self.opto.max_mA

        self.camera = None
        self.pFrameBuffer = None
        self.exposure_time = 50
        self.gain = 1

        self.vis_cam = cv2.VideoCapture(0)

        self.dis_mode = 'norm'

        self.keylist = [ord(x) for x in ['n','q', 't', 'd', 'f']]

        self.InitCamera()
        self._config()
        self.cur_ele = 20

    def InitCamera(self):
        try:
            DevList = mvsdk.CameraEnumerateDevice()
            nDev = len(DevList)
            if nDev < 1:
                raise Exception("No camera was found!")

            DevInfo = DevList[0]

            self.camera = mvsdk.CameraInit(DevInfo, -1, -1)
            cap = mvsdk.CameraGetCapability(self.camera)

            self.monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
            if self.monoCamera:
                mvsdk.CameraSetIspOutFormat(self.camera,
                                            mvsdk.CAMERA_MEDIA_TYPE_MONO8)
            else:
                mvsdk.CameraSetIspOutFormat(self.camera,
                                            mvsdk.CAMERA_MEDIA_TYPE_RGB8)
            mvsdk.CameraSetTriggerMode(self.camera, 0)
            mvsdk.CameraSetAeState(self.camera, False)

            mvsdk.CameraSetExposureTime(self.camera, self.exposure_time * 1000)
            mvsdk.CameraSetAnalogGain(self.camera, self.gain)

            FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (
                1 if self.monoCamera else 3)
            self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

            mvsdk.CameraPlay(self.camera)

        except mvsdk.CameraException as e:
            raise Exception("[{}]: CameraInit Failed({}): {}".format(
                self._now, e.error_code, e.message))
        except Exception as e:
            raise Exception(e)

    def _getimg(self, pos, loc=False):
        try:
            self.cur_ele = self.opto.send_cmd(int(pos))
        except Exception as e:
            print(e)
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.camera, 200)
            mvsdk.CameraImageProcess(self.camera, pRawData, self.pFrameBuffer,
                                     FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.camera, pRawData)

            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(
                self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                                   1 if FrameHead.uiMediaType
                                   == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            frame = cv2.resize(frame, None, fx=0.125, fy=0.125)
            frame = np.flip(frame, 0)
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(
                    e.error_code, e.message))
        return frame, None

    def fast_tracking(self):
        self.dis_mode = 'fast'
        if self.model is None:
            self._init_deep_model()

        step = 0
        hidden = None
        while True:
            img, _ = self.get_img(self.cur_ele + step, True)
            img = torch.from_numpy(img.copy()) / 255.0
            img = img.unsqueeze(0).unsqueeze(0).to().to(self.device)
            with torch.no_grad():
                step, roi, hidden = self.model(img, hidden)
            step = (step.cpu().item() * 530)
            roi = torch.softmax(roi[0], dim=0)[1, :, :].cpu().numpy()
            if self.is_save_result:
                self.saved_result['roi'].append(roi)
            self.cur_roi = roi
            self._display()
            if self.key in self.keylist:
                break

    def _show(self):
        while True:
            self.get_img()
            if self.key in self.keylist:
                break

    def _loop(self):
        while self.key != ord('q'):
            if self.key == ord('f'):
                self.fibonacci_search()
            elif self.key == ord('d'):
                self.fast_tracking()
            elif self.key == ord('t'):
                self.cur_ele = self.min_ele
                while self.cur_ele < self.max_ele:
                    self.get_img(self.cur_ele)
                    self.cur_ele += 1
                    if self.key == ord('q'):
                        break
                self.cur_ele = 0
                self.get_img(self.cur_ele)
            else:
                self._show()

        cv2.destroyAllWindows()
        return 0


def camtest(cam=0):
    cap = cv2.VideoCapture(cam)
    while (True):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # camtest(1)
    cam = DemoCamera(is_display=True,
                     is_save_img=False,
                     is_save_result=False,
                     is_display_info=True)
    cam._loop()
    pass