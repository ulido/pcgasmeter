from PIL import Image
from io import BytesIO
import numpy as np

from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.filters import threshold_local
from skimage.morphology import medial_axis

import cv2

class Dial:
    def __init__(self):
        self.theta = np.nan
        self.value = 0.0
        self.last_value = None
        self.conversion_ft3 = 0.071 # ft^3 per rotation
    
    @staticmethod
    def _find_dial_angle(image):
        img = Image.open(BytesIO(image))
        d = np.array(img)
        g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        T = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 0)
        cnts = cv2.findContours(T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        idx = np.argmin([cv2.arcLength(c, True)**2/cv2.contourArea(c) if cv2.contourArea(c) > 10000 else np.inf for c in cnts])
        x, y, w, h = cv2.boundingRect(cnts[idx])
        dial = 1.0*d[y+h//6:y+(5*h)//6, x+(17*w)//24:x+w]
        rdial = np.clip(dial[:, :, 0] - 0.5*(dial[:, :, 1] + dial[:, :, 2]), 0, 255).astype(np.uint8)
        rdial = cv2.threshold(rdial, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        skel = medial_axis(rdial)

        px, py = np.argwhere(skel).T[::-1]
        x = np.arange(skel.shape[1])
        p = np.polyfit(px, py, 1)
        theta = np.arctan(p[0])
        mx = np.mean(px) - skel.shape[1]//2
        my = np.mean(py) - skel.shape[0]//2
        if np.sign(mx*my) != np.sign(p[0]):
            return np.nan
        if mx > 0:
            return theta + np.pi
        elif (mx < 0) and (my >= 0):
            return theta + 2*np.pi
        elif (mx < 0) and (my < 0):
            return theta
        elif (mx == 0) and (my > 0):
            return np.pi
        elif (mx == 0) and (my < 0):
            return 0
        return np.nan
    
    def _read_angle(self, image):
        candidate = self._find_dial_angle(image)
        if np.isnan(candidate):
            return self.theta
        if np.isnan(self.theta):
            self.theta = candidate
            return self.theta
        
        dt = candidate - self.theta
        # Avoid jitter by not allowing small angle decreases
        if (dt < -np.pi) or (dt > 0):
            self.theta = candidate
        return self.theta
    
    def convert(self):
        return (self.value * self.conversion_ft3) / (2*np.pi)
    
    def __call__(self, image):
        candidate = self._read_angle(image)
        if np.isnan(candidate):
            return self.convert()
        
        if self.last_value is None:
            self.last_value = candidate
            return self.convert()
        
        dv = candidate - self.last_value
        if (dv < -np.pi):
            self.last_value = candidate
            self.value += 2*np.pi+dv
        elif dv > 0:
            self.last_value = candidate
            self.value += dv
        return self.convert()