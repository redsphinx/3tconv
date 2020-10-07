# This is a toy dataset with 3 classes so that we can validate 3TConv
import numpy as np
import os
from PIL import Image
import time
from datetime import datetime
import skvideo.io as skvid
import cv2 as cv


'''
3 classes

sinusoid on 2D plane, moving through time

class 1: translation
class 2: rotation
class 3: scaling

for each class there are variations
- band density (frequency)
- orientation
- initial scale (how thick/thin bands are)
- speed at which the transformation takes place
- direction in which the transformation takes place




class 1: 4 directions, left right up down
class 2: clockwise, counterclockwise, point of rotation
class 3: scale up or down, origin of scaling


'''