import torch
from PIL import Image
import time

import numpy as np
import cv2 as cv

from torchvision import transforms
from models import FlowNet2
from utils import flow_utils

model = FlowNet2().cpu()
model.morphize()
#model.load_state_dict(torch.load('checkpoint/FlowNet2_checkpoint.pth.tar')['state_dict'])
#model.load_state_dict(torch.load('checkpoint/checkpoint3.pth.tar')['state_dict'])
#model.load_state_dict(torch.load('checkpoint/checkpoint8.pth.tar')['state_dict'])
model.load_state_dict(torch.load('checkpoint/checkpoint9.pth.tar')['state_dict'])
model.demorphize()
model.eval()
model.cuda()

def padding(image):
    _, w, h = image.size()
    w_pad = (64 - w % 64) % 64
    h_pad = (64 - h % 64) % 64
    new_image = torch.zeros(3, w + w_pad, h + h_pad)
    new_image[:, w_pad // 2 : w_pad // 2 + w, h_pad // 2 : h_pad // 2 + h] = image
    return new_image

prev_image = None
cap = cv.VideoCapture(0)
# Define the codec and create VideoWriter object
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    curr_image = padding(transforms.Compose([transforms.ToPILImage(), transforms.Resize((120, 160)), transforms.ToTensor()])(frame)).unsqueeze(1).unsqueeze(0)
    if prev_image == None:
        prev_image = curr_image
    inputs = torch.cat([prev_image, curr_image], dim=2).cuda()
    start = time.time()
    with torch.no_grad():
        output = model(inputs).cpu().numpy()
    end = time.time()
    result = flow_utils.flow2img(output[0].transpose(1, 2, 0))
    prev_image = curr_image
#    print (1 / (end - start))

    # write the flipped frame
    cv.imshow('origin', frame)
    cv.imshow('result', result)
    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
cv.destroyAllWindows()
