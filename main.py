#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import os
import cv2
from natsort import natsorted


def getMSE(m1, m2):
    return np.square(np.subtract(m1, m2)).mean()


videoFolder = './video/football.avi'
videocap = cv2.VideoCapture(videoFolder)
success, image = videocap.read()
count = 0
while success:
    cv2.imwrite("./frames/fotbal/frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = videocap.read()
    count += 1

image_folder = './frames/fotbal'
video_name = 'video.avi'

fpsVideo = 25
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images = natsorted(images)

roiMingeFolderLocation = 'roi/cana.jpg'
roiCanaFolderLocation = 'roi/cana.jpg'

roiCanaImageData = cv2.imread(roiCanaFolderLocation).astype(np.uint8)
heightCana, widthCana, layersCana = roiCanaImageData.shape

globalFrame = cv2.imread('frames/fotbal/frame0.jpg', cv2.IMREAD_GRAYSCALE).astype(np.uint8)
globalheightFrame, globalwidthFrame = globalFrame.shape
print(globalheightFrame, globalwidthFrame)


# video = cv2.VideoWriter(video_name, 0, fpsVideo, (widthFrame, heightFrame))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
#
# cv2.destroyAllWindows()
# video.release()

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
localTest = [images[0]]

prg = cl.Program(ctx, """
__kernel void calculateMSE(
    __global const uchar *frameData, 
    __global const uchar *roiData, 
     int roiHeight, 
     int roiWidth, 
     int m,
     int n, 
    __global float *resData)
{
    int gid = get_global_id(0);
    int mx = m - roiHeight +1;
    int nx = n - roiWidth + 1;
    int gidI = gid / mx;
    int gidJ = gid % nx;
    int sum = 0;
    for (int i = gidI; i < gidI + roiHeight; i++) {
        for (int j = gidJ; j < gidJ + roiWidth; j++) {
            int dif = frameData[(i*m)+j] - roiData[(i-gidI)*roiHeight+j-gidJ];
            sum += dif*dif;
        }
    }
    resData[gid] = (float)sum/(roiHeight*roiWidth);
}
""").build()
#TODO made if to choose between cup and ball
stackResult = []
for image in images:
    roiMingeImageData = cv2.imread(roiMingeFolderLocation, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    heightMinge, widthMinge = roiMingeImageData.shape
    currentFrameFolderLocation = os.path.join(image_folder, image)
    currentFrame = cv2.imread(currentFrameFolderLocation, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    heightFrame, widthFrame = currentFrame.shape

    mx = heightFrame - heightMinge
    nx = widthFrame - widthMinge

    frameData = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=currentFrame)
    roiData = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=roiMingeImageData)

    dim = mx * nx
    fakeVector = np.zeros(dim).astype(np.float32)
    resData = cl.Buffer(ctx, mf.WRITE_ONLY, fakeVector.nbytes)
    knl = prg.calculateMSE  # Use this Kernel object for repeated calls

    #convert data
    heightData32 = np.int32(heightMinge)
    widthData32 = np.int32(widthMinge)
    mData32 = np.int32(heightFrame)
    nData32 = np.int32(widthFrame)

    knl(queue, (dim, 1), None, frameData, roiData, heightData32, widthData32, mData32, nData32, resData)

    res_np = np.empty_like(fakeVector)
    cl.enqueue_copy(queue, res_np, resData)
    stackResult.append(res_np)


def getMSEMin(stackResult):
    array = []
    for arr in stackResult:
        min = arr[0]
        for (index, mse) in enumerate(arr):
            local = mse
            if local < min:
                min = local
                i = index
        array.append((min, i))
    return array


def getCoordinates(index, frameHeight, frameWidth):
    x = index / frameHeight
    y = index % frameWidth
    return (round(x), round(y))

minArray = getMSEMin(stackResult)
for tuple in minArray:
    print(getCoordinates(tuple[1], globalheightFrame, globalwidthFrame))


# for testing
# Check on CPU with Numpy:
# for image in localTest:
#     cpuArray = []
#     frame = cv2.imread('roi/cana_mica_rau.jpg', cv2.IMREAD_GRAYSCALE).astype(np.int16)
#     heightFrame, widthFrame = frame.shape
#     #print(frame)
#     roiMingeImageData = cv2.imread(roiMingeFolderLocation, cv2.IMREAD_GRAYSCALE).astype(np.int16)
#     heightMinge, widthMinge = roiMingeImageData.shape
#     for i in range(0, mx+1):
#         for j in range(0, nx+1):
#             localM1 = frame[i:i+widthMinge, j:j+heightMinge]
#             diff = localM1 - roiMingeImageData
#             # mse = getMSE(localM1, roiMingeImageData)
#             localSum = np.sum(np.square(diff))
#             cpuArray.append(localSum/(heightMinge*widthMinge))
#
# print(res_np)
# print(cpuArray)
# print(res_np - cpuArray)
# print(np.linalg.norm(res_np - cpuArray))
# assert np.allclose(res_np, cpuArray)
