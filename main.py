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

roiMingeFolderLocation = 'roi/cana_mica_extra_rau.jpg'
roiCanaFolderLocation = 'roi/cana.jpg'

roiMingeImageData = cv2.imread(roiMingeFolderLocation, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
heightMinge, widthMinge= roiMingeImageData.shape
roiCanaImageData = cv2.imread(roiCanaFolderLocation).astype(np.uint8)
heightCana, widthCana, layersCana = roiCanaImageData.shape

frame = cv2.imread('roi/cana_mica_rau.jpg', cv2.IMREAD_GRAYSCALE).astype(np.uint8)
heightFrame, widthFrame = frame.shape

mx = heightFrame - heightMinge
nx = widthFrame - widthMinge


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
            if (gid == 0) {
                 printf (" (%d, %d) ", frameData[(i*m)+j], roiData[(i-gidI)*roiHeight+j-gidJ]);
                printf (" dif: %d ", dif);
            }
        }
    }
    if (gid == 0) {
        printf (" sum: %d ", sum);
    }
    resData[gid] = (float)sum/(roiHeight*roiWidth);
}
""").build()
#TODO fix the data transmition
for image in localTest:
    # print(roiMingeImageData)
    frameData = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frame)
    roiData = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=roiMingeImageData)
    # b_m = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(heightMinge))
    # b_n = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(widthMinge))
    # m = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(mx))
    # n = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(nx))

    dim = 6 * 6
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


# Check on CPU with Numpy:
#print(res_np)
for image in localTest:
    cpuArray = []
    frame = cv2.imread('roi/cana_mica_rau.jpg', cv2.IMREAD_GRAYSCALE).astype(np.int16)
    heightFrame, widthFrame = frame.shape
    #print(frame)
    roiMingeImageData = cv2.imread(roiMingeFolderLocation, cv2.IMREAD_GRAYSCALE).astype(np.int16)
    heightMinge, widthMinge = roiMingeImageData.shape
    for i in range(0, mx+1):
        for j in range(0, nx+1):
            localM1 = frame[i:i+widthMinge, j:j+heightMinge]
            diff = localM1 - roiMingeImageData
            # mse = getMSE(localM1, roiMingeImageData)
            localSum = np.sum(np.square(diff))
            cpuArray.append(localSum/(heightMinge*widthMinge))

print(res_np)
print(cpuArray)
print(res_np - cpuArray)
print(np.linalg.norm(res_np - cpuArray))
# assert np.allclose(res_np, cpuArray)
