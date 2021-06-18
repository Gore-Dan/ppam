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

roiMingeImageData = cv2.imread(roiMingeFolderLocation, cv2.IMREAD_GRAYSCALE)
heightMinge, widthMinge= roiMingeImageData.shape
roiCanaImageData = cv2.imread(roiCanaFolderLocation).astype(np.int32)
heightCana, widthCana, layersCana = roiCanaImageData.shape

frame = cv2.imread('roi/cana_mica_rau.jpg', cv2.IMREAD_GRAYSCALE)
heightFrame, widthFrame = frame.shape
print(type(np.int32(heightFrame)))
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
    __global const float *frameData, 
    __global const float *roiData, 
     int roiHeight, 
     int roiWidth, 
     int m,
     int n, 
    __global float *resData)
{
    int gid = get_global_id(0);
    int gidI = gid / roiHeight;
    int gidJ = gid % roiHeight;
    printf("a:%d ", *frameData); 
    int sum = 0;
    for (int i = gidI; i < gidI + roiHeight; i++) {
        for (int j = gidJ; j < gidJ + roiWidth; j++) {
            int dif = frameData[(i*roiHeight)+j] - roiData[(i-gidI)*roiHeight+j-gidJ];
            sum += dif*dif;
        }
    }
    resData[gid] = sum/(roiHeight*roiWidth);
    printf("res:%d ", resData[gid]); 
}
""").build()
#TODO fix the data transmition
for image in localTest:
    print(roiMingeImageData)
    print(type(frame.ravel()))
    frameData = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frame.ravel())
    roiData = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=roiMingeImageData)
    # b_m = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(heightMinge))
    # b_n = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(widthMinge))
    # m = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(mx))
    # n = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(nx))

    dim = mx * nx
    resData = cl.Buffer(ctx, mf.WRITE_ONLY, dim)
    knl = prg.calculateMSE  # Use this Kernel object for repeated calls

    #convert data
    heightData32 = np.int32(heightMinge)
    widthData32 = np.int32(widthMinge)
    mData32 = np.int32(mx)
    nData32 = np.int32(nx)

    knl(queue, (dim, 1), None, frameData, roiData, heightData32, widthData32, mData32, nData32, resData)

    res_np = np.empty_like(dim)
    cl.enqueue_copy(queue, res_np, resData)

print(res_np)
#
# # Check on CPU with Numpy:
# print(res_np - (a_np + b_np))
# print((a_np + b_np))
# print(np.linalg.norm(res_np - (a_np + b_np)))
# assert np.allclose(res_np, a_np + b_np)
