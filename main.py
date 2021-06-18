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


video = cv2.VideoWriter(video_name, 0, fpsVideo, (widthFrame, heightFrame))

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
    __global const float *a_g, 
    __global const float *b_g, 
     int b_m, 
     int b_n, 
     int m,
     int n, 
    __global float *res_g)
{
  int gid = get_global_id(0);
  int gidI = gid / b_m;
  int gidJ = gid % b_m;
  printf("a:%d ", *a_g); 
        int sum = 0;
        for (int i = gidI; i < gidI + b_m; i++) {
          for (int j = gidJ; j < gidJ + b_n; j++) {
             int dif = a_g[(i*b_m)+j] - b_g[(i-gidI)*b_m+j-gidJ];
             sum += dif*dif;
          }
        }
   
   res_g[gid] = sum/(b_m*b_n);
   printf("res:%d ", res_g[gid]); 
}
""").build()
#TODO fix the data transmition
for image in localTest:
    print(roiMingeImageData)
    print(type(frame.ravel()))
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frame.ravel())
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=roiMingeImageData)
    # b_m = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(heightMinge))
    # b_n = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(widthMinge))
    # m = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(mx))
    # n = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(nx))

    dim = mx * nx
    print(dim)
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, dim)
    knl = prg.calculateMSE  # Use this Kernel object for repeated calls
    knl(queue, (dim, 1), None, a_g, b_g, np.int32(heightMinge), np.int32(widthMinge), np.int32(mx), np.int32(nx), res_g)

    res_np = np.empty_like(dim)
    cl.enqueue_copy(queue, res_np, res_g)

print(res_np)
#
# # Check on CPU with Numpy:
# print(res_np - (a_np + b_np))
# print((a_np + b_np))
# print(np.linalg.norm(res_np - (a_np + b_np)))
# assert np.allclose(res_np, a_np + b_np)
