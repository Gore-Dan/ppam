#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import os
import cv2
from natsort import natsorted
from matplotlib import patches
import matplotlib.pyplot as plt
import time

def getMSE(m1, m2):
    return np.square(np.subtract(m1, m2)).mean()

start_time = time.time()
dataset_name = 'football'

#Generate frames for gpu
typeVideo = '.avi'
videoFolder = 'video/' + dataset_name + typeVideo
videocap = cv2.VideoCapture(videoFolder)
success, image = videocap.read()

count = 0
while success:
    folderName = './frames/' + dataset_name + '/frame%d.jpg' % count
    cv2.imwrite(folderName, image)  # save frame as JPEG file
    success, image = videocap.read()
    count += 1

image_folder = 'frames/' + dataset_name

video_name = dataset_name + typeVideo
fpsVideo = 25

# reading frames generated before and sorting them by name
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images = natsorted(images)

globalFrame = cv2.imread('./frames/' + dataset_name + '/frame0.jpg', cv2.IMREAD_GRAYSCALE).astype(np.uint8)
globalheightFrame, globalwidthFrame = globalFrame.shape

# Section of code where we build the kernel
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
localTest = [images[0]]

# the main logic is, we split the image we send in macroblocks and we calculate their MAD/MSE to find the ROI
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
            if(dif < 0){
                dif = -dif;
            }
            sum += dif;
        }
    }
    resData[gid] = (float)sum/(roiHeight*roiWidth);
}
""").build()

# for img in images:
#     currentFrameFolderLocation = os.path.join(image_folder, img)
#     currentFrame = cv2.imread(currentFrameFolderLocation).astype(np.uint8)
#     currentFramex = currentFrame.copy()
#     # set blue and green channels to 0
#     currentFramex[:, :, 0] = 0
#     currentFramex[:, :, 1] = 0
#     cv2.imwrite('./removed_channels/' + dataset_name + '/%s' % img, currentFramex)

typeImage = '.jpg'
roiFolderLocation = 'roi/' + dataset_name + typeImage

roiImageData = cv2.imread(roiFolderLocation, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
heightRoi, widthRoi = roiImageData.shape

stackResult = []

if any(os.scandir('./removed_channels/' + dataset_name)):
    print('not empty')
    image_folder = './removed_channels/' + dataset_name
    roiImageData = cv2.imread(roiFolderLocation).astype(np.uint8)

    roiImageData = roiImageData.copy()
    # set blue and green channels to 0
    roiImageData[:, :, 0] = 0
    roiImageData[:, :, 1] = 0

    cv2.imwrite("./roi/minge_red.jpg", roiImageData)


for image in images:
    currentFrameFolderLocation = os.path.join(image_folder, image)
    currentFrame = cv2.imread(currentFrameFolderLocation, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

    heightFrame, widthFrame = currentFrame.shape

    mx = heightFrame - heightRoi
    nx = widthFrame - widthRoi

    frameData = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=currentFrame)
    roiData = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=roiImageData)

    dim = mx * nx
    fakeVector = np.zeros(dim).astype(np.float32)
    resData = cl.Buffer(ctx, mf.WRITE_ONLY, fakeVector.nbytes)
    knl = prg.calculateMSE  # Use this Kernel object for repeated calls

    #convert data
    heightData32 = np.int32(heightRoi)
    widthData32 = np.int32(widthRoi)
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
        i = 0
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

minArray = getMSEMin(stackResult)  # Array with image and MSE min

roiImageData = cv2.imread(roiFolderLocation, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
heightRoi, widthRoi = roiImageData.shape
found_element_coordinates = []
for pair in minArray:
    coord = getCoordinates(pair[1], globalheightFrame - heightRoi, globalwidthFrame - widthRoi)
    found_element_coordinates.append(coord)

finalFolderLocation = 'final/' + dataset_name
image_folder = 'frames/' + dataset_name
images_with_object_located = []
for index, image in enumerate(images):
    current_image_path = os.path.join(image_folder, image)
    img = cv2.imread(current_image_path, cv2.COLOR_BGR2RGB)
    x, y = found_element_coordinates[index]
    img_with_square = cv2.rectangle(img, (y, x), (y + widthRoi, x - heightRoi), (0, 255, 0), 2)
    images_with_object_located.append(img_with_square)
    cv2.imwrite(finalFolderLocation + '/frame%d.jpg' % index, img_with_square)


# img_with_square = patches.Rectangle((y, x), heightMinge, widthMinge, linewidth=1, edgecolor='r', facecolor='none')
# ax.imshow(img)
# ax.add_patch(img_with_square)
# plt.savefig('./final/fotbal/'+image)

video = cv2.VideoWriter(video_name, 0, fpsVideo, (globalwidthFrame, globalheightFrame))
for image in images:
    video.write(cv2.imread(os.path.join(finalFolderLocation, image)))

cv2.destroyAllWindows()
video.release()


print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

# for testing
# Check on CPU with Numpy:
# cpu_results = []
# processed_images = 0
# for image in images:
#     cpuArray = []
#     currentFrameFolderLocation = os.path.join(image_folder, image)
#     frame = cv2.imread(currentFrameFolderLocation, cv2.IMREAD_GRAYSCALE).astype(np.int16)
#     heightFrame, widthFrame = frame.shape
#     roiMingeImageData = cv2.imread(roiMingeFolderLocation, cv2.IMREAD_GRAYSCALE).astype(np.int16)
#     heightMinge, widthMinge = roiMingeImageData.shape
#     for i in range(0, mx+1):
#         for j in range(0, nx+1):
#             localM1 = frame[i:i+heightMinge, j:j+widthMinge]
#             # print(localM1.shape)
#             # print(i, i + heightMinge, j, j + widthMinge)
#             diff = localM1 - roiMingeImageData
#             # mse = getMSE(localM1, roiMingeImageData)
#             localSum = np.sum(np.abs(diff))
#             cpuArray.append(localSum/(heightMinge*widthMinge))
#
#
#     processed_images += 1
#     print("Processed images: ", processed_images)
#     cpu_results.append(cpuArray)
#     minArrayCPU = getMSEMin(cpu_results)
#     print(minArrayCPU)
#
# # print(res_np)
#
# print(minArray)  # GPU results
# minArrayCPU = getMSEMin(cpu_results)
# print(minArrayCPU)  # CPU results
#
#
#
# images_with_object_located = []
# for index, image in enumerate(images):
#     current_image_path = os.path.join(image_folder, image)
#     img = cv2.imread(current_image_path, cv2.COLOR_BGR2RGB)
#     x, y = found_element_coordinates[index]
#     img_with_square = cv2.rectangle(img, (y, x), (y + heightMinge, x + widthMinge), (0, 255, 0), 2)
#     images_with_object_located.append(img_with_square)
#
#     cv2.imwrite("./final/dot_cpu/frame%d.jpg" % index, img_with_square)
#
# video = cv2.VideoWriter('dot_cpu.avi', 0, fpsVideo, (globalwidthFrame, globalheightFrame))
#
# for image in images:
#     video.write(cv2.imread(os.path.join("./final/dot_cpu/", image)))
#
# cv2.destroyAllWindows()
# video.release()


# print(np.linalg.norm(minArray - minArrayCPU))
print("--- %s seconds ---" % (time.time() - start_time))
# print(res_np - cpuArray)
# print(np.linalg.norm(res_np - cpuArray))
# assert np.allclose(res_np, cpuArray)
