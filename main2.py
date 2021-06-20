import numpy as np
import pyopencl as cl
import os
import cv2
from natsort import natsorted
from matplotlib import patches
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Fotbal, cup, dot
dataset_name = 'fotbal'
image_folder = './frames/' + dataset_name
video_name = dataset_name + '_GPU.avi'

# Aceasta parte imparte videoclipul in frame-uri si le salveaza separat intr-un alt folder
if(dataset_name == 'dot'):
    videoFolder = './video/dot.mp4'
else:
    videoFolder = './video/' + dataset_name + '.avi'
videocap = cv2.VideoCapture(videoFolder)
success, image = videocap.read()
count = 0
while success:
    cv2.imwrite("./frames/%s/frame%d.jpg" % (dataset_name, count), image)  # save frame as JPEG file
    success, image = videocap.read()
    count += 1


# Acesta parte citeste toate numele imaginilor si le sorteaza
fpsVideo = 30
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images = natsorted(images)


roiFolderLocation = 'roi/' + dataset_name + '.jpg'

roiImageData = cv2.imread(roiFolderLocation).astype(np.uint8)
heightRoi, widthRoi, layersRoi = roiImageData.shape

globalFrame = cv2.imread('frames/' + dataset_name + '/frame0.jpg', cv2.IMREAD_GRAYSCALE).astype(np.uint8)
globalHeightFrame, globalWidthFrame = globalFrame.shape


# video = cv2.VideoWriter(video_name, 0, fpsVideo, (globalwidthFrame, globalheightFrame))
#
# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
#
# cv2.destroyAllWindows()
# video.release()

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



# START GPU PART

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

prg = cl.Program(ctx, """
__kernel void matrix_diff(
        __global const uchar *a_g, __global const uchar *b_g, const unsigned int size,  __global int *res_g) {
            int i = get_global_id(1);
            int j = get_global_id(0);
            res_g[i + size * j] = a_g[i + size * j] - b_g[i + size * j];
            res_g[i + size * j] = res_g[i + size * j] * res_g[i + size * j];
    }
""").build()


roiImageData = cv2.imread(roiFolderLocation, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
heightRoi, widthRoi = roiImageData.shape
stackResult = []
images_nr = 0
for image in images:
    currentFrameFolderLocation = os.path.join(image_folder, image)
    currentFrame = cv2.imread(currentFrameFolderLocation, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    heightFrame, widthFrame = currentFrame.shape

    mx = heightFrame - heightRoi
    nx = widthFrame - widthRoi

    result = []
    for x in range(mx):
        for y in range(nx):
            currentBox = currentFrame[x:x + heightRoi, y: y + widthRoi]
            currentBox = np.array(currentBox)
            boxData = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=currentBox)
            roiData = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=roiImageData)

            fakeVector = np.zeros(heightFrame*widthFrame).astype(np.int16)
            resData = cl.Buffer(ctx, mf.WRITE_ONLY, fakeVector.nbytes)
            knl = prg.matrix_diff  # Use this Kernel object for repeated calls


            knl(queue, (heightRoi, widthRoi), None, boxData, roiData, np.int32(widthFrame), resData)
            res_np = np.empty_like(fakeVector)
            cl.enqueue_copy(queue, res_np, resData).wait()
            suma = np.sum(res_np)
            result.append(suma)

            boxData.release()
            roiData.release()
            resData.release()
    print("Total images = " + str(images_nr))
    images_nr = images_nr + 1
    stackResult.append(result)

    current_image_path = os.path.join(image_folder, image)
    img = cv2.imread(current_image_path, cv2.COLOR_BGR2RGB)

    array = []
    min = result[0]
    i = 0
    for (index, mse) in enumerate(result):
        local = mse
        if local < min:
            min = local
            i = index

    x, y = getCoordinates(i, mx, nx)
    print(x, y)
    img_with_square = cv2.rectangle(img, (y, x), (y + widthRoi, x - heightRoi), (0, 255, 0), 2)

    cv2.imwrite("./final/%s/frame%d.jpg" % (dataset_name, images_nr), img_with_square)

print(stackResult)

minArray = getMSEMin(stackResult)  # Array with image and MSE min




found_element_coordinates = []
for pair in minArray:
    coord = getCoordinates(pair[1], globalHeightFrame, globalWidthFrame)
    found_element_coordinates.append(coord)

images_with_object_located = []
for index, image in enumerate(images):
    current_image_path = os.path.join(image_folder, image)
    img = cv2.imread(current_image_path, cv2.COLOR_BGR2RGB)
    x, y = found_element_coordinates[index]
    img_with_square = cv2.rectangle(img, (x, y), (x + widthRoi, y + heightRoi), (0, 255, 0), 2)
    images_with_object_located.append(img_with_square)

    cv2.imwrite("./final/%s/frame%d.jpg" % (dataset_name, index), img_with_square)



video = cv2.VideoWriter(video_name, 0, fpsVideo, (globalWidthFrame, globalHeightFrame))

for image in images:
    video.write(cv2.imread(os.path.join("./final/" + dataset_name + "/", image)))

cv2.destroyAllWindows()
video.release()


print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()



