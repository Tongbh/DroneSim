import cv2
import numpy as np
import pyrealsense2 as rs
import time
np.random.seed(1337)
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.models import load_model
import time
import traceback
import sys
np.set_printoptions(threshold=1e6)
#eclare pointcloud object, for calculating pointclouds and texture mappings
# pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
# points = rs.points()
def detect_num(image):
    #image = cv2.imread('hello9.jpg')
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    LowerYellow = np.array([11, 125, 150])
    UpperYellow = np.array([25, 255, 255])
    mask = cv2.inRange(HSV, LowerYellow, UpperYellow)
    mask[np.where(mask[:, :] < 255)] = 0
    ret, img = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)  # 黑白二值化

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))  # 定义结构元素

    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算

    image = img
    # cv2.imshow("img", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image[np.where(image[:, :] < 255)] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))  # 定义结构元素

    img1 = cv2.dilate(image, kernel, iterations=1)
    img1 = cv2.erode(img1, kernel, iterations=4)
    img1 = cv2.dilate(img1, kernel, iterations=3)
    ret, image = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV)  # 黑白二值化

    image_x = image.sum(axis=1).reshape(480, 1) / 255
    image_y = image.sum(axis=0).reshape(640, 1) / 255
    for i in range(-480, -1):
        #     print(image_x[-i-1])
        if (image_x[-i - 1] == image_x[-i - 5]) and (image_x[-i - 1] != [0.]):
            length = image_x[-i - 1]
            bondary_x = -i - 1
            for j in range(0, 639):
                if image[-i - 1][j] == 255 and image[-i - 1][j - 5] == 255:
                    bondary_y = j
            break
        else:
            continue

    # print(length[0])
    length = int(length[0])
    # print(bondary_x,bondary_y,length)
    image = image[bondary_x - length - 80:bondary_x + 80, bondary_y - length - 80:bondary_y + 80]

    image_cut = image
    # image_cut = cv2.copyMakeBorder(img,10,10,10,10, cv2.BORDER_CONSTANT,value=[0,0,0])
    cv2.imwrite('image_cut.jpg', image_cut)
    print('cut step 1 succeed')
    # cv2.imshow("img", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    length = image_cut.shape[0]
    # print(length)
    image_x = image_cut.sum(axis=1).reshape(length, 1) / 255
    image_y = image_cut.sum(axis=0).reshape(length, 1) / 255
    # print(image_x)
    for m in range(0, length):
        if image_x[m, 0] < 0.7 * length:
            image_cut[m] = 255
        else:
            mm = m
            break
    for n in range(0, length):
        if image_x[length - n - 1, 0] < 0.7 * length:
            image_cut[length - 1 - n] = 255
        else:
            nn = n
            break
    for i in range(0, length):
        if image_y[i, 0] < 0.7 * length:
            image_cut[..., i] = 255
        else:
            ii = i
            break
    for j in range(0, length):
        if image_y[length - j - 1, 0] < 0.75 * length:
            image_cut[..., length - 1 - j] = 255
        else:
            jj = j
            break
    # cv2.imshow("2", image_cut)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))  # 定义结构元素

    img1 = cv2.dilate(image_cut, kernel, iterations=1)
    img1 = cv2.erode(img1, kernel, iterations=4)
    image_cut = cv2.dilate(img1, kernel, iterations=3)
    image_cut[np.where(image[:, :] == 255)] = 1
    image_cut[np.where(image[:, :] == 0)] = 255
    image_cut[np.where(image[:, :] == 1)] = 0
    # cv2.imshow('1', image_cut)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('fffffff.jpg',image_cut)

    x, y = image_cut.shape
    image_cx = image_cut.sum(axis=1).reshape(x, 1)
    image_cy = image_cut.sum(axis=0).reshape(y, 1)
    # print(x,y)
    # print(image_cy)
    aa = bb = cc = dd = 0
    for a in range(0, x):
        if image_cx[a, 0] > 20000:
            aa = a
            #     print(a)
            break
    for b in range(0, x):
        # print(image_cx[x-b-1,0])
        if image_cx[x - b - 1, 0] > 20000:
            bb = x - b
            #    print(b)
            break
    for c in range(0, y):
        if image_cy[c, 0] > 20000:
            cc = c
            #     print(c)
            break
    for d in range(0, y):
        if image_cy[y - d - 1, 0] > 20000:
            dd = y - d
            #     print(d)
            break
    x = bb - aa
    y = cc - dd
    image_cut = image_cut[aa:bb, cc:dd]
    # cv2.imshow('1', image_cut)
    # time.sleep(1)
    dim = (28, 28)
    resized = cv2.resize(image_cut, dim, interpolation=cv2.INTER_NEAREST)
    blur = cv2.medianBlur(resized, 5)
    cv2.imwrite('resized_new.jpg', blur)
    print('cut step 2 succeed')
    # 构建keras模型
    model = Sequential()

    # 第一个卷积层，输出结果为（32, 28, 28)
    model.add(Convolution2D(
        batch_input_shape=(None, 1, 28, 28),
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        data_format='channels_first',
    ))
    model.add(Activation('relu'))

    # 第一个最大值池化层，输出为(32, 14, 14)
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',
        data_format='channels_first',
    ))

    # 第二个卷积层，输出结果为(64, 14, 14)
    model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
    model.add(Activation('relu'))

    # 第二个最大值池化层，输出为 (64, 7, 7)
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

    # 全连接层1 input shape (64 * 7 * 7) = (3136), output shape (512)
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    # 全连接层softmax分为九类
    model.add(Dense(9))
    model.add(Activation('softmax'))

    model = load_model('cnn_model.h5')
    scene = blur
    scene = scene.reshape(1, 1, 28, 28)
    model = model.predict(scene).flatten()

    model = (np.argwhere(model == 1) + 1)[0][0]

    print('test after load: ', model)

    return model

if __name__ == '__main__':
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    # Start streaming
    pipe_profile = pipeline.start(config)
    print('pipe started!')
    align_to = rs.stream.color
    align = rs.align(align_to)

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        img_color = np.asanyarray(color_frame.get_data())
        img_depth = np.asanyarray(depth_frame.get_data())

        #dis = np.sum(np.where(img_depth))
        #print(img_depth)
        #print(np.shape(img_depth))

        # dis = 0
        # count = 0
        # for i in range(640):
        #     for j in range(430):
        #         if depth_frame.get_distance(i,j)<3:
        #             dis = dis +depth_frame.get_distance(i,j)
        #             img_color[j][i] = 255
        #             count = count+1
        # print('dis : '+str(dis/count))
        #t=time.time()
        print(np.shape(img_depth))
        cv2.waitKey(0)
        for i in range(640):
            for j in range(480):
                if depth_frame.get_distance(i,j)>4:
                    img_color[j][i]=255
        #t1=time.time()
        #print('time'+str(t1-t))
        print('trying..')
        print(time.time())
        try:
            a = detect_num(img_color)
            print('detected: ' + str(a))
        except Exception as e:
            traceback.print_exc()
        #print(depth_frame.get_distance(320,240))

        #cv2.circle(img_color, (320, 240), 8, [255, 0, 255], thickness=-1)
        #cv2.circle(img_color, (320, 400), 8, [255, 0, 255], thickness=-1)

        cv2.imshow('depth_frame', img_color)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    pipeline.stop()