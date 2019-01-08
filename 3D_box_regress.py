from keras.layers.core import Flatten, Dropout, Reshape, Lambda
import copy

import cv2
import numpy as np
import os
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dropout, Reshape, Lambda
from keras.models import Model
from keras.optimizers import SGD,Adam
from util.Final_KITTI_Evaluation import gen_3D_box

# get_ipython().magic(u'matplotlib inline')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

BIN, OVERLAP = 2, 0.1
W = 1.
ALPHA = 1.
MAX_JIT = 3
NORM_H, NORM_W = 224, 224
VEHICLES = ['Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram']
BATCH_SIZE = 8

label_dir = 'F:/dataset/kitti/training/label_2/'
image_dir = 'F:/dataset/kitti/training/image_2/'
calib_dir = 'F:/dataset/kitti/training/calib/'


def compute_anchors(angle):
    anchors = []

    wedge = 2. * np.pi / BIN
    l_index = int(angle / wedge)
    r_index = l_index + 1

    if (angle - l_index * wedge) < wedge / 2 * (1 + OVERLAP / 2):
        anchors.append([l_index, angle - l_index * wedge])

    if (r_index * wedge - angle) < wedge / 2 * (1 + OVERLAP / 2):
        anchors.append([r_index % BIN, angle - r_index * wedge])

    return anchors

def parse_annotation(label_dir):
    all_objs = []
    dims_avg = {key: np.array([0, 0, 0]) for key in VEHICLES}
    dims_cnt = {key: 0 for key in VEHICLES}

    for label_file in os.listdir(label_dir):
        image_file = label_file.replace('txt', 'png')

        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded = np.abs(float(line[2]))

            if line[0] in VEHICLES and truncated < 0.1 and occluded < 0.1:
                new_alpha = float(line[3]) + np.pi / 2.
                if new_alpha < 0:
                    new_alpha = new_alpha + 2. * np.pi
                new_alpha = new_alpha - int(new_alpha / (2. * np.pi)) * (2. * np.pi)

                obj = {'name': line[0],
                       'image': image_file,
                       'xmin': int(float(line[4])),
                       'ymin': int(float(line[5])),
                       'xmax': int(float(line[6])),
                       'ymax': int(float(line[7])),
                       'dims': np.array([float(number) for number in line[8:11]]),
                       'new_alpha': new_alpha
                       }

                dims_avg[obj['name']] = dims_cnt[obj['name']] * dims_avg[obj['name']] + obj['dims']
                dims_cnt[obj['name']] += 1
                dims_avg[obj['name']] /= dims_cnt[obj['name']]

                all_objs.append(obj)

    return all_objs, dims_avg


all_objs, dims_avg = parse_annotation(label_dir)

for obj in all_objs:
    # Fix dimensions
    obj['dims'] = obj['dims'] - dims_avg[obj['name']]

    # Fix orientation and confidence for no flip
    orientation = np.zeros((BIN, 2))
    confidence = np.zeros(BIN)

    anchors = compute_anchors(obj['new_alpha'])

    for anchor in anchors:
        orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
        confidence[anchor[0]] = 1.

    confidence = confidence / np.sum(confidence)

    obj['orient'] = orientation
    obj['conf'] = confidence

    # Fix orientation and confidence for flip
    orientation = np.zeros((BIN, 2))
    confidence = np.zeros(BIN)

    anchors = compute_anchors(2. * np.pi - obj['new_alpha'])

    for anchor in anchors:
        orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
        confidence[anchor[0]] = 1

    confidence = confidence / np.sum(confidence)

    obj['orient_flipped'] = orientation
    obj['conf_flipped'] = confidence


def prepare_input_and_output(train_inst):
    ### Prepare image patch
    xmin = train_inst['xmin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymin = train_inst['ymin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    xmax = train_inst['xmax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymax = train_inst['ymax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)

    img = cv2.imread(image_dir + train_inst['image'])
    img = copy.deepcopy(img[ymin:ymax + 1, xmin:xmax + 1]).astype(np.float32)

    # flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5: img = cv2.flip(img, 1)

    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    img = img - np.array([[[103.939, 116.779, 123.68]]])

    ### Fix orientation and confidence
    if flip > 0.5:
        return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped']
    else:
        return img, train_inst['dims'], train_inst['orient'], train_inst['conf']


def data_gen(all_objs, batch_size):
    num_obj = len(all_objs)

    keys = list(range(num_obj))
    np.random.shuffle(keys)

    l_bound = 0
    r_bound = batch_size if batch_size < num_obj else num_obj

    while True:
        if l_bound == r_bound:
            l_bound = 0
            r_bound = batch_size if batch_size < num_obj else num_obj
            np.random.shuffle(keys)

        currt_inst = 0
        x_batch = np.zeros((r_bound - l_bound, 224, 224, 3))
        d_batch = np.zeros((r_bound - l_bound, 3))
        o_batch = np.zeros((r_bound - l_bound, BIN, 2))
        c_batch = np.zeros((r_bound - l_bound, BIN))

        for key in keys[l_bound:r_bound]:
            # augment input image and fix object's orientation and confidence
            image, dimension, orientation, confidence = prepare_input_and_output(all_objs[key])

            x_batch[currt_inst, :] = image
            d_batch[currt_inst, :] = dimension
            o_batch[currt_inst, :] = orientation
            c_batch[currt_inst, :] = confidence

            currt_inst += 1

        yield x_batch, [d_batch, o_batch, c_batch]

        l_bound = r_bound
        r_bound = r_bound + batch_size
        if r_bound > num_obj: r_bound = num_obj

def l2_normalize(x):
    return tf.nn.l2_normalize(x, dim=2)

# Construct the network
inputs = Input(shape=(224,224,3))
# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

x = Flatten()(x)

dimension   = Dense(512)(x)
dimension   = LeakyReLU(alpha=0.1)(dimension)
dimension   = Dropout(0.5)(dimension)
dimension   = Dense(3)(dimension)
dimension   = LeakyReLU(alpha=0.1, name='dimension')(dimension)

orientation = Dense(256)(x)
orientation = LeakyReLU(alpha=0.1)(orientation)
orientation = Dropout(0.5)(orientation)
orientation = Dense(BIN*2)(orientation)
orientation = LeakyReLU(alpha=0.1)(orientation)
orientation = Reshape((BIN,-1))(orientation)
orientation = Lambda(l2_normalize, name='orientation')(orientation)

confidence  = Dense(256)(x)
confidence  = LeakyReLU(alpha=0.1)(confidence)
confidence  = Dropout(0.5)(confidence)
confidence  = Dense(BIN, activation='softmax', name='confidence')(confidence)

model = Model(inputs, outputs=[dimension, orientation, confidence])


def orientation_loss(y_true, y_pred):
    # Find number of anchors
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)

    # Define the loss
    loss = -(tf.cos(y_true[:, :, 0] - y_pred[:, :, 0]) + tf.cos(y_true[:, :, 1] - y_pred[:, :, 1]))
    loss = tf.reduce_sum(loss, axis=1)
    loss = loss / anchors

    return tf.reduce_mean(loss)

early_stop  = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)
checkpoint  = ModelCheckpoint('weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
tensorboard = TensorBoard(log_dir='../logs/', histogram_freq=0, write_graph=True, write_images=False)

all_exams  = len(all_objs)
trv_split  = int(0.9*all_exams)
batch_size = 32
np.random.shuffle(all_objs)

train_gen = data_gen(all_objs[:trv_split],          batch_size)
valid_gen = data_gen(all_objs[trv_split:all_exams], batch_size)

train_num = int(np.ceil(trv_split/batch_size))
valid_num = int(np.ceil((all_exams - trv_split)/batch_size))

# minimizer  = Adam(lr=1e-4)
# model.compile(optimizer=minimizer,#minimizer,
#               loss={'dimension': 'mean_squared_error', 'orientation': orientation_loss, 'confidence': 'binary_crossentropy'},
#                   loss_weights={'dimension': 1., 'orientation': 1., 'confidence': 1.})
# model.fit_generator(generator = train_gen,
#                     steps_per_epoch = train_num,
#                     epochs = 3,
#                     verbose = 1,
#                     validation_data = valid_gen,
#                     validation_steps = valid_num,
#                     callbacks = [early_stop, checkpoint, tensorboard],
#                     max_q_size = 3)
model.load_weights('weights.h5')

image_dir = 'F:/dataset/kitti/training/image_2/'
box2d_loc = 'F:/dataset/kitti/training/label_2/'
box3d_loc = 'F:/dataset/kitti/training/label_3/'

dpi = 250
in_to_cm = 2.54

all_image = sorted(os.listdir(image_dir))
# np.random.shuffle(all_image)

for f in all_image:
    image_file = image_dir + f
    box2d_file = box2d_loc + f.replace('png', 'txt')
    box3d_file = box3d_loc + f.replace('png', 'txt')
    calib_file = calib_dir + f.replace('png', 'txt')

    for line in open(calib_file):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
    fx = cam_to_img[0][0]
    u0 = cam_to_img[0][2]
    v0 = cam_to_img[1][2]

    img = cv2.imread(image_file)

    for line in open(box2d_file):
        line = line.strip().split(' ')
        class_name = line[0]
        # Transform regressed dimension
        if line[0] not in dims_avg.keys():
            continue
        truncated = np.abs(float(line[1]))
        occluded = np.abs(float(line[2]))

        obj = {'xmin': int(float(line[4])),
               'ymin': int(float(line[5])),
               'xmax': int(float(line[6])),
               'ymax': int(float(line[7])),
               }

        box_2D = np.asarray([obj['xmin'], obj['xmax'], obj['ymin'], obj['ymax']])

        patch = img[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax']]
        patch = cv2.resize(patch, (NORM_H, NORM_W))
        patch = patch - np.array([[[103.939, 116.779, 123.68]]])
        patch = np.expand_dims(patch, 0)

        prediction = model.predict(patch)

        # compute dims
        dims = dims_avg[class_name] + prediction[0][0]

        # Transform regressed angle
        box2d_center_x = (obj['xmin'] + obj['xmax']) / 2.0
        theta_ray = np.arctan((fx / 100) / ((box2d_center_x - u0) / dpi * in_to_cm))

        max_anc = np.argmax(prediction[2][0])
        anchors = prediction[1][0][max_anc]

        if anchors[1] > 0:
            angle_offset = np.arccos(anchors[0])
        else:
            angle_offset = -np.arccos(anchors[0])

        wedge = 2. * np.pi / BIN
        angle_offset = angle_offset + max_anc * wedge

        rot_y = np.pi / 2 - (-angle_offset + theta_ray)

        gen_3D_box(img, rot_y, dims, cam_to_img, box_2D)

    cv2.imshow(f, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
