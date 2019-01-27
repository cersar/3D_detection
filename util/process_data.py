import os
import numpy as np
import cv2
import copy


def parse_annotation(label_dir, cls_to_ind):
    all_objs = []

    for label_file in os.listdir(label_dir):
        image_file = label_file.replace('txt', 'png')

        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded = np.abs(float(line[2]))
            cls = line[0]

            # add objects to train data only when their truncated level <0.3 and occluded level <= 1
            if cls in cls_to_ind.keys() and truncated < 0.3 and occluded <= 1:
                theta_loc = -float(line[3]) + 3*np.pi / 2.
                # Make sure object's theta_loc is in [0..2*pi].
                theta_loc = theta_loc - np.floor(theta_loc / (2. * np.pi)) * (2. * np.pi)

                obj = {'name': cls,
                       'image': image_file,
                       'xmin': int(float(line[4])),
                       'ymin': int(float(line[5])),
                       'xmax': int(float(line[6])),
                       'ymax': int(float(line[7])),
                       'dims': np.array([float(number) for number in line[8:11]]),
                       'theta_loc': theta_loc
                       }

                all_objs.append(obj)

    return all_objs


def compute_anchors(angle, bin_num=6, overlap=0.1):
    anchors = []

    wedge = 2. * np.pi / bin_num
    l_index = int(angle / wedge)
    r_index = l_index + 1

    if (angle - l_index * wedge) < wedge / 2 * (1 + overlap / 2):
        anchors.append([l_index, angle - l_index * wedge])

    if (r_index * wedge - angle) < wedge / 2 * (1 + overlap / 2):
        anchors.append([r_index % bin_num, angle - r_index * wedge])

    return anchors


def process_obj_attributes(objs, dims_avg, cls_to_ind, bin_num=6, overlap=0.1):
    for obj in objs:
        # Fix dimensions
        obj['dims'] = obj['dims'] - dims_avg[cls_to_ind[obj['name']]]

        # Fix orientation and confidence for no flip
        orientation = np.zeros((bin_num, 2))
        confidence = np.zeros(bin_num)

        anchors = compute_anchors(obj['theta_loc'], bin_num, overlap)

        for anchor in anchors:
            orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1.

        confidence = confidence / np.sum(confidence)

        obj['orient'] = orientation
        obj['conf'] = confidence

        # Fix orientation and confidence for flip
        orientation = np.zeros((bin_num, 2))
        confidence = np.zeros(bin_num)

        anchors = compute_anchors(2. * np.pi - obj['theta_loc'], bin_num)

        for anchor in anchors:
            orientation[anchor[0]] = np.array([np.cos(anchor[1]), np.sin(anchor[1])])
            confidence[anchor[0]] = 1

        confidence = confidence / np.sum(confidence)

        obj['orient_flipped'] = orientation
        obj['conf_flipped'] = confidence

    return objs


def get_obj_patch(image_dir, obj, target_size = (224, 224)):
    ### Prepare image patch
    xmin = obj['xmin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymin = obj['ymin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    xmax = obj['xmax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymax = obj['ymax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)

    img = cv2.imread(image_dir + obj['image'])
    img = copy.deepcopy(img[ymin:ymax + 1, xmin:xmax + 1]).astype(np.float32)

    # flip the image
    flip = np.random.binomial(1, .5)
    is_flipped = False
    if flip > 0.5:
        img = cv2.flip(img, 1)
        is_flipped = True

    # resize the image to standard size
    img = cv2.resize(img, target_size)
    img = img - np.array([[[103.939, 116.779, 123.68]]])

    return img, is_flipped


def load_and_process_annotation_data(label_dir,dims_avg,cls_to_ind):
    objs = parse_annotation(label_dir,cls_to_ind)
    return process_obj_attributes(objs, dims_avg, cls_to_ind)


def train_data_gen(all_objs, image_dir, batch_size,bin_num=6):

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
        o_batch = np.zeros((r_bound - l_bound, bin_num, 2))
        c_batch = np.zeros((r_bound - l_bound, bin_num))

        for key in keys[l_bound:r_bound]:
            # get object patch and do augment
            obj = all_objs[key]
            image, is_flipped= get_obj_patch(image_dir,all_objs[key])
            # fix object's orientation and confidence
            if is_flipped:
                dimension, orientation, confidence = obj['dims'], obj['orient_flipped'], obj['conf_flipped']
            else:
                dimension, orientation, confidence = obj['dims'], obj['orient'], obj['conf']

            x_batch[currt_inst, :] = image
            d_batch[currt_inst, :] = dimension
            o_batch[currt_inst, :] = orientation
            c_batch[currt_inst, :] = confidence

            currt_inst += 1

        yield x_batch, [d_batch, o_batch, c_batch]

        l_bound = r_bound
        r_bound = r_bound + batch_size
        if r_bound > num_obj: r_bound = num_obj


def get_cam_data(calib_file):
    for line in open(calib_file):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
            return cam_to_img


def get_dect2D_data(box2d_file,classes):
    dect2D_data = []
    box2d_reserved = []
    for line in open(box2d_file):
        line = line.strip().split(' ')
        cls = line[0]
        truncated = np.abs(float(line[1]))
        occluded = np.abs(float(line[2]))

        # Transform regressed dimension
        if cls in classes:
            box_2D = np.asarray(line[4:8],dtype=np.float)
            # draw 3D box only when the object's truncated level <0.3 and occluded level <= 1,
            # the rests are to be drawn their origin 2D box
            if truncated < 0.3 and occluded <= 1:
                dect2D_data.append([cls, box_2D])
            else:
                box2d_reserved.append([cls, box_2D])

    return dect2D_data,box2d_reserved
