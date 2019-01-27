import numpy as np
import cv2

# Just consider 256 situations.
# There are some jobs to do to reduce situations to 64 according to the paper.
inds = []
indx = [1, 3, 5, 7]
indy = [0, 1, 2, 3]
for i in indx:
    for j in indx:
        for m in indy:
            for n in indy:
                inds.append([i, j, m, n])


def init_points3D(dims):
    points3D = np.zeros((8, 3))
    cnt = 0
    for i in [1, -1]:
        for j in [1, -1]:
            for k in [1, -1]:
                points3D[cnt] = dims[[1, 0, 2]].T / 2.0 * [i, k, j * i]
                cnt += 1
    return points3D


def solve_least_squre(W,y):
    U, Sigma, VT = np.linalg.svd(W)
    result = np.dot(np.dot(np.dot(VT.T, np.linalg.pinv(np.eye(4, 3) * Sigma)), U.T), y)
    return result


def points3D_to_2D(points3D,center,rot_M,cam_to_img):
    points2D = []
    for point3D in points3D:
        point3D = point3D.reshape((-1,1))
        point = center + np.dot(rot_M, point3D)
        point = np.append(point, 1)
        point = np.dot(cam_to_img, point)
        point2D = point[:2] / point[2]
        points2D.append(point2D)
    points2D = np.asarray(points2D)

    return points2D


def compute_error(points3D,center,rot_M, cam_to_img,box_2D):
    points2D = points3D_to_2D(points3D, center, rot_M, cam_to_img)
    new_box_2D = np.asarray([np.min(points2D[:,0]),
                  np.max(points2D[:,0]),
                  np.min(points2D[:,1]),
                  np.max(points2D[:,1])]).reshape((-1,1))
    error = np.sum(np.abs(new_box_2D - box_2D))

    return error


def compute_center(points3D,rot_M,cam_to_img,box_2D,inds):
    fx = cam_to_img[0][0]
    fy = cam_to_img[1][1]
    u0 = cam_to_img[0][2]
    v0 = cam_to_img[1][2]
    W = np.array([[fx, 0, u0 - box_2D[0]],
                  [fx, 0, u0 - box_2D[2]],
                  [0, fy, v0 - box_2D[1]],
                  [0, fy, v0 - box_2D[3]]])
    center =None
    error_min = 1e10

    for ind in inds:
        y = np.zeros((4, 1))
        for i in range(len(ind)):
            RP = np.dot(rot_M, (points3D[ind[i]]).reshape((-1, 1)))
            y[i] = box_2D[i] * cam_to_img[2, 3] - np.dot(W[i], RP) - cam_to_img[i // 2, 3]
        result = solve_least_squre(W, y)
        error = compute_error(points3D, result, rot_M, cam_to_img, box_2D)
        if error < error_min and result[2,0]>0:
            center = result
            error_min = error
    return center


def draw_3D_box(image,points):
    points = points.astype(np.int)

    for i in range(4):
        point_1_ = points[2 * i]
        point_2_ = points[2 * i + 1]
        cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0, 255, 0), 2)

    cv2.line(image,tuple(points[0]),tuple(points[7]),(0, 0, 255), 2)
    cv2.line(image, tuple(points[1]), tuple(points[6]), (0, 0, 255), 2)

    for i in range(8):
        point_1_ = points[i]
        point_2_ = points[(i + 2) % 8]
        cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0, 255, 0), 2)


def draw_2D_box(image,points):
    points = points.astype(np.int)
    cv2.rectangle(image,tuple(points[0:2]),tuple(points[2:4]),(0, 255, 0), 2)


def gen_3D_box(yaw,dims,cam_to_img,box_2D):
    # yaw = -93.08241511578096/180*np.pi
    dims = dims.reshape((-1,1))
    box_2D = box_2D.reshape((-1,1))
    points3D = init_points3D(dims)

    rot_M = np.asarray([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])

    center = compute_center(points3D, rot_M, cam_to_img, box_2D, inds)

    points2D = points3D_to_2D(points3D, center, rot_M, cam_to_img)

    return points2D



