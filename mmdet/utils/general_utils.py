import os
import time
import numpy as np
import cv2
import mmcv

import torch
from torchvision import transforms
import torchvision.utils as vutils


def getPathList(path, suffix='png'):
    if (path[-1] != '/') & (path[-1] != '\\'):
        path = path + '/'
    pathlist = list()
    g = os.walk(path)
    for p, d, filelist in g:
        for filename in filelist:
            if filename.endswith(suffix):
                pathlist.append(os.path.join(p, filename))
    return pathlist


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    # if not os.path.isdir(path):
    #     mkdir(os.path.split(path)[0])
    # else:
    #     return
    # if not os.path.isdir(path):
    #     os.mkdir(path)
    return


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = (mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb))
        img = (np.clip(img * 255, a_min=0, a_max=255)).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def path_join(root, name):
    if root == '':
        return name
    if name[0] == '/':
        return os.path.join(root, name[1:])
    else:
        return os.path.join(root, name)


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def __write_images(im_outs, dis_img_n, file_name, normalize=True):
    im_outs = [images.expand(-1, 3, -1, -1) for images in im_outs]
    image_tensor = torch.cat([images[:dis_img_n] for images in im_outs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=dis_img_n, padding=0, normalize=normalize)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_images_for_tesnsor(feature_list, image_directory, postfix):
    display_image_num = feature_list[0].size(0)
    img_file = '%s/gen_%s_hough.jpg' % (image_directory, postfix)
    __write_images(feature_list, display_image_num, img_file, normalize=True)


def write_images(outputs, image_directory, postfix, img_w=640, img_h=360):
    disp_lane, disp_map, disp_hough = outputs
    gt_img, pre_lane = disp_lane

    device = gt_img.device
    display_image_num = gt_img.size(0)
    img_outs = [gt_img]
    colors = [[1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # vp map
    for k, map in enumerate(disp_map):
        B, C, H, W = map.shape  # B, 6, 360, 640
        map = map.cpu()  # B, 6, 72
        for k in range(C):
            disp_img = gt_img.clone() * 0.1 - 1
            color = torch.tensor(np.array(colors[k]), dtype=torch.float).to(device)
            for m in range(B):
                alpha = map[m, k].expand((3, -1, -1)).to(device)
                disp_img[m] = disp_img[m] * (1 - alpha) + alpha * color.reshape(3, 1, 1)
            img_outs.append(disp_img)

    # lane
    pos_y = np.tile(np.linspace(0, img_h - 1, img_h), 6).astype(np.int)
    for lanes in [pre_lane]:
        B, N, L = lanes.shape  # B, 6, 72
        lanes = lanes.cpu()  # B, 6, 72
        disp_img = torch.zeros_like(gt_img).to(device)
        for i in range(B):
            disp_img[i] = gt_img[i].clone()

            for j in range(N):
                real_points = [(int(x), int(y)) for (x, y) in zip(lanes[i, j], pos_y) if x != img_w]

                for point in real_points:
                    x, y = point
                    min_x, max_x = max(0, x - 2), min(x + 2, img_w - 1)
                    min_y, max_y = max(0, y - 2), min(y + 2, img_h - 1)
                    if min_x > max_x or min_y > max_y:
                        continue

                    color = torch.tensor(np.array(colors[j]).reshape((3, 1, 1)), dtype=torch.float).to(device)
                    color = color.expand((disp_img[i].shape[0], max_y - min_y, max_x - min_x))
                    disp_img[i][:, min_y:max_y, min_x:max_x] = color  # [1, 3, 720, 1280]
        img_outs.append(disp_img)

    img_file = '%s/gen_%s_lane.jpg' % (image_directory, postfix)
    __write_images(img_outs, display_image_num, img_file, normalize=True)

    img_file = '%s/gen_%s_hough.jpg' % (image_directory, postfix)
    __write_images(disp_hough, display_image_num, img_file, normalize=True)

    # img_file = '%s/gen_%s_aug_weight.jpg' % (image_directory, postfix)
    # __write_images([pre_aug_weight], display_image_num, img_file, normalize=True)
    return img_file
