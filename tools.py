import logging
import logging.handlers
import os
import random
import shutil
import sys
import cv2
from PIL import Image
import numpy as np

from data_aug.blur import GlassBlur
from data_aug.noise import ShotNoise, GaussianNoise, SpeckleNoise


def add_log_file(infile=None, level=20, backup_count=10):
    logger = logging.getLogger()
    logger.setLevel(level)
    for oneHandler in logger.handlers:
        logger.removeHandler(oneHandler)
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=infile, when='midnight',
                                                            backupCount=backup_count, encoding='utf-8')
    # fileHandler = logging.FileHandler(infile)
    consoleHandler = logging.StreamHandler(sys.stdout)
    fileHandler.setLevel(level)
    consoleHandler.setLevel(logging.WARNING)
    formatter = logging.Formatter('[%(levelname)s,%(asctime)s %(filename)s:%(lineno)d]%(message)s',
                                  "%m-%d %H:%M:%S.%3M")
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)


def clean_data_cache():
    Work_Dir = os.path.split(os.path.realpath(__file__))[0]  # E:\gen_data_for_OD
    # 删除crop文件
    # gen_data_for_OD/dataset/yolo_result/
    crop_folder = os.path.join(os.path.join(Work_Dir, "dataset"), "yolo_result")
    print("del:", crop_folder)
    shutil.rmtree(crop_folder)
    print("remake:", crop_folder)
    os.makedirs(crop_folder)


def check_seg_result_onebyone(crop_result, mask_result):
    logging.info("check whether mask and crop images matching...")
    set_crop = set(os.listdir(crop_result))
    set_mask = set(os.listdir(mask_result))
    not_in_crop = set_mask - set_crop
    not_in_mask = set_crop - set_mask
    if len(not_in_crop) > 0:
        print("have mask, not have crop with:".format(not_in_crop))
        logging.warning("have mask, not have crop with:".format(not_in_crop))
    if len(not_in_mask) > 0:
        print("have crop, not have mask with:".format(not_in_mask))
        logging.warning("have crop, not have mask with:".format(not_in_mask))
    result = set_crop - not_in_crop
    return result - not_in_mask


def resize_maxside(result_img, max_side):
    # 对输入图片resize，最大边不超过max_sied
    h, w = result_img.shape[:2]
    if max(h, w) <= max_side:
        return result_img
    else:
        if h > w:
            new_h = max_side
            new_w = w / h * max_side
        else:
            new_w = max_side
            new_h = h / w * max_side
        target_ = cv2.resize(result_img, (int(new_w), int(new_h)))
        return target_


def aug():
    # 随机数的产生需要先创建一个随机数生成器（Random Number Generator）
    rng = np.random.default_rng(0)
    glassblur = GlassBlur(rng)
    gaussnoise = GaussianNoise(rng)
    shotnoise = ShotNoise(rng)
    specknoise = SpeckleNoise(rng)

    img_path = os.path.join("/ffdisk", "file")
    img = Image.open(img_path)
    out_img = glassblur(img)
    out_img = gaussnoise(img)
    out_img = shotnoise(img)
    out_img = specknoise(img)


def check_overside(x, y, t_h, t_w, h, w):
    if x + t_w > w:
        return True
    if y + t_h > h:
        return True
    return False


# 背景监控效果处理
def bg_processing(imgae_or_path, img_type="image"):
    if img_type == "image":
        img = imgae_or_path
    else:
        img = cv2.imread(imgae_or_path)
    h, w, c = img.shape
    print(h, w, c)
    # resize
    hight_size = cv2.resize(img, (int(w * 0.4), int(h * 0.4)))
    low_size = cv2.resize(hight_size, (w, h), cv2.INTER_NEAREST)
    # blur
    blur_img = cv2.GaussianBlur(low_size, (3, 3), 0)
    # sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen = cv2.filter2D(blur_img, -1, kernel)
    # 降低对比度
    img_hsv = cv2.cvtColor(sharpen, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = 0.6 * img_hsv[:, :, 1]
    colorless_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    B = colorless_img[:, :, 0]
    B[:, :] = B[:, :] * 0.94
    G = colorless_img[:, :, 1]
    G[:, :] = G[:, :] * 0.99
    if img_type == "image":
        return colorless_img
    else:
        cv2.imwrite(imgae_or_path, colorless_img)
        return None


# 目标处理
def tg_processing(imgae_or_path, img_type="image"):
    if img_type == "image":
        img = imgae_or_path
    else:
        img = cv2.imread(imgae_or_path)
    h, w, c = img.shape
    print(h, w, c)
    # resize
    hight_size = cv2.resize(img, (int(w * 0.4), int(h * 0.4)))
    low_size = cv2.resize(hight_size, (w, h), cv2.INTER_NEAREST)
    # blur
    blur_img = cv2.GaussianBlur(low_size, (3, 3), 0)
    # sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen = cv2.filter2D(blur_img, -1, kernel)
    # 降低对比度
    img_hsv = cv2.cvtColor(sharpen, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = 0.4 * img_hsv[:, :, 1]
    colorless_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    # # 增加图像亮度
    # res1 = np.uint8(np.clip((cv2.add(1 * img, 30)), 0, 255))
    # # 增加图像对比度
    # res2 = np.uint8(np.clip((cv2.add(1.5 * img, 0)), 0, 255))
    alpha = random.uniform(0.5, 0.8)
    res1 = np.uint8(np.clip((cv2.add(alpha * colorless_img, 15)), 0, 255))
    if img_type == "image":
        return colorless_img
    else:
        cv2.imwrite(imgae_or_path, res1)
        return None


def gen_imglist_txt():
    img_path = "/home/jlm/gen_data_for_OD/dataset/train/images"
    with open("/home/jlm/gen_data_for_OD/dataset/train/train.txt", "w")as f:
        for file in os.listdir(img_path):
            f.writelines("{}\n".format(os.path.join("/home/jlm/gen_data_for_OD/dataset/train/images", file)))
    txt_path = "/home/jlm/gen_data_for_OD/dataset/train/labels"
    with open("/home/jlm/gen_data_for_OD/dataset/train/labels.txt", "w")as f:
        for file in os.listdir(img_path):
            f.writelines("{}\n".format(os.path.join("/home/jlm/gen_data_for_OD/dataset/train/labels", file)))

if __name__ == '__main__':
    gen_imglist_txt()

