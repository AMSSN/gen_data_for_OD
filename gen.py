"""
步骤如下：
1、使用yolov5-7.0的segment功能，先crop出目标主体，再分割目标主体与背景，得到mask。
2、对mask+目标图片进行处理，如缩放、低分辨率处理、模糊处理、旋转等。
3、将处理后的图片与背景图片进行随机组合，得到人造数据集，并且制作标注文件。
"""
import argparse
import logging
import os
import random
import time
from PIL import Image
import cv2
import numpy as np

from segment.predict import run_segment
from tools import add_log_file, check_seg_result_onebyone, check_overside, resize_maxside

Work_Dir = os.path.split(os.path.realpath(__file__))[0]  # E:\gen_data_for_OD


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="seg_yolov5_7.0/yolov5x-seg.pt", help='model path')
    parser.add_argument('--source', type=str, default="ori_images", help='original images folder')
    parser.add_argument('--background', type=str, default="background", help='background images folder')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default="dataset", help='save results to project')
    parser.add_argument('--yoloresult', default="yolo_result", help='save results to project')
    parser.add_argument('--dataresult', default="train", help='save results to project')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--maxside', type=int, default=300, help='max side for resize target')
    parser.add_argument('--datasize', type=int, default=100, help='number of artificial images')
    parser.add_argument('--dataepoch', type=str, default="CZ_", help='number of artificial images')
    opt = parser.parse_args()
    return opt


def run_compose(background_path, yolo_path, data_path, data_size, max_side, data_epoch):
    background_list = os.listdir(background_path)
    crop_path = os.path.join(yolo_path, "crop")
    mask_path = os.path.join(yolo_path, "mask")
    set_images = check_seg_result_onebyone(crop_path, mask_path)
    label_path = os.path.join(data_path, "label")
    if not os.path.isdir(os.path.join(data_path, "image")):
        os.mkdir(os.path.join(data_path, "image"))
    if not os.path.isdir(label_path):
        os.mkdir(label_path)
    index = 0
    for bg_i in np.random.choice(background_list, data_size):
        index += 1
        time_now = time.time()
        background_img = cv2.imread(os.path.join(background_path, bg_i))
        bg_name = data_epoch + str(index)
        f = open(os.path.join(label_path, bg_name+".txt"), "a")
        # 根据宽高比确定贴几张图片，密度多少
        h, w = background_img.shape[:2]
        num = (h // 800) * (w // 800)
        logging.info("bg h={}, w={},num={}*{}={}".format(h, w, (h // 800), (w // 800), num))
        print("bg h={}, w={},num={}*{}={}".format(h, w, (h // 800), (w //800), num))
        for target_i in np.random.choice(list(set_images), num):
            mask_img = np.array(Image.open(os.path.join(mask_path, target_i)))
            th = 240
            mask_img[mask_img < th] = 0
            mask_img[mask_img >= th] = 255
            crop_img = np.array(Image.open(os.path.join(crop_path, target_i)))
            # 处理mask
            mask = (mask_img / 255.0).astype(np.uint8)
            kernel = np.ones((5, 5), dtype=np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            # mask图和crop图resize
            resize_mask = resize_maxside(mask, max_side)
            resize_crop = resize_maxside(crop_img, max_side)
            # crop图特殊处理，暂时没有
            # 随机生成位置
            t_h, t_w = resize_mask.shape[:2]
            0, int(w - t_w / 2)
            ext_midel = [i for i in range(int(w/2)-t_w)]+[j for j in range(int(w/2), w - t_w)]
            x = random.choice(ext_midel)
            y = random.randint(0, h - t_h)
            logging.info("random point in x={}, y={}, with image h={}, w={}".format(x, y, t_h, t_w))
            print("random point in x={}, y={}, with image h={}, w={}".format(x, y, t_h, t_w))
            # 检查越界,按位置贴图，保存坐标
            if not check_overside(x, y, t_h, t_w, h, w):
                # img1[:, :, :] = img1[:, :, :] * mask
                # img2[:, :, :] = img2[:, :, :] * (1 - mask)
                # img1 = img1 + img2
                background_img[y:y + t_h, x:x + t_w, :] = background_img[y:y + t_h, x:x + t_w, :] * (1 - resize_mask)
                resize_crop = resize_crop * resize_mask
                background_img[y:y + t_h, x:x + t_w, :] = background_img[y:y + t_h, x:x + t_w, :] + resize_crop
                """
                label_index :为标签名称在标签数组中的索引，下标从 0 开始。
                cx：标记框中心点的 x 坐标，数值是原始中心点 x 坐标除以 图宽 后的结果。
                cy：标记框中心点的 y 坐标，数值是原始中心点 y 坐标除以 图高 后的结果。
                w：标记框的 宽，数值为 原始标记框的 宽 除以 图宽 后的结果。
                h：标记框的 高，数值为 原始标记框的 高 除以 图高 后的结果。
                """
                x = x+int(t_w/2)
                y = y+int(t_h/2)
                cx = x / w
                cy = y / h
                cw = t_w / w
                ch = t_h / h
                f.write("14 {} {} {} {}\n".format(cx, cy, cw, ch))
        # /home/jlm/gen_data_for_OD/dataset/train/image/
        data_save_path = os.path.join(os.path.join(data_path, "image"), bg_name + ".jpg")
        cv2.imwrite(data_save_path, background_img)
        f.close()


def run_compose2(background_path, yolo_path, data_path, data_size, max_side, data_epoch):
    background_list = os.listdir(background_path)
    crop_path = os.path.join(yolo_path, "crop")
    mask_path = os.path.join(yolo_path, "mask")
    set_images = check_seg_result_onebyone(crop_path, mask_path)
    label_path = os.path.join(data_path, "labels")
    if not os.path.isdir(os.path.join(data_path, "images")):
        os.mkdir(os.path.join(data_path, "images"))
    if not os.path.isdir(label_path):
        os.mkdir(label_path)
    index = 0
    for bg_i in np.random.choice(background_list, data_size):
        index += 1
        time_now = time.time()
        background_img = cv2.imread(os.path.join(background_path, bg_i))
        bg_name = data_epoch + str(index)
        f = open(os.path.join(label_path, bg_name+".txt"), "a")
        # 根据宽高比确定贴几张图片，密度多少
        h, w = background_img.shape[:2]
        num = (h // 800) * (w // 800)
        logging.info("bg h={}, w={},num={}*{}={}".format(h, w, (h // 800), (w // 800), num))
        print("bg h={}, w={},num={}*{}={}".format(h, w, (h // 800), (w //800), num))
        for target_i in np.random.choice(list(set_images), num):
            mask_img = np.array(Image.open(os.path.join(mask_path, target_i)))
            th = 240
            mask_img[mask_img < th] = 0
            mask_img[mask_img >= th] = 255
            crop_img = np.array(Image.open(os.path.join(crop_path, target_i)))
            # 处理mask
            mask = (mask_img / 255.0).astype(np.uint8)
            kernel = np.ones((5, 5), dtype=np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            # mask图和crop图resize
            resize_mask = resize_maxside(mask, max_side)
            resize_crop = resize_maxside(crop_img, max_side)
            # crop图特殊处理，暂时没有
            # 随机生成位置
            t_h, t_w = resize_mask.shape[:2]
            0, int(w - t_w / 2)
            ext_midel = [i for i in range(int(w/2)-t_w)]+[j for j in range(int(w/2), w - t_w)]
            x = random.choice(ext_midel)
            y = random.randint(0, h - t_h)
            logging.info("random point in x={}, y={}, with image h={}, w={}".format(x, y, t_h, t_w))
            print("random point in x={}, y={}, with image h={}, w={}".format(x, y, t_h, t_w))
            # 检查越界,按位置贴图，保存坐标
            if not check_overside(x, y, t_h, t_w, h, w):
                # img1[:, :, :] = img1[:, :, :] * mask
                # img2[:, :, :] = img2[:, :, :] * (1 - mask)
                # img1 = img1 + img2
                background_img[y:y + t_h, x:x + t_w, :] = background_img[y:y + t_h, x:x + t_w, :] * (1 - resize_mask)
                resize_crop = resize_crop * resize_mask
                background_img[y:y + t_h, x:x + t_w, :] = background_img[y:y + t_h, x:x + t_w, :] + resize_crop
                """
                label_index :为标签名称在标签数组中的索引，下标从 0 开始。
                cx：标记框中心点的 x 坐标，数值是原始中心点 x 坐标除以 图宽 后的结果。
                cy：标记框中心点的 y 坐标，数值是原始中心点 y 坐标除以 图高 后的结果。
                w：标记框的 宽，数值为 原始标记框的 宽 除以 图宽 后的结果。
                h：标记框的 高，数值为 原始标记框的 高 除以 图高 后的结果。
                """
                cx = (x+int(t_w/2)) / w
                cy = (y+int(t_h/2)) / h
                cw = t_w / w
                ch = t_h / h
                f.write("14 {} {} {} {}\n".format(cx, cy, cw, ch))
        # /home/jlm/gen_data_for_OD/dataset/train/image/
        data_save_path = os.path.join(os.path.join(data_path, "images"), bg_name + ".jpg")
        cv2.imwrite(data_save_path, background_img)
        f.close()


if __name__ == "__main__":
    infile = os.path.join(Work_Dir, "gen-data.log")
    add_log_file(infile, 20, 100)
    opt = parse_opt()
    yolo_path = os.path.join(opt.project, opt.yoloresult)
    data_path = os.path.join(opt.project, opt.dataresult)
    # 1、使用yolov5-7.0的segment功能，先crop出目标主体，再分割目标主体与背景，得到mask。
    run_segment(source=opt.source, device=opt.device, project=yolo_path, classes=opt.classes)
    # 2、对mask+目标图片进行处理，如缩放、低分辨率处理、模糊处理、旋转等。
    # 3、将处理后的图片与背景图片进行随机组合，得到人造数据集，并且制作标注文件。
    # run_compose(opt.background, yolo_path, data_path, opt.datasize, opt.maxside, opt.dataepoch)
