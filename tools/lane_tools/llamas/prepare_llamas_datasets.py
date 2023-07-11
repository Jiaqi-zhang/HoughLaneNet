import os
import ujson
import cv2
import argparse
from tqdm import tqdm
import numpy as np
from llamas_utils import get_horizontal_values_for_four_lanes


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preparation")
    parser.add_argument("path", type=str, help="LLAMAS dataset root path")
    args = parser.parse_args()
    return args


def img_show(img, lane_lines, y_samples, fname, img_path):
    lanes_vis = [
        [(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in lane_lines
    ]
    img_vis = img.copy()
    for lane in lanes_vis:
        cv2.polylines(
            img_vis, np.int32([lane]), isClosed=False, color=(0, 255, 0), thickness=3
        )
    cv2.imwrite(os.path.join(img_path, fname), img_vis)


def prepare_labels(base_path, save_path, data):
    ori_img_w = 1276
    ori_img_h = 717

    for key, value in data.items():
        json_img_path = os.path.join(save_path, f"json_lanes_{key}")
        if not os.path.exists(json_img_path):
            os.makedirs(json_img_path)
        print(f"Preparing {key} label data...")
        
        # vis_img_path = os.path.join(save_path, f"vis_lanes_{key}")
        # if not os.path.exists(vis_img_path):
        #     os.makedirs(vis_img_path)

        file_list = []
        data_path = os.path.join(base_path, value)
        for d in os.listdir(data_path):
            for f in tqdm(os.listdir(os.path.join(data_path, d))):
                fpath = os.path.join(data_path, d, f)
                if not fpath.endswith(".json"):
                    continue

                lanes = get_horizontal_values_for_four_lanes(fpath)
                lanes = [
                    lane for lane in lanes if len(lane) > 2
                ]  # remove lanes with less than 2 points
                mask_path = f.replace(".json", ".png")

                # save info
                raw_file_path = os.path.join(
                    "color_images", key, d, f"{f[0:-5]}_color_rect.png"
                )
                lane_dict = {
                    "raw_file": raw_file_path,
                    "h_samples": np.arange(0, ori_img_h, 1).tolist(),
                    "lanes": lanes,
                }
                file_list.append(raw_file_path + "\n")

                out_file = open(os.path.join(json_img_path, f"{d}_{f}"), "w")
                ujson.dump(lane_dict, out_file, escape_forward_slashes=False)

                # raw_file_path = os.path.join(base_path, 'color_images', key, d, f"{f[0:-5]}_color_rect.png")
                # print(raw_file_path)
                # img = cv2.imread(raw_file_path)
                # img_show(img, lane_dict['lanes'], lane_dict['h_samples'], lane_dict['raw_file'].replace('/', '_'))

        # save files
        with open(os.path.join(base_path, f"{key}.txt"), "w") as f:
            f.writelines(file_list)

    # # ! only for test
    # print("Load test images.")
    # file_list = []
    # data_path = os.path.join(base_path, "color_images", "test")
    # for d in os.listdir(data_path):
    #     for f in os.listdir(os.path.join(data_path, d)):
    #         if not f.endswith(".png"):
    #             continue
    #         file_list.append(os.path.join("color_images", "test", d, f) + "\n")

    # with open(os.path.join(base_path, "test.txt"), "w") as f:
    #     f.writelines(file_list)

    return


def main():
    args = parse_args()
    base_path = args.path
    save_path = base_path
    
    data = {"train": "./labels/train/", "valid": "./labels/valid/"}
    prepare_labels(base_path, save_path, data)


if __name__ == "__main__":
    main()
