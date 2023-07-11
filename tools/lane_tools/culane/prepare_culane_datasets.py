import os
import ujson
import cv2
import argparse
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preparation")
    parser.add_argument("path", type=str, help="CULane dataset root path")
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
    for key, value in data.items():
        json_img_path = os.path.join(save_path, f"json_lanes_{key}")
        if not os.path.exists(json_img_path):
            os.makedirs(json_img_path)
        print(f"Preparing {key} label data...")
        # vis_img_path = os.path.join(save_path, f"vis_lanes_{key}")
        # if not os.path.exists(vis_img_path):
        #     os.makedirs(vis_img_path)

        with open(os.path.join(base_path, value), "r") as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line_data = line.strip().split(" ")
            img = cv2.imread(os.path.join(base_path, "." + line_data[0]))
            img_h, img_w, _ = img.shape
            lane_dict = {
                "raw_file": "." + line_data[0],
                "h_samples": np.arange(0, img_h, 1).tolist(),
            }

            if key != "test":
                assert len(line_data) == 6
                lane_dict["seg_file"] = "." + line_data[1]
                lane_dict["label_lane"] = list(map(int, line_data[2:]))

            with open(
                os.path.join(
                    base_path, lane_dict["raw_file"].replace(".jpg", ".lines.txt")
                ),
                "r",
            ) as f:
                lane_points = f.readlines()
            lane_points = [
                list(map(float, points.strip().split(" "))) for points in lane_points
            ]

            lane_dict["lanes"] = []
            for lane in lane_points:
                x_points, y_points = lane[0::2], lane[1::2]
                inter_y = np.arange(y_points[-1], y_points[0] + 1, 1)
                inter_x = np.interp(inter_y, y_points[::-1], x_points[::-1])

                x_coor = [-2] * img_h
                for x, y in zip(inter_x, inter_y):
                    x, y = int(x), int(y)
                    if x < 0 or x >= img_w or y < 0 or y >= img_h:
                        continue
                    x_coor[y] = x
                lane_dict["lanes"].append(x_coor)

            out_file = open(
                os.path.join(
                    json_img_path,
                    lane_dict["raw_file"].replace("/", "_").replace(".jpg", ".json"),
                ),
                "w",
            )
            ujson.dump(lane_dict, out_file, escape_forward_slashes=False)
            # img_show(img, lane_dict['lanes'], lane_dict['h_samples'], lane_dict['raw_file'].replace('/', '_'))
    return


def main():
    args = parse_args()
    base_path = args.path
    save_path = base_path

    data = {
        "train": "./list/train_gt.txt",
        "val": "./list/val_gt.txt",
        "test": "./list/test.txt",
    }
    prepare_labels(base_path, save_path, data)


if __name__ == "__main__":
    main()
