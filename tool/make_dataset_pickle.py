from pathlib import Path
from util.io import save_pickle
import pandas as pd


def make_v47_gt(dir_root, path_gt):
    gt = pd.read_csv(path_gt, sep="\s+")
    all_gts = []
    for idx, data in gt.iterrows():
        print(idx, data)
        label = data["Image_No"]
        lty = data["Start_Row"]
        rby = data["End_Row"]
        ltx = data["Start_Col"]
        rbx = data["End_Col"]
        tmp = str(path_gt).replace(str(dir_root), "").replace("Bounding Box", "Images").replace("\\", "/").replace(".txt", "")[1:]
        path_img = tmp + "/" + str(label) + ".bmp"
        cx = 0.5 * (ltx + rbx)
        cy = 0.5 * (lty + rby)
        w = rbx - ltx
        h = rby - lty
        bb = [ltx, lty, rbx, rby]

        gt = {
            "img_name": path_img,
            "bb": bb}

        all_gts += [gt]

    return all_gts


def make_v47_dataset(dir_root, path_out):
    dir_root = Path(dir_root)
    path_gts = list(dir_root.glob("**/*.txt"))

    all_gts = []
    for path_gt in path_gts:
        all_gts += make_v47_gt(dir_root, path_gt)

    save_pickle(path_out, all_gts)


if __name__ == "__main__":
    dir_root = "../dataset/v47"
    path_dst = Path(dir_root, "v47.pkl")
    make_v47_dataset(dir_root, path_dst)
