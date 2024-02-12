import tqdm
from typing import Tuple
import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os


def read_well(data_path, partition, i, return_first_frame_only):
    data = pd.read_csv(os.path.join(data_path, "metaData.csv"))
    data = data.loc[data["Partition"] == partition]

    well_name = data.iloc[i]["Name_Well"]
    file_name = data.iloc[i]["Name_file"]
    labels_tie = data.iloc[i]["TIE Label"]
    labels_tie_list = sorted(labels_tie.strip("][").split(", "))
    labels_tie_list = [x.replace("'", "") for x in labels_tie_list]

    frames = []
    for frame in tqdm.tqdm(labels_tie_list):
        name_frame = frame.split("_")[0]
        image = prepare_data(name_frame, os.path.join(data_path, well_name, file_name))
        if return_first_frame_only:
            return [image]
        frames.append(image)
    return frames


def prepare_data(name_frame: str, path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function reads and preprocesses the data and labels

    Args:
        name_frame (str): name of the image ('UF...')
        path (str): path of the image

    Returns:
        image : processed image
        label : processed label with 3 classes (bg, tie, casing)
    """
    # Load data and labels
    image = np.load(os.path.join(path, name_frame + ".npy"))
    tdep = np.load(path + "/" + "TDEP.npy")

    #  Sort the depths
    if np.all((np.sort(tdep) == tdep)):
        tdep = tdep
    else:
        image = np.flipud(image)
        tdep = np.flipud(tdep)

    return image


def showimg(img, save=True, name=""):
    if len(img.shape) in [2, 3]:
        if save:
            plt.imsave(f"out/{name}.png", img[:400])
        else:
            plt.figure(figsize=(10, 10))
            plt.imshow(img[:400])
            if len(name):
                plt.title(name)
            plt.axis("off")
    elif len(img.shape) == 1:
        plt.figure()
        plt.plot(img)
        if len(name):
            plt.title(name)
        if save:
            plt.savefig(f"out/{name}.png")
    plt.close()


def process(raw_img, vis=True, inv_width=40, var_thresh=1e-6):
    # remove empty rows
    vars = np.var(raw_img, axis=1) > var_thresh  # one per row
    raw_img = raw_img[vars]

    # fill rows that aren't close to the mean with the mean
    profile = raw_img.sum(axis=0)  # get profile
    if vis:
        showimg(profile, name="S0_profile")
    norm_profile = (profile - profile.min()) / (profile - profile.min()).sum()
    corrs = (
        (raw_img - raw_img.min()) / (raw_img - raw_img.min()).sum(axis=1, keepdims=True)
    ) @ norm_profile
    hist, bin_edges = np.histogram(corrs, bins=100)
    for i in range(len(hist) - 1, -1, -1):  # not very efficient
        if hist[i] == 0:
            break
    corr_thresh = bin_edges[i - 1]
    bad_row_indices = corrs < corr_thresh
    raw_img[bad_row_indices] = raw_img.mean(axis=0)
    if vis:
        showimg(raw_img, name="S1_badrows")

    # remove first casing, assuming it's very high and very low
    min_at = np.argmin(profile)  # where the min is
    max_at = np.argmax(profile)  # where the max is
    if max_at < min_at:
        cut_at = min_at + (min_at - max_at) // 2  # use min and max to get a width param
    else:
        cut_at = max_at + (max_at - min_at) // 2
    img = raw_img[
        :, cut_at + 1 :
    ]  # crop a little to the right of it (then we should crop the right side so that all angles have the same shape)

    if vis:
        showimg(img, name="S2_cut")

    # now enhance by substracting the mean per column (remove vertical reflections, this assumes the casing is always perfectly vertical)
    img = img - img.mean(axis=0)
    if vis:
        showimg(img, name="S3_rmvertical")

    # now half wave rectify
    img = np.abs(img - img.mean())
    if vis:
        showimg(img, name="S4_abs")

    # gaussian blur on the horizontal dimension
    img = scipy.ndimage.gaussian_filter1d(img, axis=1, sigma=img.shape[1] // inv_width)
    if vis:
        showimg(img, name="S5_gauss")
    return img, cut_at, vars


def unprocess(img, cut_at, vars):
    """Undoes the processing by adding zeroed pixels at the positions that were removed."""
    uimg = np.zeros((len(vars), img.shape[1] + cut_at + 1))
    count = 0
    for row_ind, var in enumerate(vars):
        if var:
            uimg[row_ind, cut_at + 1 :] = img[count]
            count += 1
    return uimg



def main(
    # data_path="/HomeToo/data/curated/TIE",
    # partition="train",
    # i=0,
    no_vis=False,
):
    vis = not no_vis

    # read image
    # frames = read_well(data_path, partition, i, return_first_frame_only=True)
    frames = np.array(np.load("frames.npy"))  # locally
    # frames is [angles, depth, time]
    uimgs = np.zeros_like(frames, dtype=float)
    for find, raw_img in tqdm.tqdm(enumerate(frames)):
        # visualize
        if vis: showimg(raw_img, name="rawimg")
        img, cut_at, vars = process(raw_img, vis=True)
        if vis: showimg(img, name="processed")
        uimg = unprocess(img, cut_at, vars)
        uimgs[find] = uimg

    np.save('uimgs.npy', uimgs)
    
if __name__ == "__main__":
    from fire import Fire

    Fire(main)
