import tqdm
from typing import Tuple
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.morphology import skeletonize
from cosines import inference


def read_well(data, data_path, i, return_first_frame_only, return_labels=False):

    well_name = data.iloc[i]["Name_Well"]
    file_name = data.iloc[i]["Name_file"]
    labels_tie = data.iloc[i]["TIE Label"]
    labels_tie_list = sorted(labels_tie.strip("][").split(", "))
    labels_tie_list = [x.replace("'", "") for x in labels_tie_list]

    frames = []
    if return_labels:
        casings, ties = [], []
    for frame in tqdm.tqdm(labels_tie_list):
        name_frame = frame.split("_")[0]
        if return_labels:
            image, casing, tie = prepare_data_with_labels(
                name_frame, os.path.join(data_path, well_name, file_name)
            )
        else:
            image = prepare_data(
                name_frame, os.path.join(data_path, well_name, file_name)
            )
        if return_first_frame_only:
            if return_labels:
                return [image], [casing], [tie]
            return [image]
        frames.append(image)
        if return_labels:
            casings.append(casing)
            ties.append(tie)
    if return_labels:
        return frames, casings, ties
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


def prepare_data_with_labels(
    name_frame: str, path: str
) -> Tuple[np.ndarray, np.ndarray]:
    casing_name = os.path.join(path, name_frame + "_casing_label.npy")
    tie_name = os.path.join(path, name_frame + "_label.npy")
    image = np.load(os.path.join(path, name_frame + ".npy"))
    tdep = np.load(path + "/" + "TDEP.npy")
    casing_original = np.load(casing_name).astype(np.int64)
    tie = np.load(tie_name).astype(np.int64)

    #  Sort the depths
    if np.all((np.sort(tdep) == tdep)):
        tdep = tdep
    else:
        image = np.flipud(image)
        tdep = np.flipud(tdep)

    return image, casing_original, tie


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
    if vis: showimg(raw_img, name='Start_raw')
    # remove empty rows
    vars = np.var(raw_img, axis=1) > var_thresh  # one per row
    raw_img = raw_img[vars]

    raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min())  # take to [0,1]
    # fill rows that aren't close to the mean with the mean
    profile = raw_img.sum(axis=0)  # get profile
    if vis:
        showimg(profile, name="S0_profile")
    norm_profile = (profile - profile.min()) / (profile - profile.min()).sum()
    # align rows to profile
    for r, row in enumerate(raw_img):
        # find max corr
        maxcorr = 0
        row = row / np.sum(row)
        for shift in range(-15, 16):
            newrow = np.roll(row, shift) # normalized shifted row
            corr = newrow @ norm_profile
            if corr > maxcorr:
                maxcorr = corr
                bestshift = shift
        raw_img[r] = np.roll(row, bestshift)
    corrs = (
        (raw_img - raw_img.min()) / (raw_img - raw_img.min()).sum(axis=1, keepdims=True)
    ) @ norm_profile
    corr_thresh = np.percentile(corrs, 5)
    bad_row_indices = corrs < corr_thresh
    raw_img[bad_row_indices] = raw_img.mean(axis=0)
    if vis:
        showimg(raw_img, name="S1_badrows")

    # remove first casing, assuming it's very high and very low
    plen = profile.shape[0]  # assume casing is on the left half of the image
    min_at = np.argmin(profile[: plen // 2])  # where the min is
    max_at = np.argmax(profile[: plen // 2])  # where the max is
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
    sigma = img.shape[1] // inv_width
    if sigma > 0:
        img = scipy.ndimage.gaussian_filter1d(img, axis=1, sigma=sigma)
    if vis:
        showimg(img, name="S5_gauss")

    # attenuate extreme values 
    img = np.sqrt(img)
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


def avg_displacement_error(paths, ties):
    paths, ties = np.array(paths), np.array(ties)
    A, R, T = paths.shape
    assert paths.shape == ties.shape
    pred_missing_rows, target_missing_rows = 0, 0
    errors = []
    for a in range(A):
        tie_img = skeletonize(ties[a])
        path_img = skeletonize(paths[a])
        aerrors = []
        for r in range(R):
            line = tie_img[r]
            linesum = line.sum()
            if linesum == 0:
                target_missing_rows += 1
                continue  # skip if no ground truth
            elif linesum > 2:
                print(f"Warning! tie has too many targets at {a, r}!", end="\r")
                offset = linesum // 2
                continue
            else:
                offset = 0
            pathline = path_img[r]
            pathsum = pathline.sum()
            if pathsum == 0:
                pred_missing_rows += 1
            pred = np.argmax(pathline)
            target = np.argmax(line) + offset
            displacement_error = np.abs(pred - target)
            aerrors.append(displacement_error)
        # print(f"a = {a}, error = {np.mean(aerrors)}")
        errors += aerrors

    return np.mean(errors), pred_missing_rows, target_missing_rows


def main(
    data_path="/HomeToo/data/curated/TIE",
    partition="train",
    no_vis=False,
    skip_prediction=False,
):
    vis = not no_vis

    print("reading metadata..." + " " * 40, end="\r")
    data = pd.read_csv(os.path.join(data_path, "metaData.csv"))
    data = data.loc[data["Partition"] == partition]
    if not skip_prediction:
        # make predictions
        # for i in range(len(data)):
        for i in [0, 1, 2, 4, 6, 3, 5]:  # slow results later
            print(f"working {i+1}/{len(data)}...")
            print("reading..." + " " * 40, end="\r")
            frames, casings, ties = read_well(
                data, data_path, i, return_first_frame_only=False, return_labels=True
            )
            frames = np.array(frames)
            # frames is [angles, depth, time]
            print("preprocessing..." + " " * 40, end="\r")
            uimgs = np.zeros_like(frames, dtype=float)
            for find, raw_img in tqdm.tqdm(enumerate(frames)):
                # visualize
                if vis:
                    showimg(raw_img, name="rawimg")
                img, cut_at, vars = process(raw_img, vis=vis)
                if vis:
                    showimg(img, name="processed")
                uimg = unprocess(img, cut_at, vars)
                uimgs[find] = uimg
            print("predicting..." + " " * 40, end="\r")
            trr, paths = inference(uimgs)
            print("saving..." + " " * 40, end="\r")
            np.save(
                f"everything_{partition}_{i}.npy",
                {
                    "trr": trr,
                    "paths": paths,
                    "frames": frames,
                    "uimgs": uimgs,
                    "casings": casings,
                    "ties": ties,
                },
            )

    for i in range(len(data)):
        print("-" * 20, i, "-" * 20)
        print("loading..." + " " * 40, end="\r")
        everything = np.load(
            f"everything_{partition}_{i}.npy", allow_pickle=True
        ).item()
        # print('computing error...')
        error, missing_preds, missing_targets = avg_displacement_error(
            everything["paths"], everything["ties"]
        )
        showimg(everything["uimgs"][0], name=f"{partition}_{i}_" + "uimgs0")
        showimg(everything["uimgs"][17], name=f"{partition}_{i}_" + "uimgs17")
        showimg(everything["frames"][0], name=f"{partition}_{i}_" + "frame0")
        showimg(everything["frames"][17], name=f"{partition}_{i}_" + "frame17")
        showimg(everything["trr"][0], name=f"{partition}_{i}_" + "trr0")
        showimg(everything["trr"][17], name=f"{partition}_{i}_" + "trr17")
        showimg(everything["ties"][0], name=f"{partition}_{i}_" + "ties0")
        showimg(everything["ties"][17], name=f"{partition}_{i}_" + "ties17")
        showimg(everything["paths"][0], name=f"{partition}_{i}_" + "paths0")
        showimg(everything["paths"][17], name=f"{partition}_{i}_" + "paths17")

        print("average displacement =", error)
        print("missing preds =", missing_preds)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
    # partition = 'train'
    # i = 3
    # tutto = np.load(
    #     f"everything_{partition}_{i}.npy", allow_pickle=True
    # ).item()
    # error, missing_preds, missing_targets = avg_displacement_error(
    #     tutto["paths"], tutto["ties"]
    # )
    # critical_frame = 0
    # breakpoint()
    # frames = tutto['frames']
    # img, cut_at, vars = process(frames[critical_frame], vis=True)
    # uimg = unprocess(img, cut_at, vars)
    # showimg(frames[critical_frame], name='critical_frame')
    # showimg(uimg, name='critical_uimg')
    # showimg(tutto['ties'][critical_frame], name='critical_ties')
    # showimg(tutto['trr'][critical_frame], name='critical_trr')
    # showimg(tutto['paths'][critical_frame], name='critical_paths')
    # showimg(uimg / uimg.max() + tutto['paths'][critical_frame], name='critical_paths2')
    # breakpoint()