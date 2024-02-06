import tqdm
from typing import Tuple
import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
from sing import SING


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
    if vis: showimg(profile, name="S0_profile")
    norm_profile = (profile - profile.min()) / (profile - profile.min()).sum()
    corrs = ((raw_img - raw_img.min()) / (raw_img - raw_img.min()).sum(axis=1, keepdims=True)) @ norm_profile
    hist, bin_edges = np.histogram(corrs, bins=100)
    for i in range(len(hist)-1, -1, -1):  # not very efficient
        if hist[i] == 0:
            break
    corr_thresh = bin_edges[i-1]
    bad_row_indices = corrs < corr_thresh
    raw_img[bad_row_indices] = raw_img.mean(axis=0)
    if vis: showimg(raw_img, name="S1_badrows")
    
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
    uimg = np.zeros((len(vars), img.shape[1]+cut_at+1))
    count = 0
    for row_ind, var in enumerate(vars):
        if var:
            uimg[row_ind, cut_at+1:] = img[count]
            count += 1
    return uimg

class PredFromLogits(torch.nn.Module):
    def __init__(self, norm="softmax", eps=1):
        assert norm in [
            "softmax",
            "factor",
        ], f"norm is {norm} but should be in ['softmax', 'factor']"
        super().__init__()
        self.norm = norm
        self.eps = eps

    def forward(self, logits):
        x = torch.nn.functional.softplus(logits)
        x = x + self.eps
        if self.norm == "softmax":
            x = torch.nn.functional.softmax(x, dim=1)
        elif self.norm == "factor":
            x = x / x.sum(dim=1, keepdims=True)
        return x


# all processing steps
class OurLoss(torch.nn.Module):
    def __init__(self, scale_corr=1, scale_cont=1, penalize_after=100, max_drift=None):
        """When max_drift is None, use correlation continuity loss instead"""
        super().__init__()
        self.scale_corr = scale_corr
        self.scale_cont = scale_cont
        self.penalize_after = penalize_after
        self.max_drift = max_drift

    def forward(self, x, timg):
        loss_corr = (
            -(timg * x).sum(dim=1).mean()
        )  # maximize the mean row correlation between image and estimation
        # continuity loss
        if self.max_drift is not None:
            ## first version (barycenters)
            row_coords = torch.linspace(0, 1, timg.shape[1]).to(x.device)
            barycenters = (x * row_coords[None]).sum(
                dim=1
            )  # compute the barycenter per row
            diffs = barycenters[1:] - barycenters[:-1]
            big_diffs = torch.relu(diffs - 2 / timg.shape[1])
            loss_cont = (
                -1 + torch.exp(big_diffs)
            ).sum()  # penalize big diffs between consecutive barycenters
        else:
            # second version (correlations)
            corrs = (x[1:] * x[:-1]).sum(dim=1).sum()
            loss_cont = -corrs

        # final loss
        loss = loss_corr * self.scale_corr + loss_cont * self.scale_cont
        return loss, {
            "corr": (loss_corr * self.scale_corr).item(),
            "cont": (loss_cont * self.scale_cont).item(),
        }


# Define the seed everything function
def seed_everything(seed):
    # Set the seed for pytorch, numpy, and python.random
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Set the seed for torch.cuda if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSED"] = str(seed)


import time


def fit(
    img,
    iterations=1000,
    max_lr=1,
    print_every_pct=0.02,
    seed=0,
    scale=100,
    init_as_img=True,
    optim="sing",
    norm="softmax",
    norm_eps=0,
    scale_cont=1,
    scale_corr=1,
    max_drift=1,
    init_logits=None,
):
    st = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(seed)
    timg = torch.from_numpy(img)
    timg = scale * (timg - timg.min()) / (timg.max() - timg.min())
    # initialize as the same image
    if init_as_img:
        logits = timg.clone()
    else:  # init as noise
        logits = torch.rand(timg.shape)
    if init_logits is not None:
        logits = init_logits
    logits = logits.requires_grad_(True)
    logits.to(device)

    div_factor = 25  # default of onecyclelr
    optim_class = {"sing": SING, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}[
        optim
    ]
    optim = optim_class([logits], lr=max_lr / div_factor)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=max_lr,
        total_steps=iterations,
        pct_start=0.05,
        div_factor=div_factor,
    )
    print_every = int(iterations * print_every_pct)

    model = PredFromLogits(norm=norm, eps=norm_eps)
    loss_fn = OurLoss(
        scale_cont=scale_cont,
        scale_corr=scale_corr,
        penalize_after=scale,
        max_drift=max_drift,
    )

    print(f"training on {device}")
    losses, min_loss = [], np.inf
    for i in range(iterations):
        optim.zero_grad(set_to_none=True)
        x = model(logits)
        magnitude_penalty = (
            torch.abs(x) * (torch.abs(x) > scale)
        ).sum()  # penalize really large values
        loss, comp = loss_fn(x, timg)
        loss = loss + magnitude_penalty
        loss.backward()
        optim.step()
        scheduler.step()
        losses.append(loss.item())

        if i % print_every == 0 or i == iterations - 1:
            with torch.no_grad():
                z = model(x)
                z = z.detach().cpu().numpy()
            if i == iterations - 1:
                if losses[-1] < min_loss:
                    min_loss = losses[-1]
                showimg(np.log1p(np.array(losses) - min_loss), name="loss")
                showimg(z, name=f"pred")
            print(
                f"{i} / {iterations}, Loss={loss.item()}, {' '.join([f'{key}={val:.3e}' for key,val in comp.items()])}",
                end="\r",
            )

    print()
    print("Optimization took", time.time() - st, "s")
    return z, logits


def find_dag_path(img):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # build the terrain
    terrain = torch.cat((torch.zeros(1, img.shape[1]), torch.from_numpy(img)), dim=0).to(device)
    maxpool = torch.nn.MaxPool1d(3, stride=1, padding=1)
    for row_ind in range(1, terrain.shape[0]):
        terrain[row_ind] = maxpool(terrain[row_ind-1:row_ind])[0] + terrain[row_ind]
    terrain = terrain[1:]
    # now find the path
    path = [torch.argmax(terrain[-1])]
    for row_ind in range(terrain.shape[0]-2, -1, -1):
        last = path[-1]
        if last == 0:
            path.append(torch.argmax(terrain[row_ind, :2]))
        elif last == terrain.shape[1]-1:
            path.append(torch.argmax(terrain[row_ind, -2:])+terrain.shape[1]-2)
        else:
            path.append(torch.argmax(terrain[row_ind, last-1:last+2])+last-1)
    path = path[::-1]
    # build the path image
    path_img = torch.zeros_like(terrain)
    path_img[range(len(path)), path] = 1
    return path_img
    

def main(
    # data_path="/HomeToo/data/curated/TIE",
    # partition="train",
    # i=0,
    no_vis=False,
    iterations=1000,
    init_as_img=False,
    max_lr=1,
    scale=1,
    seed=0,
    optim="sing",
    norm="softmax",
    norm_eps=0,
    scale_cont=1,
    scale_corr=1,
    max_drift=1,
):
    vis = not no_vis

    # read image
    # frames = read_well(data_path, partition, i, return_first_frame_only=True)
    frames = np.load('frames.npy')  # locally

    raw_img = frames[0]
    # visualize
    if vis: showimg(raw_img, name="rawimg")
    img, cut_at, vars = process(raw_img, vis=True)
    if vis: showimg(img, name="processed")

    x = find_dag_path(img)
    showimg(x, name="path")
    x2 = unprocess(x, cut_at, vars)
    showimg(raw_img + (2*raw_img.max()) * x2, name="pred")
    # showimg(np.exp(img) / np.sum(np.exp(img), axis=1, keepdims=True), name="softmax")
    # x, logits = fit(
    #     img,
    #     iterations=iterations,
    #     init_as_img=init_as_img,
    #     max_lr=max_lr,
    #     scale=scale,
    #     seed=seed,
    #     optim=optim,
    #     norm=norm,
    #     norm_eps=norm_eps,
    #     scale_cont=scale_cont,
    #     scale_corr=scale_corr,
    #     max_drift=max_drift,
    #     init_logits=None
    # )
    # showimg(x, name="pred")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
