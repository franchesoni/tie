import sys
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import jit, njit, prange
import torch


@jit(nopython=True)
def compute_cosine_terrain(A, T, tindices, times, phases, minns, maxxs, img):
    terrain = np.zeros((A, T))
    # for phase_ind in tqdm(range(A)):
    for phase_ind in range(A):
        phase = phases[phase_ind]
        # for min_ind in tqdm(range(T), leave=False):
        for min_ind in range(T):
            minn = minns[min_ind]
            for max_ind in range(min_ind, T):
                maxx = maxxs[max_ind]
                ys = (maxx - minn) / 2 * np.cos(2 * np.pi * (times + phase)) + (
                    maxx + minn
                ) / 2
                ys = (ys * T).astype(np.int16)
                score = 0
                for i, y in enumerate(ys):
                    score += img[tindices[i], y]
                score /= len(ys)
                for i, y in enumerate(ys):
                    terrain[tindices[i], y] = np.maximum(terrain[tindices[i], y], score)
    return terrain


def compute_energy(img, ball_size=5):
    """Compute energy, assume second dim is time"""
    assert ball_size % 2 == 1, "Ball size must be odd"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # build the terrain
    timg = torch.from_numpy(img)
    terrain = torch.cat(
        (torch.zeros(1, img.shape[1]), timg, torch.zeros(1, img.shape[1])), dim=0
    ).to(device)
    timg = timg.to(device)
    terrain_down = terrain.clone()
    terrain_up = terrain.clone()
    maxpool = torch.nn.MaxPool1d(ball_size, stride=1, padding=ball_size // 2)
    for row_ind in range(1, terrain.shape[0] - 1):  # forward
        # now the terrain has the max value that can be achieved from top to the current row
        terrain_down[row_ind] = (
            maxpool(terrain_down[row_ind - 1 : row_ind])[0] + terrain_down[row_ind]
        )
    for row_ind in range(terrain.shape[0] - 2, 0, -1):  # backward
        # now the terrain has the max value that can be achieved from bottom to the current row
        terrain_up[row_ind] = (
            maxpool(terrain_up[row_ind + 1 : row_ind + 2])[0] + terrain_up[row_ind]
        )
    terrain = (terrain_up + terrain_down)[
        1:-1
    ] - timg  # remove the padding and the original image$
    terrain = terrain / terrain.shape[1]  # mean node on path
    return terrain.cpu().numpy()


def savee(img, name):
    plt.imsave(f"out/{name}.png", img, cmap="gray")


def inference(trr):
    st = time.time()
    trr = trr / trr.max()

    A, R, T = trr.shape

    depth_terrains = []
    for a in tqdm(range(A)):
        depth_terrain = compute_energy(trr[a])
        depth_terrains.append(depth_terrain)
    dtrr = np.array(depth_terrains)

    ctrr = cosine_terrain(trr, A, R, T)

    trr = dtrr * ctrr
    final_trr = []
    for a in tqdm(range(A)):
        final_trr.append(compute_energy(trr[a]))
    final_trr = np.array(final_trr)
    final_trr = final_trr / final_trr.max()
    paths = final_trr == final_trr.max(axis=2, keepdims=True)
    print(f"Inference took {time.time() - st:.2e} s")
    return final_trr, paths

def cosine_terrain(trr, A, R, T):
    cosine_terrains = []
    tindices = np.arange(A)
    times = tindices / A
    phases = np.arange(A) / A
    minns = np.arange(T) / T
    maxxs = np.arange(T) / T

    cosine_terrains = numba_loop(trr, A, R, T, tindices, times, phases, minns, maxxs)
    ctrr = cosine_terrains.transpose(1, 0, 2)
    return ctrr

@njit(parallel=True)
def numba_loop(trr, A, R, T, tindices, times, phases, minns, maxxs):
    cosine_terrains = np.zeros((R, A, T))
    for r in prange(R):
        if r % (R // 10) == 0:
            print('.')
        # we will create a new energy map
        img = trr[:, r]
        terrain = compute_cosine_terrain(
            A, T, tindices, times, phases, minns, maxxs, img
        )
        cosine_terrains[r] = terrain
    return cosine_terrains


if __name__ == "__main__":
    trr = np.load("uimgs.npy")
    final_trr, paths = inference(trr)
    np.save("final_trr.npy", final_trr)
    np.save("paths.npy", paths)
