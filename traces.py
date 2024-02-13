
#%%
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy
import torch

def compute_energy_right(img):
    """Compute energy, assume second dim is time"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # build the terrain
    timg = torch.from_numpy(img)
    terrain = torch.cat(
        (torch.zeros(1, img.shape[1]), timg, torch.zeros(1, img.shape[1])), dim=0
    ).to(device)
    timg = timg.to(device)
    terrain_down = terrain.clone()
    terrain_up = terrain.clone()
    maxpool = torch.nn.MaxPool1d(2, stride=1, padding=1)
    for row_ind in range(1, terrain.shape[0] - 1):  # forward
        # now the terrain has the max value that can be achieved from top to the current row
        terrain_down[row_ind] = (
            maxpool(terrain_down[row_ind - 1 : row_ind])[0][:-1] + terrain_down[row_ind]
        )
    for row_ind in range(terrain.shape[0] - 2, 0, -1):  # backward
        # now the terrain has the max value that can be achieved from bottom to the current row
        terrain_up[row_ind] = (
            maxpool(terrain_up[row_ind + 1 : row_ind + 2])[0][1:] + terrain_up[row_ind]
        )
    terrain = (terrain_up[1:-1] + terrain_down[1:-1]) - timg  # remove the padding and the original image$
    terrain = terrain / terrain.shape[1]  # mean node on path
    return terrain.cpu().numpy()

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




def align(img, path):
    H, W = img.shape
    target_column = W // 2  # Target column to align the ones
    
    # Create a copy of the img to avoid modifying the original
    aligned_img = np.copy(img)
    
    for row in range(H):
        row_path = path[row, :]
        if 1 in row_path:
            # Find the index of the first '1' in the row
            current_column = np.where(row_path == 1)[0][0]
            # Calculate the amount of shift needed
            shift_amount = target_column - current_column
            # Roll this row by the calculated shift amount
            aligned_img[row, :] = np.roll(aligned_img[row, :], shift_amount)
    
    return aligned_img

# Example usage:
# Ensure img and path are NumPy arrays of the same shape (H, W)
# img = np.array(...) # Your image array
# path = np.array(...) # Your path variable, same shape as img
# aligned_img = align_image_horizontally_with_roll(img, path)


#%%
import time
datapath = Path('/CurrentProjects/data/curated/1_click_VSP/hDVS/DOUGHTIE_OTHER_4')
filepath = datapath / 'Traces.npy'
print('loading...')
raw_img = - np.load(filepath)
plt.imsave('raw.png', raw_img)
print('processing...')
rowmin, rowmax = raw_img.min(axis=1, keepdims=True), raw_img.max(axis=1, keepdims=True)
norm_img = (raw_img - rowmin) / (rowmax - rowmin)
plt.imsave('img_norm.png', norm_img)
img = norm_img - np.mean(norm_img)
img = np.clip(img, 0, img.max())
# plt.imsave('img_half.png', img)
# img = scipy.ndimage.gaussian_filter1d(img, axis=1, sigma=3)
# plt.imsave('processed.png', img)
print('computing energy...')
st = time.time()
energy = compute_energy_right(img)
print('took', time.time() - st)
plt.imsave('energy.png', energy)
print('getting path...')
path = energy == energy.max(axis=1, keepdims=True)
combined = norm_img[..., None] * np.ones((1,1,3)) + path[..., None] * np.array([[[1, 0, 0]]])
plt.imsave('path.png', path)
plt.imsave('raw_with_path.png', 0.5 * combined)
aligned = align(img, path)
plt.imsave('aligned.png', aligned)
#%%
profile = aligned.sum(axis=0)
norm_profile = (profile - profile.min()) / (profile - profile.min()).sum()
maxpos = np.argmax(norm_profile)
norm_profile[:maxpos-20] = 0
norm_profile[maxpos+20:] = 0
plt.switch_backend('WebAgg')

aligned2 = aligned.copy()
for r, row in enumerate(aligned2):
    # find max corr
    maxcorr = 0
    for shift in range(-20, 20):
        newrow = np.roll(row, shift) # normalized shifted row
        corr = newrow @ norm_profile
        if corr > maxcorr:
            maxcorr = corr
            bestshift = shift
    aligned2[r] = np.roll(row, bestshift)

plt.close('all')
plt.figure()
fig, ax = plt.subplots(1, 4, sharex=True, sharey=True)
ax[0].imshow(aligned)
ax[1].imshow(aligned2)
ax[2].imshow(norm_img)
ax[3].imshow(aligned2*norm_profile[None])
plt.show()

plt.imsave('aligned2.png', aligned)


breakpoint()
# %%
