import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from challenge_conv1d import np_pct_minmax_norm



def load(data, well, section, class_name="tie"):
    # visualize
    patch = 0
    patches = []
    while True:
        name = f"well_{well}_section_{section}_patch_{patch}"
        if name in data[class_name]:
            patches.append(data[class_name][name])
            patch += 1
        else:
            break
    patches = np.concatenate(patches, axis=0)
    return patches


def main(well=7, section=0, w=40):
    # first compute pred patches using challenge_conv1d.py, then postprocess with this notebook
    # load probs predictions
    data = np.load("tmp/pred_patches.npy", allow_pickle=True).item()
    # load images
    data["image"] = dict()
    for image_path in tqdm.tqdm(Path("data/test/images").glob("*.npy")):
        image = np_pct_minmax_norm(np.load(image_path))
        data["image"][image_path.stem] = image
    # process show
    tie = load(data, well, section, class_name="tie")
    casing = load(data, well, section, class_name="casing")
    tie_adj, casing_adj = compute_conditional_probabilities_batched(casing, tie, w=w)

    plt.imsave("tmp0.png", np.concatenate((tie, tie_adj, tie_adj - tie), axis=1))
    plt.imsave(
        "tmp1.png", np.concatenate((casing, casing_adj, casing_adj - casing), axis=1)
    )
    plt.imsave("tmp2.png", load(data, well, section, class_name="image"))


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
