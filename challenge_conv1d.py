import torch
import pandas as pd
from pathlib import Path
import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class TIEDataset(Dataset):
    def __init__(
        self,
        data_path="data",
        wells="all",
        sections="all",
        patches="all",
    ):
        train_path = Path(data_path) / "labeled"
        image_paths = sorted(list((train_path / "images").glob("*.npy")))
        print("loading y_train...")
        y_train_cache = Path("tmp") / "y_train.pkl"
        y_train_cache.parent.mkdir(exist_ok=True, parents=True)
        if y_train_cache.exists():
            y_train = pd.read_pickle(y_train_cache)
        else:
            y_train = pd.read_csv(train_path / "Y_train_T9NrBYo.csv", index_col=0)
            y_train.to_pickle(y_train_cache)

        self.samples = []
        print("loading samples...")
        for image_path in tqdm.tqdm(image_paths):
            _, well, _, section, _, patch = image_path.stem.split("_")
            if (
                ((wells == "all") or (well in wells))
                and ((sections == "all") or (section in sections))
                and ((patches == "all") or (patch in patches))
            ):
                image = np.load(image_path)
                label = np.array(y_train.loc[image_path.stem])[: image.size].reshape(
                    160, -1
                )
                if image.shape != (160, 272):
                    orig_w = image.shape[1]
                    image = np.concatenate(
                        (image, np.zeros((160, 272 - orig_w), dtype=image.dtype)),
                        axis=1,
                    )
                    label = np.concatenate(
                        (label, np.zeros((160, 272 - orig_w), dtype=label.dtype)),
                        axis=1,
                    )
                self.samples.append((image_path.stem, image, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, image, label = self.samples[idx]
        image = np_pct_minmax_norm(image)
        return image_path, image, label


def torch_pct_minmax_norm(data, pct=3):
    """
    Performs min-max normalization using percentiles to reduce the impact of outliers.
    """
    lower_pct = pct
    upper_pct = 100 - pct

    # Calculate percentiles instead of min/max
    mins = torch.quantile(data, lower_pct / 100)
    maxs = torch.quantile(data, upper_pct / 100)

    # Scale to feature range
    scaled_data = (data - mins) / (maxs - mins)

    # Clip values to respect the feature range
    return torch.clip(scaled_data, 0, 1)


def np_pct_minmax_norm(data, pct=3):
    """
    Performs min-max normalization using percentiles to reduce the impact of outliers.
    """
    lower_pct = pct
    upper_pct = 100 - pct

    # Calculate percentiles instead of min/max
    mins = np.percentile(data, lower_pct)
    maxs = np.percentile(data, upper_pct)

    # Scale to feature range
    scaled_data = (data - mins) / (maxs - mins)

    # Clip values to respect the feature range
    return np.clip(scaled_data, 0, 1)


class Conv1DNet(torch.nn.Module):
    def __init__(
        self, in_channels=1, out_channels=2, n_layers=1, hidden_dim=32, kernel_size=16
    ):
        """
        Neural network with n_layers 1D convolutions

        Args:
            in_channels: Number of input channels
            out_channels: Number of output classes (0=background, 1=tie, 2=casing)
            n_layers: Number of 1D convolutional layers
            hidden_dim: Hidden dimension size for each conv layer
        """
        super().__init__()

        if n_layers == 1:
            self.model = torch.nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding="same"
            )
        else:
            model_layers = []
            model_layers.append(
                torch.nn.Conv1d(
                    in_channels, hidden_dim, kernel_size=kernel_size, padding="same"
                )
            )
            model_layers.append(torch.nn.ReLU())
            if n_layers > 2:
                for layer in range(n_layers - 2):
                    # First layer: input to hidden_dim
                    model_layers.append(
                        torch.nn.Conv1d(
                            hidden_dim,
                            hidden_dim,
                            kernel_size=kernel_size,
                            padding="same",
                        )
                    )
                    model_layers.append(torch.nn.ReLU())

            # Last layer doesn't have ReLU
            model_layers.append(
                torch.nn.Conv1d(
                    hidden_dim, out_channels, kernel_size=kernel_size, padding="same"
                )
            )
            self.model = torch.nn.Sequential(*model_layers)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
        """
        return self.model(x)


def get_loss(image_batch, label_batch, model, image_paths, vis_val):
    B, H, W = image_batch.shape
    in_tensor = image_batch.reshape(B * H, 1, W).float()
    out = model(in_tensor).reshape(B, H, 2, W).permute(0, 1, 3, 2)  # (B, H, W, 2)
    out_casing = torch.softmax(out[..., 0], dim=2)  # (B, H, W)
    out_tie = torch.softmax(out[..., 1], dim=2)
    mask_casing = (label_batch == 2) * 1.0
    mask_tie = (label_batch == 1) * 1.0
    # cross entropy loss
    loss_casing = torch.nn.functional.binary_cross_entropy(out_casing, mask_casing)
    loss_tie = torch.nn.functional.binary_cross_entropy(out_tie, mask_tie)
    loss = loss_casing + loss_tie
    # mae
    with torch.no_grad():
        col_indices = torch.arange(W).reshape(1, 1, W).to(mask_casing.device)
        col_casing = (mask_casing * col_indices).sum(dim=2) / (mask_casing > 0).sum(
            dim=2
        )  # (B, H)
        col_tie = (mask_tie * col_indices).sum(dim=2) / (mask_tie > 0).sum(
            dim=2
        )  # (B, H)
        out_col_casing = (out_casing * col_indices).sum(dim=2)  # (B, H) expected col
        out_col_tie = (out_tie * col_indices).sum(dim=2)  # (B, H) expected col
        mae_casing = torch.nanmean(torch.abs(col_casing - out_col_casing))
        mae_tie = torch.nanmean(torch.abs(col_tie - out_col_tie))
    if vis_val:
        for sample_ind in range(B):
            img_path, image, label = (
                image_paths[sample_ind],
                image_batch[sample_ind].cpu().numpy(),
                label_batch[sample_ind].cpu().numpy(),
            )
            casing_pred, tie_pred = (
                out_casing[sample_ind].detach().cpu().numpy(),
                out_tie[sample_ind].detach().cpu().numpy(),
            )
            bigimg = np.hstack(
                [
                    np.vstack([image, np_pct_minmax_norm(label)]),
                    np.vstack(
                        [np_pct_minmax_norm(casing_pred), np_pct_minmax_norm(tie_pred)]
                    ),
                ]
            )
            plt.imsave(f"tmp/{img_path}.png", np_pct_minmax_norm(bigimg))

    return loss, loss_casing, loss_tie, mae_casing, mae_tie


def main_train(tag="baseline", batch_size=32, epochs=16, n_layers=4, num_workers=16):
    device = torch.device("cuda:0")
    all_wells = list(range(1, 7))
    writer = SummaryWriter(comment=tag)
    wells = [str(w) for w in all_wells]
    train_ds = TIEDataset(wells=wells)
    dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    model = Conv1DNet(n_layers=n_layers).to(device)
    optim = torch.optim.AdamW(model.parameters(), weight_decay=1)
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dl):
            image_paths, image_batch, label_batch = batch
            image_batch, label_batch = image_batch.to(
                device, non_blocking=True
            ), label_batch.to(device, non_blocking=True)
            loss, loss_casing, loss_tie, mae_casing, mae_tie = get_loss(
                image_batch, label_batch, model, image_paths, vis_val=False
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(
                f"epoch {epoch} step {batch_idx+1} / {len(dl)}, casing={loss_casing}, tie={loss_tie}, loss={loss}",
                end="\r",
            )
            writer.add_scalar(
                "train/total",
                loss.item(),
                global_step=batch_idx * batch_size + epoch * len(dl),
            )
            writer.add_scalar(
                "train/casing",
                loss_casing.item(),
                global_step=batch_idx * batch_size + epoch * len(dl),
            )
            writer.add_scalar(
                "train/tie",
                loss_tie.item(),
                global_step=batch_idx * batch_size + epoch * len(dl),
            )
            writer.add_scalar(
                "train/mae_casing",
                mae_casing.item(),
                global_step=batch_idx * batch_size + epoch * len(dl),
            )
            writer.add_scalar(
                "train/mae_tie",
                mae_tie.item(),
                global_step=batch_idx * batch_size + epoch * len(dl),
            )
        print(
            f"epoch {epoch} step {batch_idx+1} / {len(dl)}, casing={loss_casing}, tie={loss_tie}, loss={loss}"
        )
    print("saving model...")
    torch.save(Path(writer.log_dir) / "final_weights.pth", model.state_dict())
    print("Training done, model saved, great job!")


def main_cv(tag, vis_val=False):
    # classes are 0=background, 1=tie, 2=casing
    device = torch.device("cuda:0")
    allowed_wells = [
        1,
        2,
        3,
        4,
        5,
        6,
    ]  # there are 6 wells numbered from 1 to 6 inclusive, but not all have the same shape
    for test_well in allowed_wells:
        writer = SummaryWriter(comment="_" + tag + "_" + f"test_well_{test_well}")
        wells = [str(w) for w in allowed_wells if w != test_well]
        train_ds = TIEDataset(wells=wells)
        test_ds = TIEDataset(wells=[str(test_well)])

        batch_size = 32
        dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=16)
        test_dl = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=16
        )
        model = Conv1DNet(n_layers=4).to(device)
        optim = torch.optim.AdamW(model.parameters(), weight_decay=1)
        epochs = 10

        for epoch in range(epochs):
            total_val_loss, total_val_casing, total_val_tie = 0.0, 0.0, 0.0
            for batch_idx, batch in tqdm.tqdm(enumerate(test_dl), desc="validating..."):
                image_paths, image_batch, label_batch = batch
                image_batch, label_batch = image_batch.to(
                    device, non_blocking=True
                ), label_batch.to(device, non_blocking=True)
                B = image_batch.shape[0]
                with torch.no_grad():
                    (
                        val_loss,
                        val_loss_casing,
                        val_loss_tie,
                        val_mae_casing,
                        val_mae_tie,
                    ) = get_loss(image_batch, label_batch, model, image_paths, vis_val)
                    total_val_loss += val_loss * B / len(test_ds)
                    total_val_casing += val_loss_casing * B / len(test_ds)
                    total_val_tie += val_loss_tie * B / len(test_ds)
            writer.add_scalar(
                "val/total",
                total_val_loss.item(),
                global_step=epoch * len(dl),
            )
            writer.add_scalar(
                "val/casing",
                total_val_casing.item(),
                global_step=epoch * len(dl),
            )
            writer.add_scalar(
                "val/tie",
                total_val_tie.item(),
                global_step=epoch * len(dl),
            )
            writer.add_scalar(
                "val/mae_casing", val_mae_casing.item(), global_step=epoch * len(dl)
            )
            writer.add_scalar(
                "val/mae_tie", val_mae_tie.item(), global_step=epoch * len(dl)
            )

            for batch_idx, batch in enumerate(dl):
                image_paths, image_batch, label_batch = batch
                image_batch, label_batch = image_batch.to(
                    device, non_blocking=True
                ), label_batch.to(device, non_blocking=True)
                loss, loss_casing, loss_tie, mae_casing, mae_tie = get_loss(
                    image_batch, label_batch, model, image_paths, vis_val
                )
                optim.zero_grad()
                loss.backward()
                optim.step()
                print(
                    f"epoch {epoch} step {batch_idx+1} / {len(dl)}, casing={loss_casing}, tie={loss_tie}, loss={loss}",
                    end="\r",
                )
                writer.add_scalar(
                    "train/total",
                    loss.item(),
                    global_step=batch_idx * batch_size + epoch * len(dl),
                )
                writer.add_scalar(
                    "train/casing",
                    loss_casing.item(),
                    global_step=batch_idx * batch_size + epoch * len(dl),
                )
                writer.add_scalar(
                    "train/tie",
                    loss_tie.item(),
                    global_step=batch_idx * batch_size + epoch * len(dl),
                )
                writer.add_scalar(
                    "train/mae_casing",
                    mae_casing.item(),
                    global_step=batch_idx * batch_size + epoch * len(dl),
                )
                writer.add_scalar(
                    "train/mae_tie",
                    mae_tie.item(),
                    global_step=batch_idx * batch_size + epoch * len(dl),
                )
            print(
                f"epoch {epoch} step {batch_idx+1} / {len(dl)}, casing={loss_casing}, tie={loss_tie}, loss={loss}"
            )


def visualize(well, section, patch, split="labeled"):

    image = np.load(
        f"data/{split}/images/well_{well}_section_{section}_patch_{patch}.npy"
    )
    print("image shape =", image.shape)
    plt.imsave("tmp/vis.png", np_pct_minmax_norm(image))


if __name__ == "__main__":
    from fire import Fire

    Fire(main_train)
    # Fire(main_cv)
