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
        split="labeled",
    ):
        split_path = Path(data_path) / split
        image_paths = sorted(list((split_path / "images").glob("*.npy")))
        print("loading y_train...")
        if split == "labeled":
            y_train_cache = Path("tmp") / "y_train.pkl"
            y_train_cache.parent.mkdir(exist_ok=True, parents=True)
            if y_train_cache.exists():
                y_train = pd.read_pickle(y_train_cache)
            else:
                y_train = pd.read_csv(split_path / "Y_train_T9NrBYo.csv", index_col=0)
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
                if split == "labeled":
                    label = np.array(y_train.loc[image_path.stem])[
                        : image.size
                    ].reshape(160, -1)
                else:
                    label = -1
                if image.shape != (160, 272):
                    orig_w = image.shape[1]
                    image = np.concatenate(
                        (image, np.zeros((160, 272 - orig_w), dtype=image.dtype)),
                        axis=1,
                    )
                    if split == "labeled":
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


def log_metrics(writer, prefix, loss, loss_casing, loss_tie, mae_casing, mae_tie, global_step):
    """Helper function to log metrics to TensorBoard"""
    writer.add_scalar(f"{prefix}/total", loss.item(), global_step=global_step)
    writer.add_scalar(f"{prefix}/casing", loss_casing.item(), global_step=global_step)
    writer.add_scalar(f"{prefix}/tie", loss_tie.item(), global_step=global_step)
    writer.add_scalar(f"{prefix}/mae_casing", mae_casing.item(), global_step=global_step)
    writer.add_scalar(f"{prefix}/mae_tie", mae_tie.item(), global_step=global_step)


def train_epoch(model, optim, dataloader, device, writer, epoch, batch_size, vis_val=False):
    """Helper function to train for one epoch"""
    for batch_idx, batch in enumerate(dataloader):
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
            f"epoch {epoch} step {batch_idx+1} / {len(dataloader)}, "
            f"casing={loss_casing:.4f}, tie={loss_tie:.4f}, loss={loss:.4f}",
            end="\r",
        )
        
        global_step = batch_idx * batch_size + epoch * len(dataloader)
        log_metrics(writer, "train", loss, loss_casing, loss_tie, mae_casing, mae_tie, global_step)
    
    print(
        f"epoch {epoch} step {batch_idx+1} / {len(dataloader)}, "
        f"casing={loss_casing:.4f}, tie={loss_tie:.4f}, loss={loss:.4f}"
    )


def validate_epoch(model, dataloader, device, writer, epoch, train_dl_len, vis_val=False):
    """Helper function to validate for one epoch"""
    total_val_loss, total_val_casing, total_val_tie = 0.0, 0.0, 0.0
    
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), desc="validating..."):
        image_paths, image_batch, label_batch = batch
        image_batch, label_batch = image_batch.to(
            device, non_blocking=True
        ), label_batch.to(device, non_blocking=True)
        B = image_batch.shape[0]
        
        with torch.no_grad():
            val_loss, val_loss_casing, val_loss_tie, val_mae_casing, val_mae_tie = get_loss(
                image_batch, label_batch, model, image_paths, vis_val
            )
            total_val_loss += val_loss * B / len(dataloader.dataset)
            total_val_casing += val_loss_casing * B / len(dataloader.dataset)
            total_val_tie += val_loss_tie * B / len(dataloader.dataset)
    
    global_step = epoch * train_dl_len
    log_metrics(writer, "val", total_val_loss, total_val_casing, total_val_tie, 
                val_mae_casing, val_mae_tie, global_step)


def setup_model_and_optim(n_layers, device):
    """Helper function to setup model and optimizer"""
    model = Conv1DNet(n_layers=n_layers).to(device)
    optim = torch.optim.AdamW(model.parameters(), weight_decay=1)
    return model, optim


def main_train(
    tag="baseline", 
    batch_size=32, 
    epochs=16, 
    n_layers=4, 
    num_workers=16, 
    cross_validation=False,
    vis_val=False
):
    """
    Unified training function that supports both regular training and cross-validation.
    
    Args:
        tag: Tag for tensorboard logging
        batch_size: Batch size for training
        epochs: Number of epochs (16 for regular training, 10 for CV by default)
        n_layers: Number of layers in the model
        num_workers: Number of workers for data loading
        cross_validation: If True, performs cross-validation across wells
        vis_val: If True, saves visualization images during validation
    """
    device = torch.device("cuda:0")
    all_wells = list(range(1, 7))
    
    # Adjust default epochs for cross-validation if not explicitly set
    if cross_validation and epochs == 16:
        epochs = 10

    if not cross_validation:
        # Regular training mode - train on all wells
        writer = SummaryWriter(comment=tag)
        wells = [str(w) for w in all_wells]
        train_ds = TIEDataset(wells=wells)
        dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        model, optim = setup_model_and_optim(n_layers, device)
        
        for epoch in range(epochs):
            train_epoch(model, optim, dl, device, writer, epoch, batch_size, vis_val=False)
        
        print("saving model...")
        torch.save(model.state_dict(), Path(writer.log_dir) / "final_weights.pth")
        print("Training done, model saved, great job!")
    
    else:
        # Cross-validation mode - train on n-1 wells, validate on 1 well
        for test_well in all_wells:
            writer = SummaryWriter(comment="_" + tag + "_" + f"test_well_{test_well}")
            wells = [str(w) for w in all_wells if w != test_well]
            train_ds = TIEDataset(wells=wells)
            test_ds = TIEDataset(wells=[str(test_well)])

            dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            model, optim = setup_model_and_optim(n_layers, device)

            for epoch in range(epochs):
                # Validation phase
                validate_epoch(model, test_dl, device, writer, epoch, len(dl), vis_val)
                
                # Training phase
                train_epoch(model, optim, dl, device, writer, epoch, batch_size, vis_val)


# Legacy wrapper functions for backward compatibility
def main_cv(tag, vis_val=False):
    """Legacy function - use main_train with cross_validation=True instead"""
    main_train(tag=tag, cross_validation=True, vis_val=vis_val)


def visualize(well, section, patch=None, split="labeled"):
    def vis_single(well, section, patch, split, outname):
        image = np.load(
            f"data/{split}/images/well_{well}_section_{section}_patch_{patch}.npy"
        )
        plt.imsave(outname, np_pct_minmax_norm(image))

    if patch is None:
        patch = 0
        while True:
            try:
                vis_single(
                    well, section, patch, split, outname=f"tmp/vis_patch_{patch}.png"
                )
                patch += 1
            except:
                break
    else:
        vis_single(well, section, patch, split, outname=f"tmp/vis.png")


def inference(ckpt_path, kernel_size=1, vis=False):
    model = Conv1DNet(n_layers=4)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    model = model.to("cuda")
    ds = TIEDataset(split="test")
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=16)
    predictions = dict()
    pred_patches = {"casing": dict(), "tie": dict()}
    with torch.no_grad():
        for batch_idx, batch in tqdm.tqdm(enumerate(dl), desc="inference..."):
            image_paths, image_batch, label_batch = batch
            image_batch = image_batch.to("cuda", non_blocking=True).float()
            B = image_batch.shape[0]
            out = (
                model(image_batch.reshape(B * 160, 1, 272))
                .reshape(B, 160, 2, 272)
                .permute(0, 1, 3, 2)
            )
            out = torch.softmax(out, dim=2)  # (B, H, W, 2)
            # save this one for futher processing
            for batch_ind in range(len(out)):
                pred_patches["casing"][image_paths[batch_ind]] = (
                    out[batch_ind, :, :, 0].cpu().numpy()
                )
                pred_patches["tie"][image_paths[batch_ind]] = (
                    out[batch_ind, :, :, 1].cpu().numpy()
                )
            out_label = out == torch.max(out, dim=2, keepdim=True)[0]
            out_label = torch.clamp(
                out_label[..., 0] * 2 + out_label[..., 1] * 1, min=0, max=2
            )  # don't allow 3s
            out_label = torch.nn.MaxPool2d(
                kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size // 2)
            )(out_label.float().unsqueeze(1)).squeeze(1)
            # try to follow the expected save format
            if (image_batch[:, :, 160:] == 0).all():
                # the image is small
                out_label = out_label[:, :160, :160]
            out_label = out_label.cpu().numpy()
            for ind_in_batch, image_path in enumerate(image_paths):
                predictions[image_path] = -np.ones(160 * 272)
                if (image_batch[ind_in_batch, :, 160:] == 0).all():
                    predictions[image_path][: 160 * 160] = out_label[
                        ind_in_batch, :160, :160
                    ].flatten()
                else:
                    predictions[image_path] = out_label[ind_in_batch].flatten()
                if vis:
                    bigimg = np.concatenate(
                        (
                            np_pct_minmax_norm(image_batch[ind_in_batch].cpu().numpy()),
                            np_pct_minmax_norm(out_label[ind_in_batch]),
                        ),
                        axis=1,
                    )
                    plt.imsave(f"tmp/test/{image_path}.png", bigimg)
        print("saving...")
        np.save("tmp/pred_patches.npy", pred_patches)
        pd.DataFrame(predictions, dtype="int").T.to_csv(Path("y_test_csv_file.csv"))
        print("saved! great!")


if __name__ == "__main__":
    from fire import Fire

    # Fire(main_train)
    # Fire(main_cv)
    # Fire(visualize)
    # Fire(inference)
