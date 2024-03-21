# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

""" Plotting functions """

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
from matplotlib.collections import LineCollection

from ws_crl.encoder.base import Inverse
from ws_crl.encoder.flow import SONEncoder
from ws_crl.utils import generate_directed_graph_matrix, get_first_batch

# Global constants
FONTSIZE = 11  # pt
PAGEWIDTH = 11  # inches


def init_plt():
    """Initialize matplotlib's rcparams to look good"""

    sns.set_style("whitegrid")

    matplotlib.rcParams.update(
        {
            # Font sizes
            "font.size": FONTSIZE,  # controls default text sizes
            "axes.titlesize": FONTSIZE,  # fontsize of the axes title
            "axes.labelsize": FONTSIZE,  # fontsize of the x and y labels
            "xtick.labelsize": FONTSIZE,  # fontsize of the tick labels
            "ytick.labelsize": FONTSIZE,  # fontsize of the tick labels
            "legend.fontsize": FONTSIZE,  # legend fontsize
            "figure.titlesize": FONTSIZE,  # fontsize of the figure title
            # Figure size and DPI
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "figure.figsize": (PAGEWIDTH / 2, PAGEWIDTH / 2),
            # colors
            "lines.markeredgewidth": 0.8,
            "axes.edgecolor": "black",
            "axes.grid": False,
            "grid.color": "0.9",
            "axes.grid.which": "both",
            # x-axis ticks and grid
            "xtick.bottom": True,
            "xtick.direction": "out",
            "xtick.color": "black",
            "xtick.major.bottom": True,
            "xtick.major.size": 4,
            "xtick.minor.bottom": True,
            "xtick.minor.size": 2,
            # y-axis ticks and grid
            "ytick.left": True,
            "ytick.direction": "out",
            "ytick.color": "black",
            "ytick.major.left": True,
            "ytick.major.size": 4,
            "ytick.minor.left": True,
            "ytick.minor.size": 2,
        }
    )


def plot_importance_matrix(cfg, metrics, latent_type: str, filename=None, artifact_folder=None):
    importance_matrix = generate_directed_graph_matrix(metrics, f"{latent_type}_importance_matrix_")

    fig, ax = plt.subplots()
    img = ax.imshow(importance_matrix)

    ax.set_xlabel("True")
    ax.set_ylabel("Model")
    ax.set_title(f"{latent_type.title()} Importance Matrix")
    ax.set_xticks(np.arange(cfg.data.dim_z))
    ax.set_yticks(np.arange(cfg.data.dim_z))
    ax.set_xticklabels(np.arange(cfg.data.dim_z))
    ax.set_yticklabels(np.arange(cfg.data.dim_z))
    cbar = ax.figure.colorbar(img, ax=ax)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
        mlflow.log_artifact(
            filename, artifact_folder if artifact_folder is not None else Path(filename).stem
        )
        plt.close(fig)
    else:
        fig.show()


def plot_latent_space(
    cfg,
    model,
    loader,
    MAP_interventions,
    device,
    components=[4, 5],
    num_batches=None,
    filename=None,
    artifact_folder=None,
):
    assert len(components) == 2

    fig, ax = plt.subplots(figsize=(10, 10))

    corresponding_components = [MAP_interventions[dim + 1] - 1 for dim in components]

    ax.set_title("Predicted noise encodings")

    ax.set_xlabel(f"$e_{components[0]}$")
    ax.set_ylabel(f"$e_{components[1]}$")

    if num_batches is None:
        num_batches = len(loader)

    for i, batch in enumerate(loader):
        x1, x2, _, _, intervention_labels, *_ = batch

        if len(x2.shape) > 2:
            x2 = x2[:, -1]

        x1, x2, intervention_labels = x1.to(device), x2.to(device), intervention_labels.to(device)

        e1_mean, e1_std = model.encoder.mean_std(x1)
        e2_mean, e2_std = model.encoder.mean_std(x2)

        intervened = torch.zeros_like(intervention_labels)
        for component in components:
            component_intervened = intervention_labels == component + 1
            intervened[component_intervened] = 1

        intervened = intervened.squeeze()

        colors = np.where(intervened.cpu().numpy() == 1, "red", "blue")

        e1_mean, e2_mean, e1_std, e2_std = (
            e1_mean.cpu().numpy(),
            e2_mean.cpu().numpy(),
            e1_std.cpu().numpy(),
            e2_std.cpu().numpy(),
        )

        # ax.scatter(e1_mean[:, corresponding_dims[0]], e1_mean[:, corresponding_dims[1]], c="b")
        ax.scatter(
            e2_mean[:, corresponding_components[0]],
            e2_mean[:, corresponding_components[1]],
            c=colors,
            alpha=0.3,
        )
        ax.set_xlabel(f"$e_{corresponding_components[0] + 1}$")
        ax.set_ylabel(f"$e_{corresponding_components[1] + 1}$")

        if i == num_batches - 1:
            break

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
        mlflow.log_artifact(
            filename, artifact_folder if artifact_folder is not None else Path(filename).stem
        )
        plt.close(fig)
    else:
        fig.show()


def plot_average_intervention_posterior(
    cfg, average_intervention_posterior, filename=None, artifact_folder=None
):
    num_interventions = cfg.data.dim_z + 1

    fig, ax = plt.subplots()

    img = ax.imshow(average_intervention_posterior.cpu().detach().numpy())
    ax.set_xticks(range(num_interventions))
    ax.set_yticks(range(num_interventions))
    ax.set_xticklabels(["empty"] + [f"$\widehat{{z_{i + 1}}}$" for i in range(cfg.data.dim_z)])
    ax.set_yticklabels(["empty"] + [f"$z_{i + 1}$" for i in range(cfg.data.dim_z)])
    ax.set_xlabel("Predicted Intervention on")
    ax.set_ylabel("True Intervention on")
    ax.set_title("Average Intervention Posterior")
    cbar = ax.figure.colorbar(img, ax=ax)
    cbar.ax.set_ylabel("Posterior", rotation=-90, va="bottom")

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
        mlflow.log_artifact(
            filename, artifact_folder if artifact_folder is not None else Path(filename).stem
        )
        plt.close(fig)
    else:
        fig.show()


def plot_noise_pairs(
    cfg,
    model,
    loader,
    MAP_interventions,
    device,
    after_proj=False,
    intervention_type="intervened",
    filename=None,
    artifact_folder=None,
):
    assert intervention_type in ["intervened", "unintervened", "all"]

    # remove empty intervention because it does not have a corresponding dimension in the latent space
    MAP_interventions = MAP_interventions[1:] - 1

    fig, axes = plt.subplots(
        nrows=cfg.data.dim_z,
        figsize=(10, 1.5 * cfg.data.dim_z),
        # sharex=True, sharey=True
    )

    title = "Noise encoding pairs $(e_1, \widehat{e_1})$ vs. $(e_2, \widehat{e_2})$"
    if after_proj:
        title += " after projection"
    title += f" ({intervention_type})"
    fig.suptitle(title)

    for component, ax in enumerate(axes):
        ax.set_title(
            f"$(e^{{({component+1})}}, \widehat{{e}}^{{({MAP_interventions[component] + 1})}})$"
        )
        ax.set_xlabel("True")
        ax.set_ylabel("Model")

    for batch in loader:
        x1, x2, z1, z2, intervention_labels, interventions, e1, e2 = batch

        x1, x2, z1, z2, e1, e2, intervention_labels = (
            x1.to(device),
            x2.to(device),
            z1.to(device),
            z2.to(device),
            e1.to(device),
            e2.to(device),
            intervention_labels.to(device),
        )

        (
            _,
            _,
            e1_mean,
            e2_mean,
            e1_proj,
            e2_proj,
            _,
            _,
            _,
        ) = model.encode_decode_pair(x1, x2)

        # Use last sequence element for sequence data
        if len(e2_mean.shape) > 2:
            e2_mean = e2_mean[:, -1]
        if len(e2_proj.shape) > 2:
            e2_proj = e2_proj[:, -1]

        e1, e2 = e1.cpu().numpy(), e2.cpu().numpy()
        e1_mean, e2_mean = e1_mean.cpu().numpy(), e2_mean.cpu().numpy()
        e1_proj, e2_proj = e1_proj.cpu().numpy(), e2_proj.cpu().numpy()
        intervention_labels = intervention_labels.squeeze().cpu().numpy()

        for component in range(cfg.data.dim_z):
            ax = axes[component]
            MAP_component = MAP_interventions[component]

            if after_proj:
                e1_hat = e1_proj
                e2_hat = e2_proj
            else:
                e1_hat = e1_mean
                e2_hat = e2_mean

            e1_component = e1[:, component]
            e2_component = e2[:, component]
            e1_hat_component = e1_hat[:, MAP_component]
            e2_hat_component = e2_hat[:, MAP_component]

            if intervention_type == "intervened":
                e1_component = e1_component[intervention_labels == component + 1]
                e2_component = e2_component[intervention_labels == component + 1]
                e1_hat_component = e1_hat_component[intervention_labels == component + 1]
                e2_hat_component = e2_hat_component[intervention_labels == component + 1]
            elif intervention_type == "unintervened":
                e1_component = e1_component[intervention_labels != component + 1]
                e2_component = e2_component[intervention_labels != component + 1]
                e1_hat_component = e1_hat_component[intervention_labels != component + 1]
                e2_hat_component = e2_hat_component[intervention_labels != component + 1]

            if (
                cfg.data.dim_z + 1 == 5
                or cfg.data.dim_z + 1 == 6
                and intervention_labels == component + 1
            ):
                pass

            lines = [
                [(_e1, _e1_hat), (_e2, _e2_hat)]
                for _e1, _e1_hat, _e2, _e2_hat in zip(
                    e1_component, e1_hat_component, e2_component, e2_hat_component
                )
            ]

            lc = LineCollection(lines, colors="black", linewidths=0.3, alpha=0.1)
            ax.add_collection(lc)

            ax.scatter(
                e1_component, e1_hat_component, s=0.5, label="Observational", color="red", alpha=0.1
            )
            ax.scatter(
                e2_component,
                e2_hat_component,
                s=0.5,
                label="Counterfactual",
                color="blue",
                alpha=0.1,
            )

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
        mlflow.log_artifact(
            filename, artifact_folder if artifact_folder is not None else Path(filename).stem
        )
        plt.close(fig)
    else:
        fig.show()


def plot_x(cfg, x, ax=None):
    fig_created = False
    if ax is None:
        fig, ax = plt.subplots()
        fig_created = True

    if "encoder" in cfg.data:
        if cfg.data.encoder.type == "son":
            encoder = SONEncoder(input_features=cfg.data.dim_x, output_features=cfg.data.dim_z)
            encoder.load_state_dict(torch.load(Path(cfg.data.data_dir) / "encoder.pt"))

            decoder = Inverse(encoder)

            encoder.to(x.device)
            decoder.to(x.device)

            x, _ = decoder(x.view(1, x.size(0)))
            x = x[0]

    if cfg.data.type == "xy_pairs":
        # data is concatenated xy pairs, i.e. [x1, y1, x2, y2, ...]
        x = x.cpu().numpy()
        ax.scatter(x[::2], x[1::2])

        # add margins
        ax.margins(0.1, 0.1)
    elif cfg.data.type == "image":
        ax.imshow(x.permute(1, 2, 0))
        ax.axis("off")
    else:
        raise ValueError(f"Unknown data type: {cfg.data.type}")

    if fig_created:
        plt.close(fig)


def plot_reconstructions(
    cfg, model, loader, device, n_samples=12, filename=None, artifact_folder=None
):
    nrows = n_samples
    ncols = 4  # x1, x1_reco, x2, x2_reco

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 2 * nrows))

    # Get data
    x1s, x2s, _, _, true_interventions, *_ = get_first_batch(loader)
    x1s, x2s, true_interventions = x1s.to(device), x2s.to(device), true_interventions.to(device)
    x1s, x2s, true_interventions = x1s[:n_samples], x2s[:n_samples], true_interventions[:n_samples]

    if len(x2s.shape) > 2:
        x2s = x2s[:, -1]

    x1s_reco = model.encode_decode(x1s)
    x2s_reco = model.encode_decode(x2s)

    for i in range(n_samples):
        plot_x(cfg, x1s[i], ax=axes[i, 0])
        plot_x(cfg, x1s_reco[i], ax=axes[i, 1])
        plot_x(cfg, x2s[i], ax=axes[i, 2])
        plot_x(cfg, x2s_reco[i], ax=axes[i, 3])

        axes[i, 0].set_xlabel("Before intervention, true")
        axes[i, 1].set_xlabel("Before intervention, reconstructed")

        axes[i, 2].set_xlabel("After intervention, true")
        axes[i, 3].set_xlabel("After intervention, reconstructed")

    for i in range(nrows):
        for j in range(ncols):
            # axes[i, j].axis("off")
            pass

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
        mlflow.log_artifact(
            filename, artifact_folder if artifact_folder is not None else Path(filename).stem
        )
        plt.close(fig)
    else:
        fig.show()


def plot_counterfactuals(
    cfg,
    model,
    loader,
    device,
    n_samples=12,
    intervention_scale=1.0,
    filename=None,
    artifact_folder=None,
):
    nrows = n_samples
    ncols = 2 + cfg.model.dim_z  # x, empty intervention, e1, e2, ...
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows))

    # Get data
    x1s, *_ = get_first_batch(loader)
    x1s = x1s.to(device)

    eps1s = model.encode_to_noise(x1s, deterministic=True)
    eps1s_std = eps1s.std(dim=0)

    eps1s = eps1s[:n_samples]
    x1s_reco = model.decode_noise(eps1s, deterministic=True)

    x1s_intervened = torch.zeros((n_samples, cfg.model.dim_z, cfg.model.dim_x)).to(device)

    for i in range(len(eps1s_std)):
        intervention_i = torch.zeros((1, cfg.model.dim_z)).to(device)
        intervention_i[:, i] = eps1s_std[i] * intervention_scale

        eps1s_intervened_i = eps1s + intervention_i
        x1s_intervened[:, i] = model.decode_noise(eps1s_intervened_i, deterministic=True)

    for i in range(n_samples):
        plot_x(cfg, x1s[i], ax=axes[i, 0])
        plot_x(cfg, x1s_reco[i], ax=axes[i, 1])

        for j in range(cfg.model.dim_z):
            plot_x(cfg, x1s_intervened[i, j], ax=axes[i, 2 + j])

        axes[i, 0].set_xlabel("$x$")
        axes[i, 1].set_xlabel("$\widehat{x}$")
        for j in range(cfg.model.dim_z):
            axes[i, 2 + j].set_xlabel(f"$\widehat{{x}}_{{intervened, {j + 1}}}$")

    for i in range(nrows):
        for j in range(ncols):
            # axes[i, j].axis("off")
            pass

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
        mlflow.log_artifact(
            filename, artifact_folder if artifact_folder is not None else Path(filename).stem
        )
        plt.close(fig)
    else:
        fig.show()
