# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

""" Plotting functions """

from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mlflow
import networkx as nx
import numpy as np
import seaborn as sns
import torch
from matplotlib.collections import LineCollection
from omegaconf import OmegaConf

from ws_crl.encoder.base import Inverse
from ws_crl.encoder.flow import SONEncoder
from ws_crl.encoder.image_vae import CoordConv2d, ImageSBDecoder
from ws_crl.utils import generate_directed_graph_matrix, get_first_batch, remove_prefix
from spatialvaes.vaes import SpatialBroadcastVAE
import json

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
            # SVGs
            "svg.fonttype": "none",
        }
    )


def save_plot(fig, filename, artifact_folder=None, save_to_mlflow=True, close_fig=True):
    fig.savefig(filename)
    if save_to_mlflow:
        mlflow.log_artifact(
            filename, artifact_folder if artifact_folder is not None else Path(filename).stem
        )
    if close_fig:
        plt.close(fig)


def plot_metric_spread(
    cfg,
    metric_dict,
    metric_name,
    metric_label=None,
    x_jitter_std=False,
    boxplot=False,
    plot_mean=True,
    plot_fails=False,
    filename=None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    metric_dict = {k: np.array(v[metric_name]) for k, v in metric_dict.items()}

    if plot_fails:
        # plot NaNs
        for i, (model, metrics) in enumerate(metric_dict.items()):
            nan_indices = np.argwhere(np.isnan(metrics))
            if len(nan_indices) > 0:
                x = np.ones_like(metrics[nan_indices]) * (i + 1)
                fail_value = 0.0
                ax.plot(x, fail_value, "x", color="red", label=f"{model} (NaN)")

    # filter out NaNs
    metric_dict = {k: v[~np.isnan(v)] for k, v in metric_dict.items()}

    if boxplot:
        ax.boxplot(metric_dict.values(), labels=metric_dict.keys(), showmeans=plot_mean)
    else:
        if plot_mean:
            for i, (model, metrics) in enumerate(metric_dict.items()):
                ax.plot(i + 1, np.mean(metrics), "^", color="black")

        # add x labels
        ax.set_xticks(range(1, len(metric_dict) + 1))
        ax.set_xticklabels(metric_dict.keys())
    for i, (model, metrics) in enumerate(metric_dict.items()):
        if x_jitter_std and x_jitter_std > 0.0:
            x = np.random.normal(i + 1, x_jitter_std, size=len(metrics))
        else:
            x = np.ones_like(metrics) * (i + 1)
        ax.plot(x, metrics, "o", alpha=0.5, label=model)

    ax.set_xlabel("Model")
    ax.set_ylabel(metric_label if metric_label is not None else metric_name)
    ax.legend()

    if filename is not None:
        save_plot(fig, filename=filename, save_to_mlflow=False)
    return fig, ax


def plot_importance_matrix(
    cfg, metrics, latent_type: str, filename=None, artifact_folder=None, save_to_mlflow=True
):
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
        save_plot(fig, filename, artifact_folder, save_to_mlflow=save_to_mlflow)
    else:
        fig.show()
    return fig, ax


def plot_latent_space(
    cfg,
    model,
    loader,
    MAP_interventions,
    device,
    causal=False,
    num_batches=10,
    filename=None,
    artifact_folder=None,
    save_to_mlflow=True,
    component_rows_to_plot=None,
):
    component_rows = (
        range(cfg.model.dim_z) if component_rows_to_plot is None else component_rows_to_plot
    )

    nrows = len(component_rows)
    ncols = cfg.model.dim_z
    fig, axes = plt.subplots(
        figsize=(ncols * 4, nrows * 4),
        nrows=nrows,
        ncols=ncols,
    )
    for i, ci in enumerate(component_rows):
        for cj in range(cfg.model.dim_z):
            ax = axes[i, cj]
            plot_latent_space_components(
                cfg,
                model,
                loader,
                MAP_interventions,
                device,
                causal=causal,
                num_batches=num_batches,
                components=[ci, cj],
                ax=ax,
            )

    fig.tight_layout()
    if filename is not None:
        save_plot(fig, filename, artifact_folder, save_to_mlflow=save_to_mlflow)
    else:
        fig.show()
    return fig, axes


def plot_latent_space_components(
    cfg,
    model,
    loader,
    MAP_interventions,
    device,
    components,
    causal=False,
    num_batches=None,
    ax=None,
    filename=None,
    artifact_folder=None,
    save_to_mlflow=True,
):
    assert len(components) == 2

    fig_created = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        fig_created = True

    corresponding_components = [MAP_interventions[dim + 1] - 1 for dim in components]

    ax.set_title(f"Predicted {'noise' if not causal else 'causal'} encodings")

    latent_label = "e" if not causal else "z"
    # ax.set_xlabel(f"${latent_label}_{corresponding_components[0] + 1}$")
    # ax.set_ylabel(f"${latent_label}_{corresponding_components[1] + 1}$")
    ax.set_xlabel(f"${latent_label}_{components[0] + 1}$")
    ax.set_ylabel(f"${latent_label}_{components[1] + 1}$")

    if num_batches is None:
        num_batches = len(loader)

    for i, batch in enumerate(loader):
        x1, x2, _, _, intervention_labels, *_ = batch

        if len(x2.shape) > 2:
            x2 = x2[:, -1]

        x1, x2, intervention_labels = x1.to(device), x2.to(device), intervention_labels.to(device)

        if not causal:
            e1 = model.encode_to_noise(x1, deterministic=True)
            e2 = model.encode_to_noise(x2, deterministic=True)
        else:
            e1 = model.encode_to_causal(x1, deterministic=True)
            e2 = model.encode_to_causal(x2, deterministic=True)

        intervened = torch.zeros_like(intervention_labels)
        for component in components:
            component_intervened = intervention_labels == component + 1
            intervened[component_intervened] = 1

        intervened = intervened.squeeze()

        colors = np.where(intervened.cpu().numpy() == 1, "red", "blue")

        e1, e2 = (
            e1.cpu().numpy(),
            e2.cpu().numpy(),
        )

        # ax.scatter(e1_mean[:, corresponding_dims[0]], e1_mean[:, corresponding_dims[1]], c="b")
        ax.scatter(
            e2[:, corresponding_components[0]],
            e2[:, corresponding_components[1]],
            c=colors,
            alpha=0.3,
        )

        if i == num_batches - 1:
            break

    if fig_created:
        fig.tight_layout()
        if filename is not None:
            save_plot(fig, filename, artifact_folder, save_to_mlflow=save_to_mlflow)
        else:
            fig.show()


def plot_average_intervention_posterior(
    cfg,
    average_intervention_posterior,
    filename=None,
    artifact_folder=None,
    save_to_mlflow=True,
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
        save_plot(fig, filename, artifact_folder, save_to_mlflow=save_to_mlflow)
    else:
        fig.show()
    return fig, ax


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
    save_to_mlflow=True,
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
        save_plot(fig, filename, artifact_folder, save_to_mlflow=save_to_mlflow)
    else:
        fig.show()


def plot_x(cfg, x, ax=None, x1=None, plot_arrows=False):
    fig_created = False
    if ax is None:
        fig, ax = plt.subplots()
        fig_created = True

    if cfg and "encoder" in cfg.data:
        if cfg.data.encoder.type == "son":
            encoder = SONEncoder(input_features=cfg.data.dim_x, output_features=cfg.data.dim_z)
            encoder.load_state_dict(torch.load(Path(cfg.data.data_dir) / "encoder.pt"))

            decoder = Inverse(encoder)

            encoder.to(x.device)
            decoder.to(x.device)

            x, _ = decoder(x.view(1, x.size(0)))
            x = x[0]

            if x1 is not None:
                x1, _ = decoder(x1.view(1, x1.size(0)))
                x1 = x1[0]
        elif cfg.data.encoder.type == "sbd":
            exp_dir = Path(cfg.data.encoder.exp_dir)
            decoder_cfg_path = exp_dir / "config.yml"

            model_path = exp_dir / "models" / cfg.data.encoder.model_name

            decoder_cfg = OmegaConf.load(decoder_cfg_path)

            decoder = ImageSBDecoder(
                in_features=decoder_cfg.model.dim_z,
                out_resolution=decoder_cfg.model.dim_x[0],
                out_features=decoder_cfg.model.dim_x[2],
                hidden_features=decoder_cfg.model.decoder.hidden_channels,
                min_std=decoder_cfg.model.decoder.min_std,
                fix_std=decoder_cfg.model.decoder.fix_std,
                conv_class=(
                    CoordConv2d
                    if decoder_cfg.model.decoder.coordinate_embeddings
                    else torch.nn.Conv2d
                ),
                mlp_layers=decoder_cfg.model.decoder.extra_mlp_layers,
                mlp_hidden=decoder_cfg.model.decoder.extra_mlp_hidden_units,
                elementwise_hidden=decoder_cfg.model.decoder.elementwise_hidden_units,
                elementwise_layers=decoder_cfg.model.decoder.elementwise_layers,
                permutation=decoder_cfg.model.encoder.permutation,
            )
            state_dict = torch.load(model_path)
            state_dict_decoder = {
                remove_prefix(k, "decoder."): v
                for k, v in state_dict.items()
                if k.startswith("decoder.")
            }
            decoder.load_state_dict(state_dict_decoder)
            decoder.to(x.device)

            x, _ = decoder(x.unsqueeze(0), deterministic=True)
            x = x[0]
        elif cfg.data.encoder.type == "spatialvaes.sbd":
            exp_dir = Path(cfg.data.encoder.exp_dir)
            state_dict_path = exp_dir / cfg.data.encoder.model_name
            config_path = exp_dir / "config.json"

            with open(config_path) as f:
                conf = json.load(f)

            vae = SpatialBroadcastVAE.from_config(conf)
            vae.load_state_dict(torch.load(state_dict_path))

            decoder = vae.decoder
            decoder.to(x.device)

            if (
                cfg.data.x1_mean is not None
                and cfg.data.x2_mean is not None
                and cfg.data.x1_std is not None
                and cfg.data.x2_std is not None
            ):
                x1_mean = torch.tensor(cfg.data.x1_mean)
                x1_std = torch.tensor(cfg.data.x1_std)
                x2_mean = torch.tensor(cfg.data.x2_mean)
                x2_std = torch.tensor(cfg.data.x2_std)

                x_mean = (x1_mean + x2_mean) / 2
                x_std = (x1_std + x2_std) / 2

                x = x * x_std.to(x.device) + x_mean.to(x.device)

            x = x.unsqueeze(0)
            x = x.view(x.size(0), x.size(1), 1, 1)
            x = decoder(x)
            x = x[0].permute(1, 2, 0)

    if not cfg or cfg.data.type == "image":
        x = x.clamp(0, 1).mul(255).cpu().to(torch.uint8)
        ax.imshow(x)
        ax.axis("off")
    elif cfg.data.type == "xy_pairs":
        # data is concatenated xy pairs, i.e. [x1, y1, x2, y2, ...]

        # x1_mean = torch.tensor(cfg.data.x1_mean)
        # x1_std = torch.tensor(cfg.data.x1_std)
        # x2_mean = torch.tensor(cfg.data.x2_mean)
        # x2_std = torch.tensor(cfg.data.x2_std)

        # x_mean = (x1_mean + x2_mean) / 2
        # x_std = (x1_std + x2_std) / 2

        # x = x * x_std.to(x.device) + x_mean.to(x.device)

        x = x.cpu().numpy()

        assert len(x) % 2 == 0

        color_pool = list(mcolors.BASE_COLORS.keys())

        if len(x) // 2 < len(color_pool):
            colors = color_pool[: len(x) // 2]
        else:
            colors = None

        ax.scatter(x[::2], x[1::2], c=colors)

        if x1 is not None:
            x1 = x1.cpu().numpy()
            ax.scatter(x1[::2], x1[1::2], c=colors, marker="x", alpha=0.5)

        if plot_arrows:
            assert x1 is not None
            for i in range(len(x) // 2):
                ax.arrow(
                    x1[2 * i],
                    x1[2 * i + 1],
                    x[2 * i] - x1[2 * i],
                    x[2 * i + 1] - x1[2 * i + 1],
                    head_width=0.05,
                    head_length=0.1,
                    fc=colors[i % len(colors)],
                    ec=colors[i % len(colors)],
                    length_includes_head=True,
                    alpha=0.3,
                )

        # add margins
        ax.margins(0.1, 0.1)
    elif cfg.data.type == "y_pos":
        # data is concatenated xy pairs, i.e. [x1, y1, x2, y2, ...]
        x = x.cpu().numpy()

        color_pool = list(mcolors.BASE_COLORS.keys())

        if len(x) < len(color_pool):
            colors = color_pool[: len(x)]
        else:
            colors = None

        offsets = np.arange(0, len(x))
        ax.scatter(offsets, x, c=colors)

        # add margins
        ax.margins(0.1, 0.1)
    elif cfg.data.type == "image":
        ax.imshow(x.clamp(0, 1).mul(255).cpu().permute([1, 2, 0]).to(torch.uint8))
        ax.axis("off")
    else:
        raise ValueError(f"Unknown data type: {cfg.data.type}")

    # plt.show()
    if fig_created:
        plt.close(fig)


def plot_reconstructions(
    cfg,
    model,
    loader,
    device,
    n_samples=12,
    filename=None,
    encode_to_causal=False,
    artifact_folder=None,
    save_to_mlflow=True,
):
    nrows = n_samples
    ncols = 4  # x1, x1_reco, x2, x2_reco

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(3 * ncols, 2 * nrows), sharex=True, sharey=True
    )

    # Get data
    x1s, x2s, _, _, true_interventions, *_ = get_first_batch(loader)
    x1s, x2s, true_interventions = x1s.to(device), x2s.to(device), true_interventions.to(device)
    x1s, x2s, true_interventions = x1s[:n_samples], x2s[:n_samples], true_interventions[:n_samples]

    if len(x2s.shape) > 2:
        x2s = x2s[:, -1]

    if encode_to_causal:
        x1s_reco = model.decode_causal(
            model.encode_to_causal(x1s, deterministic=True), deterministic=True
        )
        x2s_reco = model.decode_causal(
            model.encode_to_causal(x2s, deterministic=True), deterministic=True
        )
    else:
        x1s_reco = model.encode_decode(x1s, deterministic=True)
        x2s_reco = model.encode_decode(x2s, deterministic=True)

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
        save_plot(fig, filename, artifact_folder, save_to_mlflow=save_to_mlflow)
    else:
        fig.show()
    return fig, axes


def plot_counterfactuals(
    cfg,
    model,
    loader,
    device,
    n_samples=12,
    intervention_scale=1.0,
    filename=None,
    artifact_folder=None,
    save_to_mlflow=True,
):
    nrows = n_samples
    ncols = 2 + cfg.model.dim_z  # x, empty intervention, e1, e2, ...
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows), sharex=True, sharey=True
    )

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
            plot_x(cfg, x1s_intervened[i, j], ax=axes[i, 2 + j], x1=x1s[i], plot_arrows=True)

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
        save_plot(fig, filename, artifact_folder, save_to_mlflow=save_to_mlflow)
    else:
        fig.show()
    return fig, axes


def intervene(
    model,
    z,
    component,
    mode="sample_uniform",
    perfect=True,
    value=None,
    intervention_type="decode_noise_to_causal",
):
    z_intervened = z.clone()
    if mode == "sample":
        mean = torch.mean(z[:, component])
        std = torch.std(z[:, component])
        delta = torch.randn_like(z[:, component]) * std + mean
    elif mode == "sample_uniform":
        zmin, zmax = torch.min(z[:, component]), torch.max(z[:, component])
        delta = torch.distributions.uniform.Uniform(zmin, zmax).sample(z[:, component].shape)
    elif mode == "mean":
        delta = torch.mean(z[:, component])
    elif mode == "fixed":
        assert value is not None
        delta = value

    z_intervened[:, component] = delta

    if not perfect:
        z_intervened[:, component] += z[:, component]

    if intervention_type == "decode_noise_to_causal":
        x_intervened = model.decode_causal(
            model.scm.noise_to_causal(z_intervened), deterministic=True
        )
    elif intervention_type == "decode_noise":
        x_intervened = model.decode_noise(z_intervened, deterministic=True)
    elif intervention_type == "decode_causal":
        x_intervened = model.decode_causal(z_intervened, deterministic=True)

    return x_intervened, z_intervened


def plot_counterfactual_sequences(
    cfg,
    model,
    loader,
    device,
    n_samples=4,
    n_interventions=5,
    intervention_scale=1.0,
    filename=None,
    artifact_folder=None,
    intervention_type="decode_noise_to_causal",
    save_to_mlflow=True,
    components=None,
    perfect_intervention=True
):
    if components is None:
        components = range(cfg.model.dim_z)
    nrows = n_samples * len(components)
    ncols = 1 + n_interventions  # x, intervention_1, intervention_2, ...
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows), sharex=True, sharey=True
    )

    # Get data
    x1s, x2s, *_ = get_first_batch(loader)
    x1s, x2s = x1s.to(device), x2s.to(device)

    eps1s = model.encode_to_noise(x1s, deterministic=True)
    eps1s_std = eps1s.std(dim=0)

    # eps2s = model.encode_to_noise(x2s, deterministic=True)
    # eps2s_std = eps2s.std(dim=0)

    epss_std = eps1s_std

    if intervention_type == "decode_causal":
        component_std = torch.std(model.scm.noise_to_causal(eps1s), dim=0)
    else:
        component_std = eps1s_std

    eps1s = eps1s[:n_samples]
    x1s_reco = model.decode_noise(eps1s, deterministic=True)
    # x1s_reco = model.decode_causal(model.scm.noise_to_causal(eps1s), deterministic=True)

    subfigures = fig.subfigures(len(components))
    if len(components) == 1:
        subfigures = [subfigures]

    for j, component in enumerate(components):
        axes = subfigures[j].subplots(nrows=n_samples, ncols=ncols, sharex=True, sharey=True)
        if n_samples == 1:
            axes = np.expand_dims(axes, 0)

        title = f"{'Varying' if intervention_type == 'decode_causal' else 'Intervention on'}  ${'z' if intervention_type == 'decode_causal' else 'e'}_{component + 1}$"
        subfigures[j].suptitle(title)

        values = torch.linspace(
            -intervention_scale * component_std[component],
            intervention_scale * component_std[component],
            n_interventions,
        ).to(device)

        x1s_intervened = torch.zeros((n_samples, n_interventions, cfg.model.dim_x)).to(device)
        for i, value in enumerate(values):
            # intervention_i = torch.zeros((1, cfg.model.dim_z)).to(device)
            # intervention_i[:, component] = value
            # eps1s_intervened_i = eps1s + intervention_i
            # x_intervened = model.decode_noise(eps1s_intervened_i, deterministic=True)
            if intervention_type == "decode_causal":
                z = model.scm.noise_to_causal(eps1s)
            else:
                z = eps1s

            if not perfect_intervention:
                # expand value
                value = value.expand(z.shape[0])
                value += z[:, component]
            x_intervened, _ = intervene(
                model,
                z,
                component,
                mode="fixed",
                value=value,
                intervention_type=intervention_type,
            )
            x1s_intervened[:, i, :] = x_intervened

        for i in range(n_samples):
            plot_x(cfg, x1s_reco[i], ax=axes[i, 0])

            for j in range(n_interventions):
                plot_x(
                    cfg,
                    x1s_intervened[i, j],
                    ax=axes[i, j + 1],
                    x1=x1s_reco[i],
                    plot_arrows=True,
                )

    for i in range(nrows):
        for j in range(ncols):
            # axes[i, j].axis("off")
            pass

    # fig.tight_layout()
    if filename is not None:
        save_plot(fig, filename, artifact_folder, save_to_mlflow=save_to_mlflow)
    else:
        fig.show()

    return fig, axes


def plot_solution_function_responses(
    cfg,
    model,
    loader,
    MAP_interventions,
    device,
    n=50,
    n_samples=10,
    domain=(-3, 3),
    sort_by_MAP_interventions=True,
    filename=None,
    artifact_folder=None,
    save_to_mlflow=True,
):
    fig, axes = plt.subplots(
        nrows=cfg.model.dim_z,
        ncols=cfg.model.dim_z,
        figsize=(cfg.model.dim_z * 3, cfg.model.dim_z * 3),
        sharex=True,
        sharey="col",
    )

    if sort_by_MAP_interventions:
        if len(torch.unique(MAP_interventions)) < cfg.model.dim_z + 1:
            # don't remap if not all components can be mapped
            MAP_components = torch.arange(0, cfg.model.dim_z).int()
        else:
            # remove empty intervention
            MAP_components = MAP_interventions[MAP_interventions != 0] - 1
        MAP_components_inverse = torch.tensor(
            [list(MAP_components).index(i) for i in range(len(MAP_components))]
        )

    x_min, x_max = domain

    x1s = []
    samples_collected = 0
    for (
        i,
        (x1, *_),
    ) in enumerate(loader):
        x1 = x1.to(device)
        x1s.append(x1)
        samples_collected += x1.size(0)
        if samples_collected >= n_samples:
            break
    x1s = torch.cat(x1s, dim=0)
    x1s = x1s[:n_samples]

    x1, *_ = get_first_batch(loader)
    x1 = x1.to(device)
    e1s = model.encode_to_noise(x1)

    e1s = e1s[:n_samples]

    for e1 in e1s:
        values = torch.linspace(x_min, x_max, n).to(device)
        solution_function_responses = torch.zeros(cfg.model.dim_z, n, cfg.model.dim_z)

        for i in range(cfg.model.dim_z):
            e = e1.repeat(n, 1).to(device)
            e[:, i] = values
            if sort_by_MAP_interventions:
                solution_function_responses[MAP_components_inverse[i]] = model.scm.noise_to_causal(
                    e
                )[:, MAP_components]
            else:
                solution_function_responses[i] = model.scm.noise_to_causal(e)

        for i in range(cfg.model.dim_z):
            for j in range(cfg.model.dim_z):
                ax = axes[i, j]
                ax.plot(
                    values.detach().cpu(),
                    solution_function_responses[i, :, j].detach().cpu(),
                    alpha=0.5,
                )
                ax.set_title(f"$s_{j+1}(e)$, varying $e_{i+1}$")

    # title
    fig.suptitle(
        f"Solution functions responses",
        fontsize=20,
    )

    if sort_by_MAP_interventions:
        plt.figtext(
            0.5,
            0.01,
            f"Mapping from model components: {list(MAP_components.cpu().numpy())}",
            wrap=True,
            horizontalalignment="center",
            fontsize=12,
        )

    margin = 0.03
    fig.tight_layout(rect=[margin, margin, 1 - margin, 1 - margin])
    if filename is not None:
        save_plot(fig, filename, artifact_folder, save_to_mlflow=save_to_mlflow)
    else:
        fig.show()
    return fig, axes


def plot_graph(
    cfg,
    adjacency_matrix,
    MAP_interventions=None,
    filename=None,
    artifact_folder=None,
    save_to_mlflow=True,
):
    adjacency_matrix = adjacency_matrix.cpu().numpy()
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

    if MAP_interventions is not None:
        if len(torch.unique(MAP_interventions)) < cfg.model.dim_z + 1:
            # don't remap if not all components can be mapped
            MAP_components = torch.arange(0, cfg.model.dim_z).int()
        else:
            # remove empty intervention
            MAP_components = MAP_interventions[MAP_interventions != 0] - 1
        MAP_components_inverse = torch.tensor(
            [list(MAP_components).index(i) for i in range(len(MAP_components))]
        )

        G = nx.relabel_nodes(
            G, {i: f"$z_{MAP_components_inverse[i] + 1}$" for i in range(cfg.data.dim_z)}
        )

    pos = nx.spring_layout(G)
    # Filter out edges with weights below the threshold
    weight_threshold = 0.01
    edge_data_to_draw = [
        (u, v, d) for u, v, d in G.edges(data=True) if d["weight"] >= weight_threshold
    ]
    edges_to_draw = [(u, v) for u, v, _ in edge_data_to_draw]
    weights_to_draw = [d["weight"] for _, _, d in edge_data_to_draw]

    edge_width = 2
    widths_to_draw = [w * edge_width for w in weights_to_draw]

    if MAP_interventions is not None:
        plt.figtext(
            0.5,
            0.01,
            f"Mapping from model components: {list(MAP_components.cpu().numpy())}",
            wrap=True,
            horizontalalignment="center",
            fontsize=12,
        )

    margin = 0.03
    plt.tight_layout(rect=[margin, margin, 1 - margin, 1 - margin])

    try:
        nx.draw(G, pos, edgelist=edges_to_draw, with_labels=True, width=widths_to_draw)

        labels = nx.get_edge_attributes(G, "weight")
        labels = {k: v for k, v in labels.items() if v >= weight_threshold}
        labels = {k: round(v, 2) for k, v in labels.items()}

        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        if filename is not None:
            plt.savefig(filename)
            if save_to_mlflow:
                mlflow.log_artifact(
                    filename,
                    artifact_folder if artifact_folder is not None else Path(filename).stem,
                )
            plt.close()
        else:
            plt.show()
    except StopIteration:
        pass


def plot_grid(cfg, x, nrows=4, ncols=4, filename=None):
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(2.5 * ncols, nrows * 2.5),
        sharex=True,
        sharey=True,
    )
    for i in range(nrows):
        for j in range(ncols):
            plot_x(cfg, x[i * ncols + j], axes[i, j])

    fig.tight_layout()
    if filename:
        save_plot(fig, filename, save_to_mlflow=False)


def plot_columns(cfg, xs, nrows=4, col_names=None, filename=None):
    ncols = len(xs)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(2.5 * ncols, nrows * 2.5),
        sharex=True,
        sharey=True,
    )
    for i in range(nrows):
        for j in range(ncols):
            plot_x(cfg, xs[j][i], axes[i, j])

    if col_names:
        for j in range(ncols):
            axes[0, j].set_title(col_names[j])

    fig.tight_layout()
    if filename:
        save_plot(fig, filename, save_to_mlflow=False)
