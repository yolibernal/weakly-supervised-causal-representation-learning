#!/usr/bin/env python3
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

from collections import defaultdict
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from experiments.experiment_utils import (
    compute_metrics_on_dataset,
    create_intervention_encoder,
    create_optimizer_and_scheduler,
    determine_graph_learning_settings,
    frequency_check,
    initialize_experiment,
    log_training_step,
    logger,
    optimizer_step,
    reset_optimizer_state,
    save_config,
    save_model,
    set_manifold_thickness,
    step_schedules,
    update_dict,
)
from ws_crl.causal.implicit_scm import (
    LinearImplicitSCM,
    LipschitzMonotonicSCM,
    MLPImplicitSCM,
)
from ws_crl.causal.scm import (
    MLPFixedOrderSCM,
    MLPVariableOrderCausalModel,
    UnstructuredPrior,
)
from ws_crl.encoder import GaussianEncoder
from ws_crl.lcm import ELCM, ILCM
from ws_crl.metrics import compute_dci
from ws_crl.plotting import (
    plot_average_intervention_posterior,
    plot_counterfactuals,
    plot_graph,
    plot_importance_matrix,
    plot_latent_space,
    plot_reconstructions,
    plot_solution_function_responses,
)
from ws_crl.posthoc_graph_learning import (
    compute_implicit_causal_effects,
    find_topological_order,
    run_enco,
)
from ws_crl.training import VAEMetrics
from ws_crl.utils import (
    calculate_average_intervention_posterior,
    calculate_intervention_posteriors,
    generate_directed_graph_matrix,
)
from torch import nn


@hydra.main(
    config_path="../config",
    config_name="2dcubes_ilcm",
    # config_name="2dcubes_elcm",
)
def main(cfg):
    """Hydra wrapper around experiment to be called from commandline"""
    experiment(cfg)


def experiment(cfg):
    """High-level experiment function"""
    # Initialization
    experiment_id = initialize_experiment(cfg)

    with mlflow.start_run(experiment_id=experiment_id, run_name=cfg.general.run_name) as run:
        cfg.general.run_name = run.info.run_name

        save_config(cfg)

        # Train
        model = create_model(cfg)
        train(cfg, model)
        save_model(cfg, model)

        # Test
        metrics = evaluate(cfg, model)

    logger.info("Anders nog iets?")
    return metrics


def create_model(cfg):
    """Instantiates a (learnable) VAE model"""

    # Create model
    logger.info(f"Creating {cfg.model.type} model")
    scm = create_scm(cfg)
    encoder, decoder = create_encoder_decoder(cfg)

    if cfg.model.type == "causal_vae":
        model = ELCM(
            scm, encoder=encoder, decoder=decoder, intervention_prior=None, dim_z=cfg.model.dim_z
        )
    elif cfg.model.type == "intervention_noise_vae":
        intervention_encoder = create_intervention_encoder(cfg)
        model = ILCM(
            scm,
            encoder=encoder,
            decoder=decoder,
            intervention_encoder=intervention_encoder,
            intervention_prior=None,
            averaging_strategy=cfg.model.averaging_strategy,
            dim_z=cfg.model.dim_z,
        )
    else:
        raise ValueError(f"Unknown value for cfg.model.type: {cfg.model.type}")

    # Load model checkpoint, if specified
    if "load" in cfg.model and cfg.model.load is not None:
        logger.info(f"Loading model checkpoint from {cfg.model.load}")
        state_dict = torch.load(cfg.model.load, map_location="cpu")
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Hotfix to guarantee backwards compatibility with old state dicts
            state_dict["scm._mask_values"] = torch.zeros(cfg.model.dim_z)
            model.load_state_dict(state_dict)

    return model


def create_scm(cfg):
    """Creates an SCM"""
    logger.info(f"Creating {cfg.model.scm.type} SCM")
    noise_centric = cfg.model.type in {
        "noise_vae",
        "intervention_noise_vae",
        "alt_intervention_noise_vae",
    }

    if cfg.model.scm.type == "ground_truth":
        raise NotImplementedError
    elif cfg.model.scm.type == "unstructured":  # Baseline VAE
        scm = UnstructuredPrior(dim_z=cfg.model.dim_z)
    elif noise_centric and cfg.model.scm.type == "mlp":
        logger.info(
            f"Graph parameterization for noise-centric learning: {cfg.model.scm.adjacency_matrix}"
        )
        scm = MLPImplicitSCM(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            min_std=cfg.model.scm.min_std,
            n_transforms=cfg.model.scm.n_transforms if "n_transforms" in cfg.model.scm else 1,
            concat_masks_to_parents=(
                cfg.model.scm.concat_masks_to_parents
                if "concat_masks_to_parents" in cfg.model.scm
                else True
            ),
        )
    elif noise_centric and cfg.model.scm.type == "sparse_mlp":
        logger.info(
            f"Graph parameterization for noise-centric learning: {cfg.model.scm.adjacency_matrix}"
        )
        scm = MLPImplicitSCM(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            min_std=cfg.model.scm.min_std,
            transform_type="sparse_affine",
            n_transforms=cfg.model.scm.n_transforms if "n_transforms" in cfg.model.scm else 1,
            concat_masks_to_parents=(
                cfg.model.scm.concat_masks_to_parents
                if "concat_masks_to_parents" in cfg.model.scm
                else True
            ),
        )
    elif noise_centric and cfg.model.scm.type == "linear":
        scm = LinearImplicitSCM(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            min_std=cfg.model.scm.min_std,
            transform_type="linear",
            n_transforms=cfg.model.scm.n_transforms if "n_transforms" in cfg.model.scm else 1,
            concat_masks_to_parents=(
                cfg.model.scm.concat_masks_to_parents
                if "concat_masks_to_parents" in cfg.model.scm
                else True
            ),
        )
    elif noise_centric and cfg.model.scm.type == "sparse_linear":
        scm = LinearImplicitSCM(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            min_std=cfg.model.scm.min_std,
            transform_type="sparse_linear",
            n_transforms=cfg.model.scm.n_transforms if "n_transforms" in cfg.model.scm else 1,
            concat_masks_to_parents=(
                cfg.model.scm.concat_masks_to_parents
                if "concat_masks_to_parents" in cfg.model.scm
                else True
            ),
        )
    elif noise_centric and cfg.model.scm.type == "residual_mlp":
        scm = MLPImplicitSCM(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            transform_type="residual",
            n_transforms=cfg.model.scm.n_transforms if "n_transforms" in cfg.model.scm else 1,
            concat_masks_to_parents=(
                cfg.model.scm.concat_masks_to_parents
                if "concat_masks_to_parents" in cfg.model.scm
                else True
            ),
        )
    elif noise_centric and cfg.model.scm.type == "lipschitz_monotonic":
        scm = LipschitzMonotonicSCM(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            min_std=cfg.model.scm.min_std,
            monotonic_constraints=cfg.model.scm.monotonic_constraints,
            n_groups=cfg.model.scm.n_groups,
            kind=cfg.model.scm.weight_constraint_kind,
            lipschitz_const=cfg.model.scm.lipschitz_const,
            n_transforms=cfg.model.scm.n_transforms if "n_transforms" in cfg.model.scm else 1,
            concat_masks_to_parents=(
                cfg.model.scm.concat_masks_to_parents
                if "concat_masks_to_parents" in cfg.model.scm
                else True
            ),
        )
    elif noise_centric and cfg.model.scm.type == "sparse_lipschitz_monotonic":
        scm = LipschitzMonotonicSCM(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            min_std=cfg.model.scm.min_std,
            monotonic_constraints=cfg.model.scm.monotonic_constraints,
            n_groups=cfg.model.scm.n_groups,
            kind=cfg.model.scm.weight_constraint_kind,
            lipschitz_const=cfg.model.scm.lipschitz_const,
            transform_type="sparse_affine",
            n_transforms=cfg.model.scm.n_transforms if "n_transforms" in cfg.model.scm else 1,
            concat_masks_to_parents=(
                cfg.model.scm.concat_masks_to_parents
                if "concat_masks_to_parents" in cfg.model.scm
                else True
            ),
        )
    elif noise_centric and cfg.model.scm.type == "residual_lipschitz_monotonic":
        scm = LipschitzMonotonicSCM(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            min_std=cfg.model.scm.min_std,
            monotonic_constraints=cfg.model.scm.monotonic_constraints,
            n_groups=cfg.model.scm.n_groups,
            kind=cfg.model.scm.weight_constraint_kind,
            lipschitz_const=cfg.model.scm.lipschitz_const,
            transform_type="residual",
            n_transforms=cfg.model.scm.n_transforms if "n_transforms" in cfg.model.scm else 1,
            concat_masks_to_parents=(
                cfg.model.scm.concat_masks_to_parents
                if "concat_masks_to_parents" in cfg.model.scm
                else True
            ),
        )
    elif (
        not noise_centric
        and cfg.model.scm.type == "mlp"
        and cfg.model.scm.adjacency_matrix in {"enco", "dds"}
    ):
        logger.info(
            f"Adjacency matrix: learnable, {cfg.model.scm.adjacency_matrix} parameterization"
        )
        scm = MLPVariableOrderCausalModel(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            enhance_causal_effects_at_init=False,
            min_std=cfg.model.scm.min_std,
        )
    elif (
        not noise_centric
        and cfg.model.scm.type == "mlp"
        and cfg.model.scm.adjacency_matrix == "fixed_order"
    ):
        logger.info(f"Adjacency matrix: learnable, fixed topological order")
        scm = MLPFixedOrderSCM(
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            enhance_causal_effects_at_init=False,
            min_std=cfg.model.scm.min_std,
        )
    else:
        raise ValueError(f"Unknown value for cfg.model.scm.type: {cfg.model.scm.type}")

    return scm


def create_encoder_decoder(cfg):
    """Creates encoder and decoder"""
    logger.info(f"Creating {cfg.model.encoder.type} encoder / decoder")

    if cfg.model.encoder.type == "mlp":
        encoder_hidden_layers = cfg.model.encoder.hidden_layers
        encoder_hidden = [cfg.model.encoder.hidden_units for _ in range(encoder_hidden_layers)]
        decoder_hidden_layers = cfg.model.decoder.hidden_layers
        decoder_hidden = [cfg.model.decoder.hidden_units for _ in range(decoder_hidden_layers)]

        encoder = GaussianEncoder(
            hidden=encoder_hidden,
            input_features=cfg.model.dim_x,
            output_features=cfg.model.dim_z,
            fix_std=cfg.model.encoder.fix_std,
            init_std=cfg.model.encoder.std,
            min_std=cfg.model.encoder.min_std,
        )
        decoder = GaussianEncoder(
            hidden=decoder_hidden,
            input_features=cfg.model.dim_z,
            output_features=cfg.model.dim_x,
            fix_std=cfg.model.decoder.fix_std,
            init_std=cfg.model.decoder.std,
            min_std=cfg.model.decoder.min_std,
        )
    else:
        raise ValueError(f"Unknown value for encoder_cfg.type: {cfg.model.encoder.type}")

    return encoder, decoder


def train(cfg, model):
    """High-level training function"""

    if "skip" in cfg.training and cfg.training.skip and cfg.training.skip != "None":
        return {}, {}

    logger.info("Starting training")
    logger.info(f"Training on {cfg.training.device}")
    device = torch.device(cfg.training.device)

    # Training
    criteria = VAEMetrics(dim_z=cfg.data.dim_z)
    optim, scheduler = create_optimizer_and_scheduler(cfg, model, separate_param_groups=True)

    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    best_state = {"state_dict": None, "loss": None, "step": None}

    train_loader = get_dataloader(cfg, "train", batchsize=cfg.training.batchsize, shuffle=True)
    val_loader = get_dataloader(
        cfg, "val", batchsize=cfg.eval.batchsize, shuffle=False, include_noise_encodings=True
    )
    steps_per_epoch = len(train_loader)

    # GPU
    model = model.to(device)

    step = 0
    nan_counter = 0
    epoch_generator = trange(cfg.training.epochs, disable=not cfg.general.verbose)

    if (
        "pretrain_context_invariance" in cfg.training
        and cfg.training.pretrain_context_invariance.epochs is not None
    ):
        logger.info("Pretraining context invariance")
        for i, solution_function in enumerate(model.scm.solution_functions):
            n = cfg.training.pretrain_context_invariance.samples
            e_train = (
                torch.rand(n, cfg.model.dim_z)
                * (
                    cfg.training.pretrain_context_invariance.e_max
                    - cfg.training.pretrain_context_invariance.e_min
                )
                + cfg.training.pretrain_context_invariance.e_min
            )
            z_train = (
                e_train[:, i] * cfg.model.scm.lipschitz_const
                + torch.randn(n) * cfg.training.pretrain_context_invariance.noise_std
            )
            e_train, z_train = e_train.to(device), z_train.to(device)

            pretrain_epochs = cfg.training.pretrain_context_invariance.epochs
            batch_size = cfg.training.pretrain_context_invariance.batch_size
            n_batches = n // batch_size
            optimizer = torch.optim.Adam(
                solution_function.parameters(),
                lr=cfg.training.pretrain_context_invariance.learning_rate,
            )
            criterion = nn.MSELoss()

            for epoch in range(pretrain_epochs):
                for i_batch in range(n_batches):
                    e_batch = e_train[i_batch * batch_size : (i_batch + 1) * batch_size]
                    z_batch = z_train[i_batch * batch_size : (i_batch + 1) * batch_size]

                    optimizer.zero_grad()
                    # context = torch.concatenate([e_batch[:, 0:1], e_batch], dim=1)
                    context = e_batch
                    masked_epsilon = model.scm.get_masked_context(i, context, adjacency_matrix=None)
                    z_pred, _ = solution_function.inverse(
                        e_batch[:, i : i + 1], context=masked_epsilon
                    )
                    loss = criterion(z_pred.squeeze(), z_batch)
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                for param in solution_function.parameters():
                    param.add_(
                        torch.randn(param.size(), device=device)
                        * cfg.training.pretrain_context_invariance.weight_noise_std
                    )

    for epoch in epoch_generator:
        mlflow.log_metric("train.epoch", epoch, step=step)

        # Graph sampling settings
        graph_kwargs = determine_graph_learning_settings(cfg, epoch, model)

        # Epoch-based schedules
        (
            model_interventions,
            pretrain,
            deterministic_intervention_encoder,
            introduce_child_interventions,
        ) = epoch_schedules(cfg, model, epoch, optim)

        for x1, x2, z1, z2, intervention_labels, true_interventions in train_loader:
            fractional_epoch = step / steps_per_epoch

            if not introduce_child_interventions:
                child_intervention_labels = torch.tensor(cfg.data.child_intervention_labels)
                # filter out indices where intervention label in child interventions
                indices = torch.where(
                    ~torch.any(intervention_labels[:, None] == child_intervention_labels, dim=1)
                )[0]
                x1, x2, z1, z2, intervention_labels, true_interventions = (
                    x1[indices],
                    x2[indices],
                    z1[indices],
                    z2[indices],
                    intervention_labels[indices],
                    true_interventions[indices],
                )

            model.train()

            # Step-based schedules
            (
                beta,
                beta_intervention,
                consistency_regularization_amount,
                cyclicity_regularization_amount,
                edge_regularization_amount,
                inverse_consistency_regularization_amount,
                z_regularization_amount,
                intervention_entropy_regularization_amount,
                intervention_encoder_offset,
                solution_function_regularization_amount,
            ) = step_schedules(cfg, model, fractional_epoch)

            # GPU
            x1, x2, z1, z2, intervention_labels, true_interventions = (
                x1.to(device),
                x2.to(device),
                z1.to(device),
                z2.to(device),
                intervention_labels.to(device),
                true_interventions.to(device),
            )

            # Model forward pass
            log_prob, model_outputs = model(
                x1,
                x2,
                beta=beta,
                beta_intervention_target=beta_intervention,
                pretrain_beta=cfg.training.pretrain_beta,
                full_likelihood=cfg.training.full_likelihood,
                likelihood_reduction=cfg.training.likelihood_reduction,
                pretrain=pretrain,
                model_interventions=model_interventions,
                deterministic_intervention_encoder=deterministic_intervention_encoder,
                intervention_encoder_offset=intervention_encoder_offset,
                sample_from_intervention_posterior=(
                    cfg.training.sample_from_intervention_posterior
                    if "sample_from_intervention_posterior" in cfg.training
                    else False
                ),
                **graph_kwargs,
            )

            # Loss and metrics
            loss, metrics = criteria(
                log_prob,
                true_intervention_labels=intervention_labels,
                z_regularization_amount=z_regularization_amount,
                edge_regularization_amount=edge_regularization_amount,
                cyclicity_regularization_amount=cyclicity_regularization_amount,
                consistency_regularization_amount=consistency_regularization_amount,
                inverse_consistency_regularization_amount=inverse_consistency_regularization_amount,
                intervention_entropy_regularization_amount=intervention_entropy_regularization_amount,
                solution_function_regularization_amount=solution_function_regularization_amount,
                **model_outputs,
            )

            # Optimizer step
            finite, grad_norm = optimizer_step(cfg, loss, model, model_outputs, optim, x1, x2)
            if not finite:
                nan_counter += 1

            # Log loss and metrics
            step += 1
            log_training_step(
                cfg,
                beta,
                epoch_generator,
                finite,
                grad_norm,
                metrics,
                model,
                step,
                train_metrics,
                nan_counter,
            )

            # Validation loop
            if frequency_check(step, cfg.training.validate_every_n_steps):
                validation_loop(
                    cfg, model, criteria, val_loader, best_state, val_metrics, step, device
                )

            if frequency_check(step, cfg.training.plot_every_n_steps):
                plot_loop(cfg, model, val_loader, val_metrics, device, step)

            # Save model checkpoint
            if frequency_check(step, cfg.training.save_model_every_n_steps):
                save_model(cfg, model, f"model_step_{step}.pt")

        plot_epochs = [
            cfg.training.model_interventions_after_epoch,
            cfg.training.freeze_encoder_epoch,
            cfg.training.fix_topological_order_epoch,
            cfg.training.deterministic_intervention_encoder_after_epoch,
        ]
        if "introduce_child_interventions_after_epoch" in cfg.training:
            plot_epochs.append(cfg.training.introduce_child_interventions_after_epoch)
        if epoch in plot_epochs:
            plot_loop(cfg, model, val_loader, val_metrics, device, step)

        # LR scheduler
        if scheduler is not None and epoch < cfg.training.epochs - 1:
            scheduler.step()
            mlflow.log_metric(f"train.lr", scheduler.get_last_lr()[0], step=step)

            # Optionally reset Adam stats
            if (
                cfg.training.lr_schedule.type == "cosine_restarts_reset"
                # and "restart_every_epochs" in cfg.training.lr_schedule
                and (epoch + 1) % cfg.training.lr_schedule.restart_every_epochs == 0
                and epoch + 1 < cfg.training.epochs
            ):
                logger.info(f"Resetting optimizer at epoch {epoch + 1}")
                reset_optimizer_state(optim)

    # Final validation loop, wrapping up early stopping
    if cfg.training.validate_every_n_steps is not None and cfg.training.validate_every_n_steps > 0:
        validation_loop(cfg, model, criteria, val_loader, best_state, val_metrics, step, device)

        # noinspection PyTypeChecker
        if cfg.training.early_stopping and best_state["step"] < step:
            logger.info(
                f'Early stopping after step {best_state["step"]} '
                f'with validation loss {best_state["loss"]}'
            )
            model.load_state_dict(best_state["state_dict"])

    # Final plotting loop
    if cfg.training.plot_every_n_steps is not None and cfg.training.plot_every_n_steps > 0:
        plot_loop(cfg, model, val_loader, val_metrics, device, step)

    # Reset model: back to CPU, reset manifold thickness
    set_manifold_thickness(cfg, model, None)

    return train_metrics, val_metrics


def get_train_mean_std(cfg):
    """Calculate mean and std of x1 and x2"""
    if (
        cfg.data.x1_mean is not None
        and cfg.data.x1_std is not None
        and cfg.data.x2_mean is not None
        and cfg.data.x2_std is not None
    ):
        # Return float tensors
        return (
            torch.FloatTensor(cfg.data.x1_mean),
            torch.FloatTensor(cfg.data.x1_std),
            torch.FloatTensor(cfg.data.x2_mean),
            torch.FloatTensor(cfg.data.x2_std),
        )

    train_data = load_dataset(cfg, "train", normalize=False)
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    x1, x2, *_ = next(iter(train_loader))

    # x1_mean per variable
    if cfg.data.x1_mean is None:
        cfg.data.x1_mean = x1.mean(dim=0).tolist()
    if cfg.data.x1_std is None:
        cfg.data.x1_std = x1.std(dim=0).tolist()
    if cfg.data.x2_mean is None:
        cfg.data.x2_mean = x2.mean(dim=0).tolist()
    if cfg.data.x2_std is None:
        cfg.data.x2_std = x2.std(dim=0).tolist()

    logger.info(f"x1_mean: {cfg.data.x1_mean}")
    logger.info(f"x1_std: {cfg.data.x1_std}")
    logger.info(f"x2_mean: {cfg.data.x2_mean}")
    logger.info(f"x2_std: {cfg.data.x2_std}")

    return (
        torch.FloatTensor(cfg.data.x1_mean),
        torch.FloatTensor(cfg.data.x1_std),
        torch.FloatTensor(cfg.data.x2_mean),
        torch.FloatTensor(cfg.data.x2_std),
    )


def load_dataset(cfg, tag, include_noise_encodings=True, normalize=True):
    filename = Path(cfg.data.data_dir) / f"{tag}_encoded.pt"
    logger.debug(f"Loading data from {filename}")
    data = torch.load(filename)

    if not include_noise_encodings:
        data = list(data)
        data = data[:6]
        e1, e2 = None, None
    else:
        e1, e2 = data[-2], data[-1]

    x1, x2, z1, z2, intervention_labels, true_interventions, *_ = data

    intervention_labels = intervention_labels.int()

    device = x1.device

    if len(x2.shape) > 2:
        logger.info("Choosing last sequence element as x2")
        x2 = x2[:, -1]
        z2 = z2[:, -1]
        if include_noise_encodings:
            e2 = e2[:, -1]

    if normalize:
        x1_mean, x1_std, x2_mean, x2_std = get_train_mean_std(cfg)
        x1_mean, x1_std, x2_mean, x2_std = (
            x1_mean.to(device),
            x1_std.to(device),
            x2_mean.to(device),
            x2_std.to(device),
        )
        x_mean = (x1_mean + x2_mean) / 2
        x_std = (x1_std + x2_std) / 2

        x1 = (x1 - x_mean) / x_std
        x2 = (x2 - x_mean) / x_std

    other = [z1, z2, intervention_labels, true_interventions]
    if include_noise_encodings:
        other.extend([e1, e2])

    dataset = TensorDataset(x1, x2, *other)
    return dataset


def get_dataloader(
    cfg, tag, batchsize=None, shuffle=False, include_noise_encodings=False, normalize=True
):
    """Load data from disk and return DataLoader instance"""

    dataset = load_dataset(
        cfg, tag, include_noise_encodings=include_noise_encodings, normalize=normalize
    )
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)

    return dataloader


def load_data(cfg, tag):
    """Load data from disk and return numpy arrays"""

    assert tag in {"train", "dci_train", "test", "val"}

    filename = Path(cfg.data.data_dir) / f"{tag}.npz"
    logger.debug(f"Loading data from {filename}")
    data = np.load(filename)

    return data


@torch.no_grad()
def validation_loop(cfg, model, criteria, val_loader, best_state, val_metrics, step, device):
    """Validation loop, computing a number of metrics and checkpointing the best model"""

    loss, nll, metrics = compute_metrics_on_dataset(cfg, model, criteria, val_loader, device)
    metrics.update(eval_dci_scores(cfg, model, test_loader=val_loader))
    metrics.update(eval_dci_scores(cfg, model, test_loader=val_loader, interventional=True))
    metrics.update(eval_implicit_graph(cfg, model, dataloader=val_loader))

    # Store validation metrics (and track with MLflow)
    update_dict(val_metrics, metrics)
    for key, value in metrics.items():
        mlflow.log_metric(f"val.{key}", value, step=step)

    # Print DCI disentanglement score
    logger.info(
        f"Step {step}: causal disentanglement = {metrics['causal_disentanglement']:.2f}, "
        f"noise disentanglement = {metrics['noise_disentanglement']:.2f}"
    )

    # Early stopping: compare val loss to last val loss
    new_val_loss = metrics["nll"] if cfg.training.early_stopping_var == "nll" else loss.item()
    if best_state["loss"] is None or new_val_loss < best_state["loss"]:
        best_state["loss"] = new_val_loss
        best_state["state_dict"] = model.state_dict().copy()
        best_state["step"] = step


def plot_loop(
    cfg, model, loader, metrics, device, step, format="pdf", save_to_mlflow=True, plot_dir=None
):
    model.eval()
    model.to(device)

    with torch.no_grad():
        if plot_dir is None:
            plot_dir = Path(cfg.general.exp_dir) / "figures"
        else:
            plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        intervention_posterior, intervention_label = calculate_intervention_posteriors(
            cfg, model, loader, device
        )
        average_intervention_posterior = calculate_average_intervention_posterior(
            cfg, intervention_posterior, intervention_label, device=device
        )
        _, MAP_interventions = average_intervention_posterior.max(dim=1)

        plot_importance_matrix(
            cfg,
            metrics,
            "noise",
            filename=plot_dir / f"importance_matrix_noise_step_{step}.{format}",
            artifact_folder="importance_matrix_noise",
            save_to_mlflow=save_to_mlflow,
        )
        plot_importance_matrix(
            cfg,
            metrics,
            "causal",
            filename=plot_dir / f"importance_matrix_causal_step_{step}.{format}",
            artifact_folder="importance_matrix_causal",
            save_to_mlflow=save_to_mlflow,
        )
        plot_latent_space(
            cfg,
            model,
            loader,
            MAP_interventions,
            device,
            causal=False,
            filename=plot_dir / f"latent_space_noise_step_{step}.{format}",
            artifact_folder="latent_space_noise",
            save_to_mlflow=save_to_mlflow,
        )
        plot_latent_space(
            cfg,
            model,
            loader,
            MAP_interventions,
            device,
            causal=True,
            filename=plot_dir / f"latent_space_causal_step_{step}.{format}",
            artifact_folder="latent_space_causal",
            save_to_mlflow=save_to_mlflow,
        )
        plot_average_intervention_posterior(
            cfg,
            average_intervention_posterior,
            filename=plot_dir / f"average_intervention_posterior_step_{step}.{format}",
            artifact_folder="average_intervention_posterior",
            save_to_mlflow=save_to_mlflow,
        )
        if "encoder" in cfg.data:
            plot_reconstructions(
                cfg,
                model,
                loader,
                device,
                filename=plot_dir / f"reconstructions_step_{step}.{format}",
                artifact_folder="reconstructions",
                save_to_mlflow=save_to_mlflow,
            )
            plot_counterfactuals(
                cfg,
                model,
                loader,
                device,
                filename=plot_dir / f"counterfactuals_step_{step}.{format}",
                artifact_folder="counterfactuals",
                save_to_mlflow=save_to_mlflow,
            )
        plot_solution_function_responses(
            cfg,
            model,
            loader,
            MAP_interventions,
            device,
            filename=plot_dir / f"solution_function_responses_step_{step}.{format}",
            artifact_folder="solution_function_responses",
            save_to_mlflow=save_to_mlflow,
        )
        implicit_graph_adjacency_matrix = generate_directed_graph_matrix(
            metrics, f"implicit_graph_"
        )
        plot_graph(
            cfg,
            adjacency_matrix=implicit_graph_adjacency_matrix,
            MAP_interventions=MAP_interventions,
            filename=plot_dir / f"implicit_graph_step_{step}.{format}",
            artifact_folder="implicit_graph",
            save_to_mlflow=save_to_mlflow,
        )


def evaluate(cfg, model):
    """High-level test function"""

    logger.info("Starting evaluation")

    # Compute metrics
    test_metrics = {}
    test_metrics.update(eval_dci_scores(cfg, model, partition=cfg.eval.eval_partition))
    test_metrics.update(
        eval_dci_scores(cfg, model, partition=cfg.eval.eval_partition, interventional=True)
    )
    test_metrics.update(eval_enco_graph(cfg, model, partition=cfg.eval.eval_partition))
    test_metrics.update(eval_implicit_graph(cfg, model, partition=cfg.eval.eval_partition))
    test_metrics.update(eval_test_metrics(cfg, model))

    # Log results
    for key, val in test_metrics.items():
        mlflow.log_metric(f"eval.{key}", val)

    # Print DCI disentanglement score
    logger.info(
        f"Final evaluation: causal disentanglement = {test_metrics['causal_disentanglement']:.2f}, "
        f"noise disentanglement = {test_metrics['noise_disentanglement']:.2f}"
    )

    # Store results in csv file
    # Pandas does not like scalar values, have to be iterables
    test_metrics_ = {key: [val] for key, val in test_metrics.items()}
    df = pd.DataFrame.from_dict(test_metrics_)
    df.to_csv(Path(cfg.general.exp_dir) / "metrics" / "test_metrics.csv")

    eval_plot(cfg, model, test_metrics, partition=cfg.eval.eval_partition)

    return test_metrics


def eval_plot(cfg, model, metrics, partition="val"):
    loader = get_dataloader(cfg, partition, cfg.eval.batchsize)
    device = torch.device(cfg.training.device)

    plot_dir = Path(cfg.general.exp_dir) / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    intervention_posterior, intervention_label = calculate_intervention_posteriors(
        cfg, model, loader, device
    )
    average_intervention_posterior = calculate_average_intervention_posterior(
        cfg, intervention_posterior, intervention_label, device=device
    )
    _, MAP_interventions = average_intervention_posterior.max(dim=1)
    enco_adjacency_matrix = generate_directed_graph_matrix(metrics, f"enco_graph_")
    plot_graph(
        cfg,
        enco_adjacency_matrix,
        MAP_interventions,
        filename=plot_dir / f"enco_graph_eval.pdf",
        artifact_folder="enco_graph",
    )


@torch.no_grad()
def eval_test_metrics(cfg, model):
    """Evaluates loss terms on test data"""

    device = torch.device(cfg.training.device)
    model = model.to(device)

    criteria = VAEMetrics(dim_z=cfg.data.dim_z)
    test_loader = get_dataloader(
        cfg, cfg.eval.eval_partition, batchsize=cfg.eval.batchsize, shuffle=False
    )
    _, _, metrics = compute_metrics_on_dataset(
        cfg, model, criteria, test_loader, device=torch.device(cfg.training.device)
    )
    return metrics


@torch.no_grad()
def eval_dci_scores(
    cfg, model, partition="val", test_loader=None, full_importance_matrix=True, interventional=False
):
    """Evaluates DCI scores"""

    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")
    model = model.to(device)

    def _load(partition, device, out_device, dataloader=None):
        if dataloader is None:
            dataloader = get_dataloader(
                cfg, partition, cfg.eval.batchsize, include_noise_encodings=True
            )

        model_z, true_z = [], []
        model_e, true_e = [], []

        for (
            x1_batch,
            x2_batch,
            true_z1_batch,
            true_z2_batch,
            *_,
            true_e1_batch,
            true_e2_batch,
        ) in dataloader:

            if not interventional:
                x_batch = x1_batch
                true_z_batch = true_z1_batch
                true_e_batch = true_e1_batch
            else:
                x_batch = x2_batch
                true_z_batch = true_z2_batch
                true_e_batch = true_e2_batch

            x_batch = x_batch.to(device)
            true_z_batch = true_z_batch.to(device)
            true_e_batch = true_e_batch.to(device)

            z_batch = model.encode_to_causal(x_batch, deterministic=True)
            e_batch = model.encode_to_noise(x_batch, deterministic=True)

            model_z.append(z_batch.to(out_device))
            model_e.append(e_batch.to(out_device))
            true_z.append(true_z_batch.to(out_device))
            true_e.append(true_e_batch.to(out_device))

        model_z = torch.cat(model_z, dim=0).detach()
        true_z = torch.cat(true_z, dim=0).detach()
        model_e = torch.cat(model_e, dim=0).detach()
        true_e = torch.cat(true_e, dim=0).detach()
        return true_z, model_z, true_e, model_e

    train_true_z, train_model_z, train_true_e, train_model_e = _load("dci_train", device, cpu)
    test_true_z, test_model_z, test_true_e, test_model_e = _load(
        partition, device, cpu, dataloader=test_loader
    )

    causal_dci_metrics = compute_dci(
        train_true_z,
        train_model_z,
        test_true_z,
        test_model_z,
        return_full_importance_matrix=full_importance_matrix,
    )
    noise_dci_metrics = compute_dci(
        # HACK: remove after fixed data is uploaded
        # train_true_e,
        train_true_e.squeeze(-1),
        train_model_e,
        # test_true_e,
        test_true_e.squeeze(-1),
        test_model_e,
        return_full_importance_matrix=full_importance_matrix,
    )

    combined_metrics = {}
    for key, val in noise_dci_metrics.items():
        combined_metrics[f"noise_{key}"] = val
    for key, val in causal_dci_metrics.items():
        combined_metrics[f"causal_{key}"] = val

    return combined_metrics


@torch.no_grad()
def eval_implicit_graph(cfg, model, partition="val", dataloader=None):
    """Evaluates implicit graph"""

    # This is only defined for noise-centric models (ILCMs)
    if cfg.model.type not in ["intervention_noise_vae", "alt_intervention_noise_vae"]:
        return {}

    # Let's skip this for large latent spaces
    if cfg.model.dim_z > 8:
        return {}

    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")

    # Dataloader
    if dataloader is None:
        dataloader = get_dataloader(cfg, partition, cfg.eval.batchsize)

    # Load data and compute noise encodings
    noise = []
    for x_batch, *_ in dataloader:
        x_batch = x_batch.to(device)
        noise.append(model.encode_to_noise(x_batch, deterministic=True).to(cpu))

    noise = torch.cat(noise, dim=0).detach()

    # Evaluate causal strength
    model = model.to(cpu)
    causal_effects, topological_order = compute_implicit_causal_effects(
        model,
        noise,
        concat_mask=(
            cfg.model.scm.concat_masks_to_parents
            if "concat_masks_to_parents" in cfg.model.scm
            else True
        ),
    )

    # Package as dict
    results = {
        f"implicit_graph_{i}_{j}": causal_effects[i, j].item()
        for i in range(model.dim_z)
        for j in range(model.dim_z)
    }

    model.to(device)

    return results


def eval_enco_graph(cfg, model, partition="train"):
    """Post-hoc graph evaluation with ENCO"""

    # Only want to do this for ILCMs
    if cfg.model.type not in ["intervention_noise_vae", "alt_intervention_noise_vae"]:
        return {}

    # Let's skip this for large latent spaces
    if cfg.model.dim_z > 8:
        return {}

    logger.info("Evaluating learned graph with ENCO")

    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")
    model.to(device)

    # Load data and compute causal variables
    dataloader = get_dataloader(cfg, partition, cfg.eval.batchsize)
    z0s, z1s, interventions = [], [], []

    with torch.no_grad():
        for x0, x1, *_ in dataloader:
            x0, x1 = x0.to(device), x1.to(device)
            _, _, _, _, e0, e1, _, _, intervention = model.encode_decode_pair(
                x0.to(device), x1.to(device)
            )
            z0 = model.scm.noise_to_causal(e0)
            z1 = model.scm.noise_to_causal(e1)

            z0s.append(z0.to(cpu))
            z1s.append(z1.to(cpu))
            interventions.append(intervention.to(cpu))

        z0s = torch.cat(z0s, dim=0).detach()
        z1s = torch.cat(z1s, dim=0).detach()
        interventions = torch.cat(interventions, dim=0).detach()

    # Run ENCO
    adjacency_matrix = (
        run_enco(z0s, z1s, interventions, lambda_sparse=cfg.eval.enco_lambda, device=device)
        .cpu()
        .detach()
    )

    # Package as dict
    results = {
        f"enco_graph_{i}_{j}": adjacency_matrix[i, j].item()
        for i in range(model.dim_z)
        for j in range(model.dim_z)
    }

    return results


def epoch_schedules(cfg, model, epoch, optim):
    """Epoch-based schedulers"""

    # Pretraining?
    pretrain = cfg.training.pretrain_epochs is not None and epoch < cfg.training.pretrain_epochs
    if epoch == cfg.training.pretrain_epochs:
        logger.info(f"Stopping pretraining at epoch {epoch}")

    # Model interventions in SCM / noise model?
    model_interventions = (
        cfg.training.model_interventions_after_epoch is None
        or epoch >= cfg.training.model_interventions_after_epoch
    )
    if epoch == cfg.training.model_interventions_after_epoch:
        logger.info(f"Beginning to model intervention distributions at epoch {epoch}")

    # Freeze encoder?
    if cfg.training.freeze_encoder_epoch is not None and epoch == cfg.training.freeze_encoder_epoch:
        logger.info(f"Freezing encoder and decoder at epoch {epoch}")
        optim.param_groups[0]["lr"] = 0.0
        # model.encoder.freeze()
        # model.decoder.freeze()

    # Fix noise-centric model to a topological order encoder?
    if (
        "fix_topological_order_epoch" in cfg.training
        and cfg.training.fix_topological_order_epoch is not None
        and epoch == cfg.training.fix_topological_order_epoch
    ):
        logger.info(f"Determining topological order at epoch {epoch}")
        fix_topological_order(cfg, model, partition="val")

    # Introduce child variables
    if "introduce_child_interventions_after_epoch" not in cfg.training:
        introduce_child_interventions = True
    else:
        introduce_child_interventions = (
            cfg.training.introduce_child_interventions_after_epoch is None
            or epoch >= cfg.training.introduce_child_interventions_after_epoch
        )
    if (
        "introduce_child_interventions_after_epoch" in cfg.training
        and epoch == cfg.training.introduce_child_interventions_after_epoch
    ):
        logger.info(f"Introducing child interventions at epoch {epoch}")

    # Deterministic intervention encoders?
    if cfg.training.deterministic_intervention_encoder_after_epoch is None:
        deterministic_intervention_encoder = False
    else:
        deterministic_intervention_encoder = (
            epoch >= cfg.training.deterministic_intervention_encoder_after_epoch
        )
    if epoch == cfg.training.deterministic_intervention_encoder_after_epoch:
        logger.info(f"Switching to deterministic intervention encoder at epoch {epoch}")

    return (
        model_interventions,
        pretrain,
        deterministic_intervention_encoder,
        introduce_child_interventions,
    )


@torch.no_grad()
def fix_topological_order(cfg, model, partition="val", dataloader=None):
    """Fixes the topological order in an ILCM"""

    # This is only defined for noise-centric models (ILCMs)
    assert cfg.model.type == "intervention_noise_vae"

    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")
    model.to(device)

    # Dataloader
    if dataloader is None:
        dataloader = get_dataloader(cfg, partition, cfg.eval.batchsize)

    # Load data and compute noise encodings
    noise = []
    for x_batch, *_ in dataloader:
        x_batch = x_batch.to(device)
        noise.append(model.encode_to_noise(x_batch, deterministic=True).to(cpu))

    noise = torch.cat(noise, dim=0).detach()

    # Median values of each noise component (to be used as dummy values when masking)
    dummy_values = torch.median(noise, dim=0).values
    logger.info(f"Dummy noise encodings: {dummy_values}")

    if (
        "fix_topological_order" in cfg.training
        and "fix_using_map_interventions" in cfg.training.fix_topological_order
        and cfg.training.fix_topological_order.fix_using_map_interventions
    ):
        logger.info("Determining topological order using MAP interventions")
        intervention_posterior, intervention_label = calculate_intervention_posteriors(
            cfg, model, dataloader, device
        )
        average_intervention_posterior = calculate_average_intervention_posterior(
            cfg, intervention_posterior, intervention_label, device=device
        )
        _, MAP_interventions = average_intervention_posterior.max(dim=1)

        if len(torch.unique(MAP_interventions)) < cfg.model.dim_z + 1:
            raise ValueError(
                f"MAP interventions do not cover all components. MAP interventions: {MAP_interventions}."
            )
        else:
            # remove empty intervention
            MAP_components = MAP_interventions[MAP_interventions != 0] - 1

        topological_order = MAP_components
    elif (
        "fix_topological_order" in cfg.training
        and "fixed_order" in cfg.training.fix_topological_order
        and cfg.training.fix_topological_order.fixed_order is not None
    ):
        logger.info("Using fixed topological order")
        topological_order = cfg.training.fix_topological_order.fixed_order
    else:
        # Find topological order
        model = model.to(cpu)
        topological_order = find_topological_order(
            model,
            noise,
            concat_mask=(
                cfg.model.scm.concat_masks_to_parents
                if "concat_masks_to_parents" in cfg.model.scm
                else True
            ),
        )
    logger.info(f"Topological order: {topological_order}")

    # Fix topological order
    model.scm.set_causal_structure(
        None, "fixed_order", topological_order=topological_order, mask_values=dummy_values
    )
    model.to(device)


if __name__ == "__main__":
    main()
