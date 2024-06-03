# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import numpy as np
import torch

from ws_crl.encoder import gaussian_encode
from ws_crl.lcm.base import BaseLCM
from torch.distributions.categorical import Categorical


def sum_observational_and_counterfactual(x1, x2_batch, feature_dims=None):
    batchsize = x1.shape[0]

    if feature_dims is None:
        feature_dims = 1

    result = torch.concatenate(
        [
            torch.sum(x1, feature_dims).unsqueeze(1),
            batch_to_sequence(torch.sum(x2_batch, feature_dims), batchsize=batchsize),
        ],
        dim=1,
    ).sum(dim=1, keepdim=True)
    return result


class ILCM(BaseLCM):
    """
    Top-level class for generative models with
    - an SCM with a learned or fixed causal graph
    - separate encoder and decoder (i.e. a VAE) outputting noise encodings
    - VI over intervention targets
    """

    def __init__(
        self,
        causal_model,
        encoder,
        decoder,
        intervention_encoder,
        dim_z,
        intervention_prior=None,
        intervention_set="atomic_or_none",
        averaging_strategy="stochastic",
    ):
        super().__init__(
            causal_model,
            encoder,
            decoder=decoder,
            dim_z=dim_z,
            intervention_prior=intervention_prior,
            intervention_set=intervention_set,
        )
        self.intervention_encoder = intervention_encoder
        self.averaging_strategy = averaging_strategy

    def forward(
        self,
        x1,
        x2,
        interventions=None,
        beta=1.0,
        beta_intervention_target=None,
        pretrain_beta=None,
        full_likelihood=True,
        likelihood_reduction="sum",
        graph_mode="hard",
        graph_temperature=1.0,
        graph_samples=1,
        pretrain=False,
        model_interventions=True,
        deterministic_intervention_encoder=False,
        intervention_encoder_offset=1.0e-4,
        sample_from_intervention_posterior=False,
        **kwargs,
    ):
        """
        Evaluates an observed data pair.

        Arguments:
        ----------
        x1 : torch.Tensor of shape (batchsize, DIM_X,), dtype torch.float
            Observed data point (before the intervention)
        x2 : torch.Tensor of shape (batchsize, DIM_X,), dtype torch.float
            Observed data point (after the intervention)
        interventions : None or torch.Tensor of shape (batchsize, DIM_Z,), dtype torch.float
            If not None, specifies the interventions

        Returns:
        --------
        log_prob : torch.Tensor of shape (batchsize, 1), dtype torch.float
            If `interventions` is not None: Conditional log likelihood
            `log p(x1, x2 | interventions)`.
            If `interventions` is None: Marginal log likelihood `log p(x1, x2)`.
        outputs : dict with str keys and torch.Tensor values
            Detailed breakdown of the model outputs and internals.
        """

        # Check inputs
        if beta_intervention_target is None:
            beta_intervention_target = beta
        if pretrain_beta is None:
            pretrain_beta = beta

        batchsize = x1.shape[0]
        feature_dims = list(range(1, len(x1.shape)))
        assert torch.all(torch.isfinite(x1)) and torch.all(torch.isfinite(x2))
        assert interventions is None

        # Pretraining
        if pretrain:
            return self.forward_pretrain(
                x1,
                x2,
                beta=pretrain_beta,
                full_likelihood=full_likelihood,
                likelihood_reduction=likelihood_reduction,
            )

        # Get noise encoding means and stds
        e1_mean, e1_std = self.encoder.mean_std(x1)
        e2_mean, e2_std = self.encoder.mean_std(x2)

        # Compute intervention posterior
        intervention_posterior = self._encode_intervention(
            e1_mean, e2_mean, intervention_encoder_offset, deterministic_intervention_encoder
        )

        # Regularization terms
        e_norm, consistency_mse, _ = self._compute_latent_reg_consistency_mse(
            e1_mean, e1_std, e2_mean, e2_std, feature_dims, x1, x2, beta=beta
        )

        # Iterate over interventions
        log_posterior_eps, log_prior_eps = 0, 0
        log_posterior_int, log_prior_int, log_likelihood = 0, 0, 0
        mse, inverse_consistency_mse = 0, 0
        outputs = {}

        for (
            intervention,
            weight,
        ) in self._iterate_interventions(
            intervention_posterior,
            deterministic_intervention_encoder,
            batchsize,
            sample_from_intervention_posterior=sample_from_intervention_posterior,
        ):
            # Sample from e1, e2 given intervention (including the projection to the counterfactual
            # manifold)
            e1_proj, e2_proj, log_posterior1_proj, log_posterior2_proj = self._project_and_sample(
                e1_mean, e1_std, e2_mean, e2_std, intervention
            )

            # Compute ELBO terms
            (
                log_likelihood_proj,
                log_posterior_eps_proj,
                log_posterior_int_proj,
                log_prior_eps_proj,
                log_prior_int_proj,
                mse_proj,
                inverse_consistency_mse_proj,
                outputs_,
            ) = self._compute_elbo_terms(
                x1,
                x2,
                e1_proj,
                e2_proj,
                feature_dims,
                full_likelihood,
                intervention,
                likelihood_reduction,
                log_posterior1_proj,
                log_posterior2_proj,
                weight,
                graph_mode=graph_mode,
                graph_samples=graph_samples,
                graph_temperature=graph_temperature,
                model_interventions=model_interventions,
            )

            # Sum up results
            log_posterior_eps += weight * log_posterior_eps_proj
            log_posterior_int += weight * log_posterior_int_proj
            log_prior_eps += weight * log_prior_eps_proj
            log_prior_int += weight * log_prior_int_proj
            log_likelihood += weight * log_likelihood_proj
            mse += weight * mse_proj
            inverse_consistency_mse += inverse_consistency_mse_proj

            # Some more bookkeeping
            for key, val in outputs_.items():
                val = val.unsqueeze(1)
                if key in outputs:
                    outputs[key] = torch.cat((outputs[key], val), dim=1)
                else:
                    outputs[key] = val

        loss = self._compute_outputs(
            beta,
            beta_intervention_target,
            consistency_mse,
            e1_std,
            e2_std,
            e_norm,
            intervention_posterior,
            log_likelihood,
            log_posterior_eps,
            log_posterior_int,
            log_prior_eps,
            log_prior_int,
            mse,
            outputs,
            inverse_consistency_mse,
        )

        return loss, outputs

    def _encode_intervention(
        self, e1_mean, e2_mean, intervention_encoder_offset, deterministic_intervention_encoder
    ):
        intervention_encoder_inputs = torch.cat((e1_mean, e2_mean - e1_mean), dim=1)
        intervention_posterior = self.intervention_encoder(
            intervention_encoder_inputs, eps=intervention_encoder_offset
        )
        assert torch.all(torch.isfinite(intervention_posterior))

        # Deterministic intervention encoder: binarize, but use STE for gradients
        if deterministic_intervention_encoder:
            batchsize = e1_mean.shape[0]
            most_likely_intervention = torch.argmax(intervention_posterior, dim=1)  # (batchsize,)
            det_posterior = torch.zeros_like(intervention_posterior)
            det_posterior[torch.arange(batchsize), most_likely_intervention] = 1.0
            intervention_posterior = (
                det_posterior.detach() + intervention_posterior - intervention_posterior.detach()
            )

        return intervention_posterior

    def forward_pretrain(self, x1, x2, beta, full_likelihood=False, likelihood_reduction="sum"):
        assert torch.all(torch.isfinite(x1)) and torch.all(torch.isfinite(x2))
        feature_dims = list(range(1, len(x1.shape)))

        # Get noise encoding means and stds
        e1_mean, e1_std = self.encoder.mean_std(x1)
        e2_mean, e2_std = self.encoder.mean_std(x2)
        encoder_std = torch.mean(torch.concat([e1_std, e2_std]), dim=1, keepdim=True)

        # Regularization terms
        e_norm, consistency_mse, beta_vae_loss = self._compute_latent_reg_consistency_mse(
            e1_mean,
            e1_std,
            e2_mean,
            e2_std,
            feature_dims,
            x1,
            x2,
            beta=beta,
            full_likelihood=full_likelihood,
            likelihood_reduction=likelihood_reduction,
        )

        # Pretraining loss
        outputs = dict(
            z_regularization=e_norm, consistency_mse=consistency_mse, encoder_std=encoder_std
        )

        return beta_vae_loss, outputs

    def encode_to_noise(self, x, deterministic=True):
        """
        Given data x, returns the noise encoding.

        Arguments:
        ----------
        x : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Data point to be encoded.
        deterministic : bool, optional
            If True, enforces deterministic encoding (e.g. by not adding noise in a Gaussian VAE).

        Returns:
        --------
        e : torch.Tensor of shape (batchsize, DIM_Z), dtype torch.float
            Noise encoding phi_e(x)
        """

        e, _ = self.encoder(x, deterministic=deterministic)
        return e

    def encode_to_causal(self, x, deterministic=True):
        """
        Given data x, returns the causal encoding.

        Arguments:
        ----------
        x : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Data point to be encoded.
        deterministic : bool, optional
            If True, enforces deterministic encoding (e.g. by not adding noise in a Gaussian VAE).

        Returns:
        --------
        inputs : torch.Tensor of shape (batchsize, DIM_Z), dtype torch.float
            Causal-variable encoding phi_z(x)
        """

        e, _ = self.encoder(x, deterministic=deterministic)
        adjacency_matrix = self._sample_adjacency_matrices(mode="deterministic", n=x.shape[0])
        z = self.scm.noise_to_causal(e, adjacency_matrix=adjacency_matrix)
        return z

    def decode_noise(self, e, deterministic=True):
        """
        Given noise encoding e, returns data x.

        Arguments:
        ----------
        e : torch.Tensor of shape (batchsize, DIM_Z), dtype torch.float
            Noise-encoded data.
        deterministic : bool, optional
            If True, enforces deterministic decoding (e.g. by not adding noise in a Gaussian VAE).

        Returns:
        --------
        x : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Decoded data point.
        """

        x, _ = self.decoder(e, deterministic=deterministic)
        return x

    def decode_causal(self, z, deterministic=True):
        """
        Given causal latents inputs, returns data x.

        Arguments:
        ----------
        inputs : torch.Tensor of shape (batchsize, DIM_Z), dtype torch.float
            Causal latent variables.
        deterministic : bool, optional
            If True, enforces deterministic decoding (e.g. by not adding noise in a Gaussian VAE).

        Returns:
        --------
        x : torch.Tensor of shape (batchsize, DIM_X), dtype torch.float
            Decoded data point.
        """

        adjacency_matrix = self._sample_adjacency_matrices(mode="deterministic", n=z.shape[0])
        e = self.scm.causal_to_noise(z, adjacency_matrix=adjacency_matrix)
        x, _ = self.decoder(e, deterministic=deterministic)
        return x

    def log_likelihood(self, x1, x2, interventions=None, n_latent_samples=20, **kwargs):
        """
        Computes estimate of the log likelihood using importance weighting, like in IWAE.

        `log p(x) = log E_{inputs ~ q(inputs|x)} [ p(x|inputs) p(inputs) / q(inputs|x) ]`
        """

        # Copy each sample n_latent_samples times
        x1_ = self._expand(x1, repeats=n_latent_samples)
        x2_ = self._expand(x2, repeats=n_latent_samples)
        interventions_ = self._expand(interventions, repeats=n_latent_samples)

        # Evaluate ELBO
        negative_elbo, _ = self.forward(x1_, x2_, interventions_, beta=1.0)

        # Compute importance-weighted estimate of log likelihood
        log_likelihood = self._contract(-negative_elbo, mode="logmeanexp", repeats=n_latent_samples)

        return log_likelihood

    def encode_decode(self, x, deterministic=True):
        """Auto-encode data and return reconstruction"""
        eps = self.encode_to_noise(x, deterministic=deterministic)
        x_reco = self.decode_noise(eps, deterministic=deterministic)

        return x_reco

    def encode_decode_pair(self, x1, x2, deterministic=True):
        """Auto-encode data pair and return latent representation and reconstructions"""

        # Get noise encoding means and stds
        e1_mean, e1_std = self.encoder.mean_std(x1)
        e2_mean, e2_std = self.encoder.mean_std(x2)

        # Compute intervention posterior
        intervention_encoder_inputs = torch.cat((e1_mean, e2_mean - e1_mean), dim=1)
        intervention_posterior = self.intervention_encoder(intervention_encoder_inputs)

        # Determine most likely intervention
        most_likely_intervention_idx = torch.argmax(intervention_posterior, dim=1).flatten()
        intervention = self._interventions[most_likely_intervention_idx]

        # Project to manifold
        e1_proj, e2_proj, log_posterior1_proj, log_posterior2_proj = self._project_and_sample(
            e1_mean, e1_std, e2_mean, e2_std, intervention, deterministic=deterministic
        )

        # Project back to data space
        x1_reco = self.decode_noise(e1_proj)
        x2_reco = self.decode_noise(e2_proj)

        return (
            x1_reco,
            x2_reco,
            e1_mean,
            e2_mean,
            e1_proj,
            e2_proj,
            intervention_posterior,
            most_likely_intervention_idx,
            intervention,
        )

    def infer_intervention(
        self,
        x1,
        x2,
        deterministic=True,
    ):
        """Given data pair, infer intervention"""

        (
            x1_reco,
            x2_reco,
            e1_mean,
            e2_mean,
            e1_proj,
            e2_proj,
            intervention_posterior,
            most_likely_intervention_idx,
            intervention,
        ) = self.encode_decode_pair(x1, x2, deterministic=deterministic)

        return most_likely_intervention_idx, None, x2_reco

    def _iterate_interventions(
        self,
        intervention_posterior,
        deterministic_intervention_encoder,
        batchsize,
        sample_from_intervention_posterior=False,
    ):
        if deterministic_intervention_encoder:
            most_likely_intervention = torch.argmax(intervention_posterior, dim=1)  # (batchsize,)
            interventions = self._interventions.unsqueeze(0).expand(
                (batchsize, self._interventions.shape[0], self._interventions.shape[1])
            )
            intervention = interventions[torch.arange(batchsize), most_likely_intervention, :]
            weight = torch.ones((batchsize, 1), device=intervention_posterior.device)
            yield intervention, weight
        else:
            if not sample_from_intervention_posterior:
                for intervention, weight in zip(self._interventions, intervention_posterior.T):
                    intervention = intervention.unsqueeze(0).expand(
                        (batchsize, intervention.shape[0])
                    )
                    weight = weight.unsqueeze(1)  # (batchsize, 1)
                    yield intervention, weight
            else:
                # sample from intervention posterior
                intervention_samples = []
                for i in range(batchsize):
                    intervention_sample = Categorical(intervention_posterior[i]).sample()
                    intervention_sample = self._interventions[intervention_sample]
                    intervention_samples.append(intervention_sample)
                intervention_samples = torch.stack(intervention_samples, dim=0)
                yield intervention_samples, torch.ones(
                    (batchsize, 1), device=intervention_posterior.device
                )

    def _project_and_sample(
        self, e1_mean, e1_std, e2_mean, e2_std, intervention, deterministic=False
    ):
        # Project to manifold
        (
            e1_mean_proj,
            e1_std_proj,
            e2_mean_proj,
            e2_std_proj,
        ) = self._project_to_manifold(intervention, e1_mean, e1_std, e2_mean, e2_std)

        # Sample noise
        e1_proj, log_posterior1_proj = gaussian_encode(
            e1_mean_proj, e1_std_proj, deterministic=deterministic
        )
        e2_proj, log_posterior2_proj = gaussian_encode(
            e2_mean_proj, e2_std_proj, deterministic=deterministic, reduction="none"
        )

        # Sampling should preserve counterfactual consistency
        e2_proj = intervention * e2_proj + (1.0 - intervention) * e1_proj

        log_posterior2_proj = torch.sum(
            log_posterior2_proj * intervention, dim=list(range(1, len(e2_proj.shape)))
        ).unsqueeze(-1)

        return e1_proj, e2_proj, log_posterior1_proj, log_posterior2_proj

    def _project_to_manifold(self, intervention, e1_mean, e1_std, e2_mean, e2_std):
        if self.averaging_strategy == "z2":
            lam = torch.ones_like(e1_mean)
        elif self.averaging_strategy in ["average", "mean"]:
            lam = 0.5 * torch.ones_like(e1_mean)
        elif self.averaging_strategy == "stochastic":
            lam = torch.rand_like(e1_mean)
        else:
            raise ValueError(f"Unknown averaging strategy {self.averaging_strategy}")

        projection_mean = lam * e1_mean + (1.0 - lam) * e2_mean
        projection_std = lam * e1_std + (1.0 - lam) * e2_std

        e1_mean = intervention * e1_mean + (1.0 - intervention) * projection_mean
        e1_std = intervention * e1_std + (1.0 - intervention) * projection_std
        e2_mean = intervention * e2_mean + (1.0 - intervention) * projection_mean
        e2_std = intervention * e2_std + (1.0 - intervention) * projection_std

        return e1_mean, e1_std, e2_mean, e2_std

    def _compute_latent_reg_consistency_mse(
        self,
        e1_mean,
        e1_std,
        e2_mean,
        e2_std,
        feature_dims,
        x1,
        x2,
        beta,
        full_likelihood=False,
        likelihood_reduction="sum",
    ):
        e1, log_posterior1 = gaussian_encode(e1_mean, e1_std, deterministic=False)
        e2, log_posterior2 = gaussian_encode(e2_mean, e2_std, deterministic=False)

        # TODO: use sequence data instead of reshaping batch?

        # Compute latent regularizer (useful early in training)
        e_norm = sum_observational_and_counterfactual(e1**2, e2**2)

        # Compute consistency MSE
        consistency_x1_reco, log_likelihood1 = self.decoder(
            e1,
            eval_likelihood_at=x1,
            deterministic=True,
            full=full_likelihood,
            reduction=likelihood_reduction,
        )
        consistency_x2_reco, log_likelihood2 = self.decoder(
            e2,
            eval_likelihood_at=x2,
            deterministic=True,
            full=full_likelihood,
            reduction=likelihood_reduction,
        )

        consistency_mse = sum_observational_and_counterfactual(
            (consistency_x1_reco - x1) ** 2,
            (consistency_x2_reco - x2) ** 2,
            feature_dims=feature_dims,
        )

        # Compute prior and beta-VAE loss (for pre-training)
        log_prior1 = torch.sum(
            self.scm.base_density.log_prob(e1.reshape((-1, 1))).reshape((-1, self.dim_z)),
            dim=1,
            keepdim=True,
        )
        log_prior2 = torch.sum(
            self.scm.base_density.log_prob(e2.reshape((-1, 1))).reshape((-1, self.dim_z)),
            dim=1,
            keepdim=True,
        )

        log_likelihood = sum_observational_and_counterfactual(log_likelihood1, log_likelihood2)
        log_prior = sum_observational_and_counterfactual(log_prior1, log_prior2)
        log_posterior = sum_observational_and_counterfactual(log_posterior1, log_posterior2)

        beta_vae_loss = -log_likelihood + beta * (log_posterior - log_prior)

        return e_norm, consistency_mse, beta_vae_loss

    def _compute_outputs(
        self,
        beta,
        beta_intervention_target,
        consistency_mse,
        e1_std,
        e2_std,
        e_norm,
        intervention_posterior,
        log_likelihood,
        log_posterior_eps,
        log_posterior_int,
        log_prior_eps,
        log_prior_int,
        mse,
        outputs,
        inverse_consistency_mse,
        intervened_sequence_loss=None,
        unintervened_sequence_loss=None,
    ):
        # Put together to compute the ELBO and beta-VAE loss
        kl_int = log_posterior_int - log_prior_int
        kl_eps = log_posterior_eps - log_prior_eps
        log_posterior = log_posterior_int + log_posterior_eps
        log_prior = log_prior_int + log_prior_eps
        kl = kl_eps + kl_int
        elbo = log_likelihood - kl
        beta_vae_loss = -log_likelihood + beta * kl_eps + beta_intervention_target * kl_int

        # Track individual components
        outputs["elbo"] = elbo
        outputs["beta_vae_loss"] = beta_vae_loss
        outputs["kl"] = kl
        outputs["kl_intervention_target"] = kl_int
        outputs["kl_e"] = kl_eps
        outputs["log_likelihood"] = log_likelihood
        outputs["log_posterior"] = log_posterior
        outputs["log_prior"] = log_prior
        outputs["intervention_posterior"] = intervention_posterior
        outputs["mse"] = mse
        outputs["consistency_mse"] = consistency_mse
        outputs["inverse_consistency_mse"] = inverse_consistency_mse
        outputs["z_regularization"] = e_norm
        outputs["encoder_std"] = torch.mean(torch.concat([e1_std, e2_std]), dim=1, keepdim=True)

        outputs["intervened_sequence_loss"] = intervened_sequence_loss
        outputs["unintervened_sequence_loss"] = unintervened_sequence_loss

        return beta_vae_loss

    def _compute_elbo_terms(
        self,
        x1,
        x2,
        e1_proj,
        e2_proj,
        feature_dims,
        full_likelihood,
        intervention,
        likelihood_reduction,
        log_posterior1_proj,
        log_posterior2_proj,
        weight,
        model_interventions=True,
        sequence_length=None,
        **prior_kwargs,
    ):
        batchsize = x1.shape[0]
        # Compute posterior q(e1, e2_I | I)
        log_posterior_eps_proj = log_posterior1_proj + log_posterior2_proj
        assert torch.all(torch.isfinite(log_posterior_eps_proj))

        # Compute posterior q(I)
        log_posterior_int_proj = weight * torch.log(weight)

        # Decode compute log likelihood / reconstruction error
        x1_reco_proj, log_likelihood1_proj, _ = self.decoder(
            e1_proj,
            eval_likelihood_at=x1,
            deterministic=True,
            return_std=True,
            full=full_likelihood,
            reduction=likelihood_reduction,
        )
        x2_reco_proj, log_likelihood2_proj, _ = self.decoder(
            e2_proj,
            eval_likelihood_at=x2,
            deterministic=True,
            return_std=True,
            full=full_likelihood,
            reduction=likelihood_reduction,
        )
        # log_likelihood_proj = log_likelihood1_proj + log_likelihood2_proj
        log_likelihood_proj = sum_observational_and_counterfactual(
            log_likelihood1_proj, log_likelihood2_proj
        )
        assert torch.all(torch.isfinite(log_likelihood_proj))

        # Compute MSE
        mse_proj = sum_observational_and_counterfactual(
            (x1_reco_proj - x1) ** 2, (x2_reco_proj - x2) ** 2, feature_dims=feature_dims
        )

        # Compute inverse consistency MSE: |z - encode(decode(z))|^2
        e1_reencoded = self.encode_to_noise(x1_reco_proj, deterministic=False)
        e2_reencoded = self.encode_to_noise(x2_reco_proj, deterministic=False)

        inverse_consistency_mse_proj = sum_observational_and_counterfactual(
            (e1_reencoded - e1_proj) ** 2, (e2_reencoded - e2_proj) ** 2
        )

        # Expand to sequence length to match each e2 with corresponding e1
        if sequence_length is not None:
            e1_proj = expand_observational_to_sequence_length(e1_proj, sequence_length)
            intervention = expand_observational_to_sequence_length(intervention, sequence_length)

        # Compute prior p(e1, e2 | I)
        log_prior_eps_proj, outputs = self.scm.log_prob_noise_weakly_supervised(
            e1_proj,
            e2_proj,
            intervention,
            adjacency_matrix=None,
            include_intervened=model_interventions,
            include_nonintervened=False,
        )
        assert torch.all(torch.isfinite(log_prior_eps_proj))

        log_prior_eps_proj = log_prior_eps_proj.reshape(batchsize, -1).sum(dim=1, keepdim=True)

        # Compute prior pi(I) = 1 / n_interventions
        log_prior_int_proj = -np.log(self.n_interventions) * torch.ones_like(log_prior_eps_proj)

        return (
            log_likelihood_proj,
            log_posterior_eps_proj,
            log_posterior_int_proj,
            log_prior_eps_proj,
            log_prior_int_proj,
            mse_proj,
            inverse_consistency_mse_proj,
            outputs,
        )

    def load_state_dict(self, state_dict, strict=True):
        """Overloading the state dict loading so we can compute ancestor structure"""
        super().load_state_dict(state_dict, strict)
        self.scm._compute_ancestors()


def ranking_loss(startpoints, sequences, reduction="mean"):
    pd = sequences.unsqueeze(-2) - sequences.unsqueeze(-1)

    # Determine the direction of the ranking
    forwards = sequences[:, -1] - startpoints >= 0

    # flip to make all go in the same direction
    pd = torch.where(forwards.view(-1, 1, 1), -pd, pd)

    # Select the upper triangular part of the distance matrix to only consider one direction
    selected_pd = torch.triu(pd, diagonal=0)

    # Differences that are in the wrong direction
    backwards_diffs = torch.max(torch.zeros_like(selected_pd), selected_pd)

    if reduction == "sum":
        loss = torch.sum(backwards_diffs, dim=list(range(1, len(backwards_diffs.shape)))).unsqueeze(
            -1
        )
        return loss
    if reduction == "mean":
        loss = torch.mean(
            backwards_diffs, dim=list(range(1, len(backwards_diffs.shape)))
        ).unsqueeze(-1)
        return loss
    raise NotImplementedError(f"Reduction {reduction} not implemented")


def sequence_to_batch(x_sequence):
    return x_sequence.reshape(-1, *x_sequence.shape[2:])  # (B*T, *x_dim)


def batch_to_sequence(x, batchsize):
    return x.reshape(batchsize, -1, *x.shape[1:])  # (B, T, *x_dim)


def expand_observational_to_sequence_length(x, sequence_length):
    return x.unsqueeze(1).expand(-1, sequence_length, -1).reshape(-1, *x.shape[1:])  # (B*T, *x_dim)


class MonotonousEffectSequenceILCM(ILCM):

    def __init__(
        self,
        causal_model,
        encoder,
        decoder,
        intervention_encoder,
        dim_z,
        sequence_length,
        intervention_prior=None,
        intervention_set="atomic_or_none",
        averaging_strategy="stochastic",
    ):
        super().__init__(
            causal_model,
            encoder,
            decoder=decoder,
            dim_z=dim_z,
            intervention_prior=intervention_prior,
            intervention_set=intervention_set,
            intervention_encoder=intervention_encoder,
            averaging_strategy=averaging_strategy,
        )
        self.sequence_length = sequence_length

    def forward(
        self,
        x1,  # (B, *x_dim)
        x2_sequence,  # (B, T, *x_dim)
        interventions=None,
        beta=1.0,
        beta_intervention_target=None,
        pretrain_beta=None,
        full_likelihood=True,
        likelihood_reduction="sum",
        graph_mode="hard",
        graph_temperature=1.0,
        graph_samples=1,
        pretrain=False,
        model_interventions=True,
        deterministic_intervention_encoder=False,
        intervention_encoder_offset=1.0e-4,
        **kwargs,
    ):
        # Check inputs
        if beta_intervention_target is None:
            beta_intervention_target = beta
        if pretrain_beta is None:
            pretrain_beta = beta

        batchsize = x1.shape[0]
        feature_dims = list(range(1, len(x1.shape)))
        assert torch.all(torch.isfinite(x1)) and torch.all(torch.isfinite(x2_sequence))
        assert interventions is None

        x2_batch = sequence_to_batch(x2_sequence)  # (B*T, *x_dim)
        x2_last = x2_sequence[:, -1]

        # Pretraining
        if pretrain:
            return self.forward_pretrain(
                x1,
                x2_batch,
                beta=pretrain_beta,
                full_likelihood=full_likelihood,
                likelihood_reduction=likelihood_reduction,
            )

        # Get noise encoding means and stds
        e1_mean, e1_std = self.encoder.mean_std(x1)
        e2_mean_batch, e2_std_batch = self.encoder.mean_std(x2_batch)

        e1_mean_sequence = e1_mean.unsqueeze(1)
        e1_std_sequence = e1_std.unsqueeze(1)

        e2_mean_sequence = batch_to_sequence(e2_mean_batch, batchsize)
        e2_std_sequence = batch_to_sequence(e2_std_batch, batchsize)

        e2_mean_last = e2_mean_sequence[:, -1]
        e2_std_last = e2_std_sequence[:, -1]

        # Compute intervention posterior between observational and last sequence element
        intervention_posterior_last = self._encode_intervention(
            e1_mean,
            e2_mean_last,
            intervention_encoder_offset,
            deterministic_intervention_encoder,
        )

        # Regularization terms
        e_norm, consistency_mse, _ = self._compute_latent_reg_consistency_mse(
            e1_mean,
            e1_std,
            # e2_mean_batch,
            e2_mean_last,
            # e2_std_batch,
            e2_std_last,
            feature_dims,
            x1,
            # x2_batch,
            x2_last,
            beta=beta,
        )

        # Iterate over interventions
        log_posterior_eps, log_prior_eps = 0, 0
        log_posterior_int, log_prior_int, log_likelihood = 0, 0, 0
        mse, inverse_consistency_mse = 0, 0
        outputs = {}

        intervened_sequence_loss, unintervened_sequence_loss = torch.zeros(
            batchsize, 1, device=x1.device
        ), torch.zeros(batchsize, 1, device=x1.device)

        for (
            intervention,
            weight,
        ) in self._iterate_interventions(
            # Iterate over intervention posterior of last sequence element
            intervention_posterior_last,
            deterministic_intervention_encoder,
            batchsize,
        ):
            intervention_sequence = intervention.unsqueeze(1)
            # Sample from e1, e2 given intervention (including the projection to the counterfactual
            # manifold)
            e1_proj_sequence, e2_proj_sequence, log_posterior1_proj, log_posterior2_proj = (
                self._project_and_sample(
                    e1_mean_sequence,
                    e1_std_sequence,
                    e2_mean_sequence,
                    e2_std_sequence,
                    intervention_sequence,
                )
            )
            e_proj_sequence = torch.concatenate(
                [e1_proj_sequence, e2_proj_sequence], dim=1
            )  # (B, T+1, z_dim)

            # e1_proj, e2_last_proj, log_posterior1_proj, log_posterior2_proj = self._project_and_sample(
            #     e1_mean,
            #     e1_std,
            #     e2_mean_last,
            #     e2_std_last,
            #     intervention,
            # )
            # e_proj_sequence = torch.concatenate(
            #     [e1_proj,  e2_last_proj], dim=1
            # )  # (B, T+1, z_dim)

            # assert (intervention[0] == intervention).all()
            # NOTE: Assuming atomic interventions for now
            assert ((intervention.sum(-1) == 0) | (intervention.sum(-1) == 1)).all()

            # Select samples that were intervened on
            sample_was_intervened_on = torch.any(intervention, dim=1)
            intervention_non_empty = intervention[sample_was_intervened_on]

            if len(intervention_non_empty) > 0:
                # NOTE: num_intervened = 1 for non-empty atomic interventions

                # Samples where intervention is non-empty
                e_proj_sequence_intervened = e_proj_sequence[sample_was_intervened_on]

                # Select component that was intervened on
                intervened_e_proj_sequence = torch.masked_select(
                    e_proj_sequence_intervened, intervention_non_empty.unsqueeze(1).bool()
                ).view(
                    -1, self.sequence_length + 1, 1
                )  # (B_intervened, T+1, num_intervened)

                intervened_e_proj_sequence_batch = intervened_e_proj_sequence.transpose(
                    1, 2
                ).reshape(
                    -1, self.sequence_length + 1
                )  # (B_intervened*num_intervened, T+1)
                startpoints = intervened_e_proj_sequence_batch[:, 0]
                intervened_sequence_loss_proj = ranking_loss(
                    startpoints,
                    intervened_e_proj_sequence_batch,
                    reduction="mean",
                )
            else:
                intervened_sequence_loss_proj = 0

            # TODO: remove / make optional
            # Should also be implicitly there by projection

            # if e_proj_unintervened_sequence.shape[2] != 0:
            #     pd_e_proj_unintervened = e_proj_unintervened_sequence.unsqueeze(
            #         -2
            #     ) - e_proj_unintervened_sequence.unsqueeze(-1)
            #     unintervened_sequence_loss_proj = torch.sum(pd_e_proj_unintervened**2).unsqueeze(0)
            # else:
            #     unintervened_sequence_loss_proj = 0
            unintervened_sequence_loss_proj = torch.zeros(
                batchsize, 1, device=e_proj_sequence.device
            )

            e2_proj_last = e2_proj_sequence[:, -1]

            # Compute ELBO terms
            (
                log_likelihood_proj,
                log_posterior_eps_proj,
                log_posterior_int_proj,
                log_prior_eps_proj,
                log_prior_int_proj,
                mse_proj,
                inverse_consistency_mse_proj,
                outputs_,
            ) = self._compute_elbo_terms(
                x1,
                x2_last,
                e1_proj_sequence[:, 0],
                e2_proj_last,
                feature_dims,
                full_likelihood,
                intervention,
                likelihood_reduction,
                log_posterior1_proj[:, 0],
                log_posterior2_proj[:, -1],
                weight,
                graph_mode=graph_mode,
                graph_samples=graph_samples,
                graph_temperature=graph_temperature,
                model_interventions=model_interventions,
                # sequence_length=self.sequence_length,
                sequence_length=None,
            )

            # Sum up results
            log_posterior_eps += weight * log_posterior_eps_proj
            log_posterior_int += weight * log_posterior_int_proj
            log_prior_eps += weight * log_prior_eps_proj
            log_prior_int += weight * log_prior_int_proj
            log_likelihood += weight * log_likelihood_proj
            mse += weight * mse_proj
            inverse_consistency_mse += inverse_consistency_mse_proj

            intervened_sequence_loss = intervened_sequence_loss.index_add(
                0,
                sample_was_intervened_on.nonzero().squeeze(),
                weight[sample_was_intervened_on] * intervened_sequence_loss_proj,
            )
            # unintervened_sequence_loss = unintervened_sequence_loss.index_add(
            #     0,
            #     sample_was_intervened_on.nonzero().squeeze(),
            #     weight[sample_was_intervened_on] * unintervened_sequence_loss_proj,
            # )
            unintervened_sequence_loss = unintervened_sequence_loss_proj

            # Some more bookkeeping
            for key, val in outputs_.items():
                val = val.unsqueeze(1)
                if key in outputs:
                    outputs[key] = torch.cat((outputs[key], val), dim=1)
                else:
                    outputs[key] = val

        loss = self._compute_outputs(
            beta,
            beta_intervention_target,
            consistency_mse,
            e1_std,
            e2_std_batch,
            e_norm,
            # TODO: use only last intervention posterior?
            intervention_posterior_last,
            log_likelihood,
            log_posterior_eps,
            log_posterior_int,
            log_prior_eps,
            log_prior_int,
            mse,
            outputs,
            inverse_consistency_mse,
            intervened_sequence_loss,
            unintervened_sequence_loss,
        )

        return loss, outputs

    def _project_to_manifold(self, intervention, e1_mean, e1_std, e2_mean, e2_std):
        batchsize = e1_mean.shape[0]
        if self.averaging_strategy == "z2":
            # lam = torch.ones_like(e1_mean)
            raise NotImplementedError()
        elif self.averaging_strategy in ["average", "mean"]:
            # lam = 0.5 * torch.ones_like(e1_mean)
            raise NotImplementedError()
        elif self.averaging_strategy == "stochastic":
            # lam = torch.rand_like(e1_mean)
            lam = torch.rand((batchsize, self.sequence_length + 1, self.dim_z)).to(e1_mean.device)
            lam = lam / torch.sum(lam, dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown averaging strategy {self.averaging_strategy}")

        # TODO: test with pairwise averages
        projection_mean = torch.concatenate(
            [lam[:, 0:1, :] * e1_mean, lam[:, 1:, :] * e2_mean], dim=1
        ).sum(dim=1, keepdim=True)
        projection_std = torch.concatenate(
            [lam[:, 0:1, :] * e1_std, lam[:, 1:, :] * e2_std], dim=1
        ).sum(dim=1, keepdim=True)

        e1_mean = intervention * e1_mean + (1.0 - intervention) * projection_mean
        e1_std = intervention * e1_std + (1.0 - intervention) * projection_std
        e2_mean = intervention * e2_mean + (1.0 - intervention) * projection_mean
        e2_std = intervention * e2_std + (1.0 - intervention) * projection_std

        return e1_mean, e1_std, e2_mean, e2_std

    def encode_decode_pair(self, x1, x2_sequence, deterministic=True):
        batchsize = x1.shape[0]

        x2_batch = sequence_to_batch(x2_sequence)  # (B*T, *x_dim)

        # Get noise encoding means and stds
        e1_mean, e1_std = self.encoder.mean_std(x1)
        e2_mean_batch, e2_std_batch = self.encoder.mean_std(x2_batch)

        e1_mean_sequence = e1_mean.unsqueeze(1)
        e1_std_sequence = e1_std.unsqueeze(1)

        e2_mean_sequence = batch_to_sequence(e2_mean_batch, batchsize)
        e2_std_sequence = batch_to_sequence(e2_std_batch, batchsize)

        e2_mean_last = e2_mean_sequence[:, -1]

        # Compute intervention posterior
        intervention_encoder_inputs = torch.cat((e1_mean, e2_mean_last - e1_mean), dim=1)
        intervention_posterior = self.intervention_encoder(intervention_encoder_inputs)

        # Determine most likely intervention
        most_likely_intervention_idx = torch.argmax(intervention_posterior, dim=1).flatten()
        intervention = self._interventions[most_likely_intervention_idx]

        intervention_sequence = intervention.unsqueeze(1)

        # Project to manifold
        e1_proj_sequence, e2_proj_sequence, log_posterior1_proj, log_posterior2_proj = (
            self._project_and_sample(
                e1_mean_sequence,
                e1_std_sequence,
                e2_mean_sequence,
                e2_std_sequence,
                intervention_sequence,
                deterministic=deterministic,
            )
        )

        e1_proj = e1_proj_sequence[:, 0]

        # Project back to data space
        x1_reco = self.decode_noise(e1_proj)
        x2_reco_batch = self.decode_noise(sequence_to_batch(e2_proj_sequence))

        x2_reco_sequence = batch_to_sequence(x2_reco_batch, batchsize)

        return (
            x1_reco,
            x2_reco_sequence,
            e1_mean,
            e2_mean_sequence,
            e1_proj,
            e2_proj_sequence,
            intervention_posterior,
            most_likely_intervention_idx,
            intervention,
        )
