"""Implements single diffusion steps and partial diffusion for Boltz-1.

Author: Karson Chrispens
Last Updated: 10 March 2026, by Joyce Mo

Main changes from Joyce:
- corrected import paths for Boltz1 and pad_dim
- save tensors 
"""

from typing import Dict, Optional, Tuple, Union
import torch
from math import sqrt

from pathlib import Path
from dataclasses import asdict
from boltz.model.models.boltz1 import Boltz1 # joyce edited to boltz.model.models, not model
from boltz.main import BoltzDiffusionParams
from boltz.model.modules.utils import default, center_random_augmentation
from boltz.model.loss.diffusion import weighted_rigid_align
from dataclasses import asdict, dataclass

#from adp3d.utils.utility import try_gpu
from boltz.main import check_inputs, process_inputs, BoltzProcessedInput
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.types import Manifest, Structure
from boltz.data.pad import pad_dim # joyce edited to correct import paths 
import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn.functional as F
from typing import Iterable, Mapping, Union, Sequence, Optional

def _gaussian_kernel1d(sigma: float, dtype: torch.dtype, device: torch.device, truncate: float = 3.0) -> torch.Tensor:
    """
    Build a 1D Gaussian kernel normalized to sum=1. If sigma <= 0, returns [1.0].
    """
    if sigma is None or sigma <= 0:
        return torch.ones(1, dtype=dtype, device=device)
    # radius is ceil(truncate*sigma)
    radius = int(truncate * float(sigma) + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _conv1d_along_axis(
    t: torch.Tensor,
    kernel: torch.Tensor,
    axis: int,
) -> torch.Tensor:
    """
    Convolve tensor t with 1D kernel along a given axis using reflect padding.
    Handles small sizes by trimming the kernel (and renormalizing) so that pad <= size-1.
    """
    ndim = t.ndim
    axis = axis % ndim
    # Move target axis to the last position
    order = [i for i in range(ndim) if i != axis] + [axis]
    t_last = t.permute(order)
    size = t_last.shape[-1]

    full_len = int(kernel.numel())
    center = full_len // 2
    pad = center

    # Reflect padding requires pad <= size-1
    max_pad = max(0, min(pad, size - 1))
    if max_pad != pad:
        # Trim kernel around center and renormalize
        left = center - max_pad
        right = center + max_pad + 1
        kernel = kernel[left:right]
        kernel = kernel / kernel.sum()
        pad = max_pad

    # Flatten all leading dims to batch, keep a single "channel"
    B = int(t_last.numel() // size)
    x = t_last.reshape(B, 1, size)

    # Reflect-pad and convolve
    if pad > 0:
        x = F.pad(x, (pad, pad), mode="reflect")
    k = kernel.reshape(1, 1, -1)
    y = F.conv1d(x, k)

    y = y.reshape(t_last.shape)
    # Invert permutation to restore original axis order
    inv_order = [0] * ndim
    for i, j in enumerate(order):
        inv_order[j] = i
    return y.permute(inv_order)


def gaussian_convolve_reflect(
    x: torch.Tensor,
    sigma: Union[float, Mapping[int, float], Sequence[float]],
    axes: Union[str, int, Iterable[int]] = "all",
    weights: Optional[Union[float, torch.Tensor]] = 1.0,
    truncate: float = 3.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Separable Gaussian convolution with reflecting boundaries on a 3D tensor (L×N×M).
    Also supports an *optional normalized weighted* convolution.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (L, N, M). (Works for any ndim >= 1; choose axes accordingly.)
    sigma : float | Mapping[int,float] | Sequence[float]
        Standard deviation(s) of the Gaussian. Options:
        - float: same sigma used for all selected axes
        - mapping {axis: sigma}: per-axis sigma
        - sequence: if length == len(selected_axes), aligned in order; if length == x.ndim, aligned by axis index
    axes : "all" | int | Iterable[int], default "all"
        Which axes to convolve over.
        - "all": convolve along every axis of `x`
        - int: convolve along just that axis
        - iterable: convolve along those axes (e.g., [0, 2])
    weights : float | torch.Tensor, optional (default 1.0)
        - float or 0-dim tensor: a single global weight (standard convolution when = 1.0)
        - tensor same shape as `x`: per-element weights. Uses normalized weighted convolution:
              out = G * (x * w) / (G * w)   (G denotes separable Gaussian smoothing),
          with `eps` to avoid division by zero.
    truncate : float, default 3.0
        Kernel radius is ceil(truncate * sigma).
    eps : float, default 1e-8
        Numerical stability for the weighted normalization.

    Returns
    -------
    torch.Tensor
        Smoothed tensor (same shape as `x`, dtype is floating).
    """
    if x.ndim < 1:
        raise ValueError("Input must have at least 1 dimension.")

    # Work dtype/device
    work_dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
    device = x.device
    x_f = x.to(dtype=work_dtype, device=device)

    # Resolve axes
    if axes == "all":
        axes_list = list(range(x.ndim))
    elif isinstance(axes, int):
        axes_list = [axes]
    else:
        axes_list = list(axes)
        if len(axes_list) == 0:
            return x_f  # nothing to do

    # Resolve per-axis sigma
    def sigma_for_axis(ax: int) -> float:
        if isinstance(sigma, Mapping):
            return float(sigma.get(ax, sigma.get(ax % x.ndim, 0.0)))
        if isinstance(sigma, Sequence) and not isinstance(sigma, (str, bytes)):
            if len(sigma) == len(axes_list):
                return float(sigma[axes_list.index(ax)])
            if len(sigma) == x.ndim:
                return float(sigma[ax])
        return float(sigma)  # scalar fallback

    # Prepare weights
    if isinstance(weights, torch.Tensor):
        if weights.shape != x.shape:
            raise ValueError(f"'weights' tensor must match input shape {x.shape}, got {weights.shape}.")
        w = weights.to(dtype=work_dtype, device=device)
    else:
        # scalar or None defaults to 1.0 everywhere
        w = torch.as_tensor(1.0 if weights is None else float(weights), dtype=work_dtype, device=device)
        if w.ndim == 0:
            w = torch.full_like(x_f, w)

    # Numerator & denominator for normalized weighted convolution
    num = x_f * w
    den = w

    # Apply separable 1D Gaussian along each requested axis
    for ax in axes_list:
        s = sigma_for_axis(ax)
        k_full = _gaussian_kernel1d(s, dtype=work_dtype, device=device, truncate=truncate)
        num = _conv1d_along_axis(num, k_full, axis=ax)
        den = _conv1d_along_axis(den, k_full, axis=ax)

    out = num / (den + eps)
    return out

@dataclass
class PredictArgs:
    """Arguments for model prediction."""

    recycling_steps: int = 0  # default in Boltz1
    sampling_steps: int = 200
    diffusion_samples: int = (
        1  # number of samples you want to generate, will be used as multiplicity
    )
    write_confidence_summary: bool = True
    write_full_pae: bool = False
    write_full_pde: bool = False


class DiffusionStepper:
    """Controls fine-grained diffusion steps using the pretrained Boltz1 model.

    This class provides granular control over the diffusion process by:
    1. Loading and caching model representations after the pairformer stage
    2. Enabling step-by-step diffusion with custom parameters
    3. Maintaining the original model weights and architecture
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        data_path: Union[str, Path],
        out_dir: Union[str, Path],
        model: Optional[Boltz1] = None,
        use_msa_server: bool = True,
        predict_args: PredictArgs = PredictArgs(),
        diffusion_args: BoltzDiffusionParams = BoltzDiffusionParams(),
        device: Optional[torch.device] = None,
        pair_rep_func = None,
        single_rep_func = None,
    ) -> None:
        """Load Boltz-1 pretrained model weights and components from checkpoint.

        Parameters
        ----------
        checkpoint_path : Union[str, Path]
            Path to the model checkpoint file.
        data_path : Union[str, Path]
            Path to the input data (folder of YAML files, FASTA files, or a FASTA or YAML file).
        out_dir : Union[str, Path]
            Path to the output directory.
        model : Optional[Boltz1], optional
            Preloaded model, by default None.
        use_msa_server : bool, optional
            Whether to use the MSA server, by default True.
        predict_args : PredictArgs, optional
            Arguments for model prediction, by default PredictArgs().
        diffusion_args : BoltzDiffusionParams, optional
            Diffusion parameters, by default BoltzDiffusionParams(). step_scale is most useful,
            set to a lower value (default 1.638) to get more diversity.
        device : Optional[torch.device], optional
            Device to load the model to, by default None.

        Returns
        -------
        None
        """
        self.device = device# or try_gpu()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.cache_path = Path(
            checkpoint_path
        ).parent  # NOTE: assumes checkpoint and ccd dictionary get downloaded to the same place

        if model is not None:
            self.model = model.to(self.device).eval()
        else:
            self.model = (
                Boltz1.load_from_checkpoint(
                    checkpoint_path,
                    strict=True,
                    predict_args=asdict(predict_args),
                    map_location="cpu",
                    diffusion_process_args=asdict(diffusion_args),
                    ema=False,
                )
                .to(self.device)
                .eval()
            )

        self.setup(data_path=data_path, out_dir=out_dir, use_msa_server=use_msa_server)
        self.pair_rep_func = pair_rep_func
        self.single_rep_func = single_rep_func
        self.cached_representations: Dict[str, torch.Tensor] = {}
        self.cached_diffusion_init = {}
        self.diffusion_trajectory: Dict[str, torch.Tensor] = {}
        self.current_step: int = 0

    def setup(
        self,
        data_path: Union[str, Path],
        out_dir: Union[str, Path],
        use_msa_server: bool = True,
    ) -> BoltzInferenceDataModule:
        """Get BoltzInferenceDataModule set up so the stepper can run on a batch.

        Parameters
        ----------
        data_path : Union[str, Path]
            Path to the input data (folder of YAML files, FASTA files, or a FASTA or YAML file).

        Returns
        -------
        BoltzInferenceDataModule
            Data module containing processed inputs.
        """
        input_path = Path(data_path) if isinstance(data_path, str) else data_path
        input_path = input_path.expanduser().resolve()
        ccd_path = self.cache_path / "ccd.pkl"
        data = check_inputs(input_path) # changed to just input_path (not output)
        # boltz main.py checkputs just takes 1 arg 

        process_inputs(
            data=data,
            out_dir=out_dir,
            ccd_path=ccd_path,
            mol_dir=self.cache_path / "mols",
            use_msa_server=use_msa_server,
            msa_server_url="https://api.colabfold.com",  # NOTE: this requires internet access on cluster
            msa_pairing_strategy="greedy",
        )

        # Load processed data
        processed_dir = out_dir / "processed"
        processed = BoltzProcessedInput(
            manifest=Manifest.load(processed_dir / "manifest.json"),
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
        )

        # Create data module 
        # TODO: set this up so batched will work with later functions? This will require getting density maps into the schema I think
        data_module = BoltzInferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            num_workers=2,  # NOTE: default in Boltz1
        )

        self.data_module = data_module

    def prepare_feats_from_datamodule_batch(
        self,
    ) -> Dict[str, torch.Tensor]:
        """Prepare features from a DataModule batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch from BoltzInferenceDataModule.

        Returns
        -------
        Dict[str, torch.Tensor]
            Processed features ready for the model.
        """
        return self.data_module.transfer_batch_to_device(
            next(iter(self.data_module.predict_dataloader())), self.device, 0
        )  # FIXME: I generally assume batch size of 1, which will break in the future.

    def compute_representations(
        self,
        feats: Dict[str, torch.Tensor],
        recycling_steps: Optional[int] = None,
    ) -> None:
        """Compute and cache main trunk representations.

        Parameters
        ----------
        feats : Dict[str, torch.Tensor]
            Input feats containing model features
        recycling_steps : Optional[int], optional
            Override default number of recycling steps, by default None
        """
        recycling_steps = recycling_steps or self.model.predict_args["recycling_steps"]

        with torch.no_grad():
            # Compute input embeddings
            s_inputs = self.model.input_embedder(feats)

            # Initialize sequence and pairwise embeddings
            s_init = self.model.s_init(s_inputs)
            z_init = (
                self.model.z_init_1(s_inputs)[:, :, None]
                + self.model.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = self.model.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + self.model.token_bonds(feats["token_bonds"].float())

            # Initialize tensors for recycling
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)

            # Compute pairwise mask
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            # Recycling iterations
            for i in range(recycling_steps + 1):
                s = s_init + self.model.s_recycle(self.model.s_norm(s))
                z = z_init + self.model.z_recycle(self.model.z_norm(z))

                if not self.model.no_msa:
                    z = z + self.model.msa_module(z, s_inputs, feats)

                s, z = self.model.pairformer_module(
                    s, z, mask=mask, pair_mask=pair_mask
                )

            if self.pair_rep_func is not None:
                z = self.pair_rep_func(z)
            
            if self.single_rep_func is not None:
                z = self.pair_rep_func(s)

            # Cache outputs
            self.cached_representations = {
                "s": s,
                "z": z,
                "s_inputs": s_inputs,
                "relative_position_encoding": relative_position_encoding,
                "feats": feats,
            }

    def save_representations(
        self,
        save_dir: Union[str, Path],
        name: str,
    ) -> None:
        """Save cached single and pair representations as .pt files.

        Parameters
        ----------
        save_dir : Union[str, Path]
            Directory to save representations to.
        name : str
            Systematic name prefix for saved files (e.g. '1smg_0recycles').
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if not self.cached_representations:
            raise RuntimeError("No cached representations. Call compute_representations first.")
        torch.save(self.cached_representations["s"].cpu(), save_dir / f"{name}_s.pt")
        torch.save(self.cached_representations["z"].cpu(), save_dir / f"{name}_z.pt")

    def initialize_partial_diffusion(
        self,
        structure: Union[Structure, torch.Tensor],
        noising_steps: int = 0,
        num_samples: Optional[int] = None,
        sampling_steps: Optional[int] = None,
        selector: NDArray[np.bool_] = None,
    ) -> None:
        """
        Initialize with a partial diffusion setup, starting from some initial set of coordinates. This allows denoising from
        a partially noised input, which is useful for perturbing from some base set of coordinates for an ensemble.

        Parameters
        ----------
        structure : Union[Structure, torch.Tensor]
            Initial structure or set of atomic coordinates. If not a tensor, it is assumed to
            have an attribute (e.g. `atom_coords`) that contains the coordinates.
        noising_steps : int, optional
            Number of noising steps.
        num_samples : Optional[int], optional
            Number of samples to generate (used to determine diffusion multiplicity),
            by default the value from predict_args.
        sampling_steps : Optional[int], optional
            Total number of sampling steps in the diffusion process,
            by default the value from the model's structure_module.
        selector : NDArray[np.bool_], optional
            Selector mask for atoms to be noised, by default None (all atoms are noised).
        """
        self.diffusion_trajectory = {}

        batch = self.prepare_feats_from_datamodule_batch()
        self.compute_representations(batch)

        num_sampling_steps = default(
            sampling_steps, self.model.structure_module.num_sampling_steps
        )
        diffusion_samples = default(
            num_samples, self.model.predict_args["diffusion_samples"]
        )

        if noising_steps < 0 or num_sampling_steps - noising_steps <= 0:
            raise ValueError(
                f"Invalid number of noising steps: ({noising_steps}) or sampling steps: ({num_sampling_steps})."
            )
        self.current_step = num_sampling_steps - noising_steps

        atom_mask = self.cached_representations["feats"]["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(diffusion_samples, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.model.structure_module.sample_schedule(num_sampling_steps)
        gammas = torch.where(
            sigmas > self.model.structure_module.gamma_min,
            self.model.structure_module.gamma_0,
            0.0,
        )
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # atom position is noise at the beginning
        atom_coords = (
            torch.tensor(structure.atoms["coords"], device=self.device)
            .unsqueeze(0)
            .repeat(diffusion_samples, 1, 1)
        )
        atom_coords = pad_dim(atom_coords, 1, shape[1] - atom_coords.shape[1])
        init_coords = atom_coords.clone()
        eps = (
            self.model.structure_module.noise_scale
            * sigmas[-noising_steps - 1]
            * torch.randn(shape, device=self.device)
        )

        if selector is not None:
            selector = torch.from_numpy(selector).to(self.device)
            selector = pad_dim(selector, 0, shape[1] - selector.shape[0])
            atom_coords[:, selector, :] += eps[:, selector, :]
        else:
            atom_coords += eps

        token_repr = None
        token_a = None

        self.cached_diffusion_init = {
            "init_coords": init_coords,
            "atom_coords": atom_coords,
            "atom_mask": atom_mask,
            "token_repr": token_repr,
            "token_a": token_a,
            "sigmas_and_gammas": sigmas_and_gammas,
            "diffusion_samples": diffusion_samples,
            "num_sampling_steps": num_sampling_steps,
        }

    def initialize_diffusion(
        self,
        num_samples: Optional[int] = None,
        sampling_steps: Optional[int] = None,
        current_step: Optional[int] = 0,
    ) -> None:
        """Initialize the diffusion process.

        Parameters
        ----------
        num_samples : Optional[int], optional
            Number of samples to generate, by default the number from predict_args in initialization
        sampling_steps : Optional[int], optional
            Number of sampling steps, by default the number from predict_args in initialization
        """

        self.current_step = current_step
        self.diffusion_trajectory = {}

        batch = self.prepare_feats_from_datamodule_batch()
        self.compute_representations(batch)

        num_sampling_steps = default(
            sampling_steps, self.model.structure_module.num_sampling_steps
        )
        diffusion_samples = default(
            num_samples, self.model.predict_args["diffusion_samples"]
        )
        atom_mask = self.cached_representations["feats"]["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(diffusion_samples, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.model.structure_module.sample_schedule(num_sampling_steps)
        gammas = torch.where(
            sigmas > self.model.structure_module.gamma_min,
            self.model.structure_module.gamma_0,
            0.0,
        )
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)

        token_repr = None
        token_a = None

        self.cached_diffusion_init = {
            "init_coords": None,
            "atom_coords": atom_coords,
            "atom_mask": atom_mask,
            "token_repr": token_repr,
            "token_a": token_a,
            "sigmas_and_gammas": sigmas_and_gammas,
            "diffusion_samples": diffusion_samples,
            "num_sampling_steps": num_sampling_steps
        }

    def step(
        self,
        atom_coords: torch.Tensor,
        override_step: Optional[int] = None,
        return_denoised: bool = False,
        augmentation: bool = True,
        align_to_input: bool = True,
        inject_noise: bool = True,
        gamma_override: Union[int, None] = None,
        reverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Execute a single diffusion denoising step.

        Parameters
        ----------
        atom_coords : torch.Tensor
            Current atomic coordinates of shape (batch, num_atoms, 3)
        return_denoised : bool, optional
            Whether to return the fully denoised coordinate prediction, by default False
        augmentation : bool, optional
            Whether to apply augmentation, by default True


        Parameters added by Dru
        -----------------------
        override_step : int, optional
            Use this step to select sigmas, gammas instead of current_step
        inject_noise : bool, optional
            Whether to add noise before forward pass

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Denoised atomic coordinates after a single step in the trajectory, and optionally the fully denoised coordinate prediction.
        """

        sampling_step = default(
            override_step, self.current_step
        )

        if reverse:
            reverse = -1
        else:
            reverse = 1

        # Get cached representations
        s = self.cached_representations["s"]
        z = self.cached_representations["z"]
        s_inputs = self.cached_representations["s_inputs"]
        relative_position_encoding = self.cached_representations[
            "relative_position_encoding"
        ]
        feats = self.cached_representations["feats"]
        multiplicity = self.cached_diffusion_init[
            "diffusion_samples"
        ]  # batch is regulated by dataloader, this lets you do ensemble prediction

        # Get cached diffusion info
        atom_mask: torch.Tensor = self.cached_diffusion_init["atom_mask"]
        sigma_tm, sigma_t, gamma = self.cached_diffusion_init["sigmas_and_gammas"][
            sampling_step
        ]
        sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

        if gamma_override is not None:
            gamma = gamma_override

        t_hat = sigma_tm * (1 + gamma)
        eps = (
            self.model.structure_module.noise_scale
            * sqrt(t_hat**2 - sigma_tm**2)
            * torch.randn(atom_coords.shape, device=self.device)
        )

        # NOTE: This might create some interesting pathologies, but in principle this augmentation should not be needed post-training
        if augmentation:
            atom_coords = center_random_augmentation(
                atom_coords,
                atom_mask,
                augmentation=True,
            )

        if inject_noise:
            atom_coords_noisy = atom_coords + eps
        else:
            atom_coords_noisy = atom_coords

        with torch.no_grad():
            atom_coords_denoised, _ = (
                self.model.structure_module.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        s_trunk=s,
                        z_trunk=z,
                        s_inputs=s_inputs,
                        feats=feats,
                        relative_position_encoding=relative_position_encoding,
                        multiplicity=multiplicity,
                    ),
                )
            )

        atom_coords_noisy = weighted_rigid_align(
            atom_coords_noisy.float(),
            atom_coords_denoised.float(),
            atom_mask.float(),
            atom_mask.float(),
        )

        atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)
        denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
        atom_coords_next: torch.Tensor = (
            atom_coords_noisy
            + self.model.structure_module.step_scale
            * (sigma_t - t_hat) * reverse
            * denoised_over_sigma
        )

        # Align to input
        if align_to_input:
            if self.cached_diffusion_init["init_coords"] is None:
                raise ValueError(
                    "No initial input coordinates found in cached diffusion init. Please change from align_to_input if you are not using partial diffusion."
                )
            atom_coords_next = weighted_rigid_align(
                atom_coords_next.float(),
                self.cached_diffusion_init["init_coords"].float(),
                atom_mask.float(),
                atom_mask.float(),
            ).to(atom_coords_next)

        pad_mask = feats["atom_pad_mask"].squeeze().bool()
        unpad_coords_next = atom_coords_next[
            :, pad_mask, :
        ]  # unpad the coords to B, N_unpad, 3
        unpad_coords_denoised = atom_coords_denoised[
            :, pad_mask, :
        ]  # unpad the coords to B, N_unpad, 3

        # Store unpadded in trajectory (0 indexed)
        self.diffusion_trajectory[f"step_{self.current_step}"] = {
            "coords": unpad_coords_next.clone(),
            "denoised": unpad_coords_denoised.clone(),  # the overall prediction from this current level (no noise mixture)
        }

        self.current_step += 1  # NOTE: current step to execute

        if return_denoised:
            return atom_coords_next, atom_coords_denoised
        else:
            return atom_coords_next


    def _denoise(
        self,
        atom_coords: torch.Tensor,
        sigma: float,
        augmentation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared denoising call: returns (aligned_noisy, denoised).

        Parameters
        ----------
        atom_coords : torch.Tensor
            Noisy coordinates (batch, num_atoms, 3).
        sigma : float
            Noise level for this evaluation.
        augmentation : bool
            Whether to apply random augmentation before the network call.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (atom_coords_aligned, atom_coords_denoised)
        """
        s = self.cached_representations["s"]
        z = self.cached_representations["z"]
        s_inputs = self.cached_representations["s_inputs"]
        relative_position_encoding = self.cached_representations["relative_position_encoding"]
        feats = self.cached_representations["feats"]
        multiplicity = self.cached_diffusion_init["diffusion_samples"]
        atom_mask = self.cached_diffusion_init["atom_mask"]

        if augmentation:
            atom_coords = center_random_augmentation(atom_coords, atom_mask, augmentation=True)

        with torch.no_grad():
            atom_coords_denoised, _ = self.model.structure_module.preconditioned_network_forward(
                atom_coords,
                sigma,
                training=False,
                network_condition_kwargs=dict(
                    s_trunk=s,
                    z_trunk=z,
                    s_inputs=s_inputs,
                    feats=feats,
                    relative_position_encoding=relative_position_encoding,
                    multiplicity=multiplicity,
                ),
            )

        atom_coords = weighted_rigid_align(
            atom_coords.float(),
            atom_coords_denoised.float(),
            atom_mask.float(),
            atom_mask.float(),
        ).to(atom_coords_denoised)

        return atom_coords, atom_coords_denoised

    def _store_trajectory(self, atom_coords_next: torch.Tensor, atom_coords_denoised: torch.Tensor) -> None:
        """Store unpadded coords in the trajectory dict and advance step counter."""
        feats = self.cached_representations["feats"]
        pad_mask = feats["atom_pad_mask"].squeeze().bool()
        self.diffusion_trajectory[f"step_{self.current_step}"] = {
            "coords": atom_coords_next[:, pad_mask, :].clone(),
            "denoised": atom_coords_denoised[:, pad_mask, :].clone(),
        }
        self.current_step += 1

    def step_langevin(
        self,
        atom_coords: torch.Tensor,
        n_langevin_steps: int = 5,
        langevin_step_size: float = 0.01,
        override_step: Optional[int] = None,
        return_denoised: bool = False,
        augmentation: bool = True,
        align_to_input: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Annealed Langevin dynamics step.

        At each noise level sigma_t, run multiple Langevin correction steps
        using the score estimate from the denoiser before transitioning to the
        next noise level via a standard Euler step.

        The score at noise level sigma is:
            score = (x_denoised - x_noisy) / sigma^2

        Each Langevin step:
            x <- x + (eps/2) * score + sqrt(eps) * z,  z ~ N(0, I)

        where eps = langevin_step_size * (sigma_t / sigma_max)^2 for annealing.

        Parameters
        ----------
        atom_coords : torch.Tensor
            Current coordinates (batch, num_atoms, 3).
        n_langevin_steps : int
            Number of Langevin correction steps per noise level.
        langevin_step_size : float
            Base step size for Langevin updates (annealed by sigma).
        override_step : Optional[int]
            Override the current step index.
        return_denoised : bool
            Whether to also return the denoised prediction.
        augmentation : bool
            Random rotation augmentation.
        align_to_input : bool
            Align output to initial coordinates (partial diffusion only).
        """
        sampling_step = default(override_step, self.current_step)
        atom_mask = self.cached_diffusion_init["atom_mask"]

        sigma_tm, sigma_t, gamma = self.cached_diffusion_init["sigmas_and_gammas"][sampling_step]
        sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()
        t_hat = sigma_tm * (1 + gamma)

        sigma_max = self.model.structure_module.sigma_max * self.model.structure_module.sigma_data

        # Inject noise to get to t_hat level
        eps_noise = self.model.structure_module.noise_scale * sqrt(t_hat**2 - sigma_tm**2) * torch.randn_like(atom_coords)
        atom_coords = atom_coords + eps_noise

        # Annealed Langevin correction steps at this noise level
        annealed_eps = langevin_step_size * (t_hat / sigma_max) ** 2
        for _ in range(n_langevin_steps):
            x_aligned, x_denoised = self._denoise(atom_coords, t_hat, augmentation=augmentation)
            # Score: ∇_x log p(x|sigma) ≈ (x_denoised - x) / sigma^2
            score = (x_denoised - x_aligned) / (t_hat ** 2)
            noise = torch.randn_like(atom_coords)
            atom_coords = x_aligned + (annealed_eps / 2) * score + sqrt(annealed_eps) * noise

        # Final denoising at this level for the Euler transition
        x_aligned, atom_coords_denoised = self._denoise(atom_coords, t_hat, augmentation=augmentation)

        # Euler step from t_hat -> sigma_t
        denoised_over_sigma = (x_aligned - atom_coords_denoised) / t_hat
        atom_coords_next = x_aligned + self.model.structure_module.step_scale * (sigma_t - t_hat) * denoised_over_sigma

        if align_to_input and self.cached_diffusion_init["init_coords"] is not None:
            atom_coords_next = weighted_rigid_align(
                atom_coords_next.float(),
                self.cached_diffusion_init["init_coords"].float(),
                atom_mask.float(), atom_mask.float(),
            ).to(atom_coords_next)

        self._store_trajectory(atom_coords_next, atom_coords_denoised)

        if return_denoised:
            return atom_coords_next, atom_coords_denoised
        return atom_coords_next

    def step_sde_euler_maruyama(
        self,
        atom_coords: torch.Tensor,
        override_step: Optional[int] = None,
        return_denoised: bool = False,
        augmentation: bool = True,
        align_to_input: bool = True,
        s_churn: float = 0.0,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Euler-Maruyama SDE sampler (stochastic counterpart of the ODE sampler).

        Treats the reverse process as an SDE:
            dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dW

        For the EDM formulation this corresponds to injecting fresh noise
        proportional to the step size at each transition, maintaining the
        correct marginal distribution.

        Parameters
        ----------
        atom_coords : torch.Tensor
            Current coordinates (batch, num_atoms, 3).
        s_churn : float
            Controls stochasticity. 0 = deterministic (ODE). Higher = more noise.
            Typical range: 0-80 (Karras et al. recommend s_churn ~ sqrt(2)-1 for
            diversity without quality loss).
        """
        sampling_step = default(override_step, self.current_step)
        atom_mask = self.cached_diffusion_init["atom_mask"]
        num_steps = self.cached_diffusion_init["num_sampling_steps"]

        sigma_tm, sigma_t, gamma = self.cached_diffusion_init["sigmas_and_gammas"][sampling_step]
        sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

        # Churn: increase effective noise level
        gamma_churn = min(s_churn / num_steps, sqrt(2.0) - 1)
        t_hat = sigma_tm * (1 + gamma_churn)

        # Add noise to reach t_hat
        eps_noise = sqrt(t_hat**2 - sigma_tm**2) * torch.randn_like(atom_coords)
        atom_coords_noisy = atom_coords + eps_noise

        x_aligned, atom_coords_denoised = self._denoise(atom_coords_noisy, t_hat, augmentation=augmentation)

        # Deterministic component (score-based drift)
        denoised_over_sigma = (x_aligned - atom_coords_denoised) / t_hat
        dt = sigma_t - t_hat

        # Stochastic component: inject noise scaled by step size
        if sigma_t > 0:
            noise = torch.randn_like(atom_coords)
            # SDE noise injection: g(t) * sqrt(|dt|) * z
            # For variance-exploding SDE: g(t) = sigma * sqrt(2 * log(sigma_max/sigma_min))
            noise_strength = sqrt(abs(dt)) * sigma_t
            atom_coords_next = x_aligned + self.model.structure_module.step_scale * dt * denoised_over_sigma + noise_strength * noise
        else:
            atom_coords_next = x_aligned + self.model.structure_module.step_scale * dt * denoised_over_sigma

        if align_to_input and self.cached_diffusion_init["init_coords"] is not None:
            atom_coords_next = weighted_rigid_align(
                atom_coords_next.float(),
                self.cached_diffusion_init["init_coords"].float(),
                atom_mask.float(), atom_mask.float(),
            ).to(atom_coords_next)

        self._store_trajectory(atom_coords_next, atom_coords_denoised)

        if return_denoised:
            return atom_coords_next, atom_coords_denoised
        return atom_coords_next

    def step_predictor_corrector(
        self,
        atom_coords: torch.Tensor,
        n_corrector_steps: int = 2,
        corrector_step_size: float = 0.01,
        override_step: Optional[int] = None,
        return_denoised: bool = False,
        augmentation: bool = True,
        align_to_input: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predictor-Corrector sampler (Song et al., 2021).

        Combines an Euler predictor step (noise level transition) with
        Langevin corrector steps (score-based refinement at fixed noise level).

        1. PREDICT: Euler step from sigma_tm -> sigma_t
        2. CORRECT: n_corrector_steps of Langevin dynamics at sigma_t

        Parameters
        ----------
        atom_coords : torch.Tensor
            Current coordinates (batch, num_atoms, 3).
        n_corrector_steps : int
            Number of Langevin corrector steps after the predictor.
        corrector_step_size : float
            Step size for Langevin corrector (auto-scaled by noise level).
        """
        sampling_step = default(override_step, self.current_step)
        atom_mask = self.cached_diffusion_init["atom_mask"]

        sigma_tm, sigma_t, gamma = self.cached_diffusion_init["sigmas_and_gammas"][sampling_step]
        sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()
        t_hat = sigma_tm * (1 + gamma)

        # Inject noise to reach t_hat
        eps_noise = self.model.structure_module.noise_scale * sqrt(t_hat**2 - sigma_tm**2) * torch.randn_like(atom_coords)
        atom_coords_noisy = atom_coords + eps_noise

        # PREDICTOR: Euler step from t_hat -> sigma_t
        x_aligned, atom_coords_denoised = self._denoise(atom_coords_noisy, t_hat, augmentation=augmentation)
        denoised_over_sigma = (x_aligned - atom_coords_denoised) / t_hat
        atom_coords_predicted = x_aligned + self.model.structure_module.step_scale * (sigma_t - t_hat) * denoised_over_sigma

        # CORRECTOR: Langevin steps at sigma_t
        if sigma_t > 0 and n_corrector_steps > 0:
            # Scale step size by noise level
            eps_corrector = corrector_step_size * (sigma_t ** 2)
            x_current = atom_coords_predicted

            for _ in range(n_corrector_steps):
                x_aligned_c, x_denoised_c = self._denoise(x_current, sigma_t, augmentation=augmentation)
                score = (x_denoised_c - x_aligned_c) / (sigma_t ** 2)
                noise = torch.randn_like(x_current)
                x_current = x_aligned_c + (eps_corrector / 2) * score + sqrt(eps_corrector) * noise

            atom_coords_next = x_current
            # Get final denoised for trajectory
            _, atom_coords_denoised = self._denoise(atom_coords_next, sigma_t, augmentation=False)
        else:
            atom_coords_next = atom_coords_predicted

        if align_to_input and self.cached_diffusion_init["init_coords"] is not None:
            atom_coords_next = weighted_rigid_align(
                atom_coords_next.float(),
                self.cached_diffusion_init["init_coords"].float(),
                atom_mask.float(), atom_mask.float(),
            ).to(atom_coords_next)

        self._store_trajectory(atom_coords_next, atom_coords_denoised)

        if return_denoised:
            return atom_coords_next, atom_coords_denoised
        return atom_coords_next

    def run_sampling(
        self,
        sampler: str = "default",
        save_structs_every_n: int = 0,
        save_current: bool = True,
        save_perceived: bool = True,
        prefix: str = "sample",
        out_dir: Optional[Union[str, Path]] = None,
        structure: Optional[Structure] = None,
        **sampler_kwargs,
    ) -> torch.Tensor:
        """Run full diffusion sampling loop with a chosen sampler.

        Parameters
        ----------
        sampler : str
            One of 'default', 'langevin', 'sde', 'predictor_corrector'.
        save_structs_every_n : int
            Save structures every N steps. 0 to disable.
        prefix : str
            Filename prefix for saved structures.
        out_dir : Optional[Union[str, Path]]
            Directory for saving structures. Required if save_structs_every_n > 0.
        structure : Optional[Structure]
            Structure object for mmCIF export. Required if save_structs_every_n > 0.
        **sampler_kwargs
            Extra keyword arguments passed to the sampler step function.

        Returns
        -------
        torch.Tensor
            Final denoised coordinates.
        """
        step_fn_map = {
            "default": self.step,
            "langevin": self.step_langevin,
            "sde": self.step_sde_euler_maruyama,
            "predictor_corrector": self.step_predictor_corrector,
        }
        if sampler not in step_fn_map:
            raise ValueError(f"Unknown sampler '{sampler}'. Choose from: {list(step_fn_map.keys())}")

        step_fn = step_fn_map[sampler]
        num_steps = self.cached_diffusion_init["num_sampling_steps"]
        coords = self.cached_diffusion_init["atom_coords"]
        pad_mask = self.cached_representations["feats"]["atom_pad_mask"].squeeze().bool()

        # Filter out kwargs not accepted by default step
        default_extra = {}
        if sampler == "default":
            default_extra = {"inject_noise": sampler_kwargs.pop("inject_noise", True)}
            sampler_kwargs = {k: v for k, v in sampler_kwargs.items()
                             if k in ("gamma_override", "reverse")}
            sampler_kwargs.update(default_extra)

        for i in range(num_steps):
            coords, perceived = step_fn(
                atom_coords=coords,
                align_to_input=False,
                return_denoised=True,
                **sampler_kwargs,
            )

            if save_structs_every_n and out_dir and structure:
                if (i % save_structs_every_n == 0) or (i == num_steps - 1):
                    from export import write_mmcif
                    out_path = Path(out_dir)
                    if save_current:
                        write_mmcif(
                            self.diffusion_trajectory[f"step_{i}"]["coords"],
                            structure, out_path / f"{prefix}_{i}_current.cif",
                        )
                    if save_perceived:
                        write_mmcif(
                            perceived[:, pad_mask, ...],
                            structure, out_path / f"{prefix}_{i}_perceived.cif",
                        )

        return coords


class DensityGuidedDiffusionStepper(DiffusionStepper):
    """Controls fine-grained diffusion steps using the pretrained Boltz1 model and guidance via the diffusion update"""

    def step(
        self,
        atom_coords: torch.Tensor,
        density_grad: torch.Tensor,
        guidance_scale: float = 0.1,
        return_denoised: bool = False,
        augmentation: bool = True,
        align_to_input: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Execute a single diffusion denoising step.

        Parameters
        ----------
        atom_coords : torch.Tensor
            Current atomic coordinates of shape (batch, num_atoms, 3)
        return_denoised : bool, optional
            Whether to return the fully denoised coordinate prediction, by default False
        augmentation : bool, optional
            Whether to apply augmentation, by default True

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Denoised atomic coordinates after a single step in the trajectory, and optionally the fully denoised coordinate prediction.
        """
        # Get cached representations
        s = self.cached_representations["s"]
        z = self.cached_representations["z"]
        s_inputs = self.cached_representations["s_inputs"]
        relative_position_encoding = self.cached_representations[
            "relative_position_encoding"
        ]
        feats = self.cached_representations["feats"]
        multiplicity = self.cached_diffusion_init[
            "diffusion_samples"
        ]  # batch is regulated by dataloader, this lets you do ensemble prediction

        # Get cached diffusion info
        atom_mask: torch.Tensor = self.cached_diffusion_init["atom_mask"]
        sigma_tm, sigma_t, gamma = self.cached_diffusion_init["sigmas_and_gammas"][
            self.current_step
        ]
        sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

        t_hat = sigma_tm * (1 + gamma)
        eps = (
            self.model.structure_module.noise_scale
            * sqrt(t_hat**2 - sigma_tm**2)
            * torch.randn(atom_coords.shape, device=self.device)
        )

        # NOTE: This might create some interesting pathologies, but in principle this augmentation should not be needed post-training
        if augmentation:
            atom_coords = center_random_augmentation(
                atom_coords,
                atom_mask,
                augmentation=True,
            )

        atom_coords_noisy = atom_coords + eps

        with torch.no_grad():
            atom_coords_denoised, _ = (
                self.model.structure_module.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        s_trunk=s,
                        z_trunk=z,
                        s_inputs=s_inputs,
                        feats=feats,
                        relative_position_encoding=relative_position_encoding,
                        multiplicity=multiplicity,
                    ),
                )
            )

        atom_coords_noisy = weighted_rigid_align(
            atom_coords_noisy.float(),
            atom_coords_denoised.float(),
            atom_mask.float(),
            atom_mask.float(),
        )

        atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

        denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat

        scaled_guidance_grad = torch.linalg.norm(denoised_over_sigma) / torch.linalg.norm(density_grad) * density_grad

        denoised_over_sigma = denoised_over_sigma + scaled_guidance_grad * guidance_scale

        atom_coords_next: torch.Tensor = (
            atom_coords_noisy
            + self.model.structure_module.step_scale
            * (sigma_t - t_hat)
            * denoised_over_sigma
        )

        # Align to input
        if align_to_input:
            if self.cached_diffusion_init["init_coords"] is None:
                raise ValueError(
                    "No initial input coordinates found in cached diffusion init. Please change from align_to_input if you are not using partial diffusion."
                )
            atom_coords_next = weighted_rigid_align(
                atom_coords_next.float(),
                self.cached_diffusion_init["init_coords"].float(),
                atom_mask.float(),
                atom_mask.float(),
            ).to(atom_coords_next)

        pad_mask = feats["atom_pad_mask"].squeeze().bool()
        unpad_coords_next = atom_coords_next[
            :, pad_mask, :
        ]  # unpad the coords to B, N_unpad, 3
        unpad_coords_denoised = atom_coords_denoised[
            :, pad_mask, :
        ]  # unpad the coords to B, N_unpad, 3

        # Store unpadded in trajectory (0 indexed)
        self.diffusion_trajectory[f"step_{self.current_step}"] = {
            "coords": unpad_coords_next.clone(),
            "denoised": unpad_coords_denoised.clone(),  # the overall prediction from this current level (no noise mixture)
        }

        self.current_step += 1  # NOTE: current step to execute

        if return_denoised:
            return atom_coords_next, atom_coords_denoised
        else:
            return atom_coords_next

import torch

class LinearInterpolation:
    def __init__(self, t, x, s=None):
        """
        t: (T,) tensor of time points (must be sorted in ascending order)
        x: (..., T, C) tensor of values corresponding to t, where:
           - "..." are batch dimensions
           - T is the number of time points
           - C is the number of channels
        s: Optional smoothing factor. If provided, the values are smoothed using a moving average.
        """
        self.t = t
        if s is not None:
            window = int(s)
            if window < 1:
                raise ValueError("Smoothing parameter s must be >= 1")
            pad = window // 2
            x_padded = torch.nn.functional.pad(x, (0, 0, pad, pad), mode='replicate')
            x_smoothed = torch.nn.functional.avg_pool1d(x_padded.transpose(-1, -2), kernel_size=window, stride=1).transpose(-1, -2)
            self.x = x_smoothed
        else:
            self.x = x

    def derivative(self, points):
        """Compute the derivative at given points using linear interpolation."""
        batch_shape = self.x.shape[:-2]
        num_channels = self.x.shape[-1]
        points = points.unsqueeze(-1)  # Ensure shape compatibility
        
        # Find indices of the right interval
        indices = torch.searchsorted(self.t, points, right=True)
        indices = torch.clamp(indices, 1, len(self.t) - 1)
        
        # Get left and right points
        t_left = self.t[indices - 1]
        t_right = self.t[indices]
        x_left = self.x[..., indices - 1, :]
        x_right = self.x[..., indices, :]
        
        # Compute slope (piecewise constant derivative)
        slopes = (x_right - x_left) / (t_right - t_left).unsqueeze(-1)
        return slopes

    def evaluate(self, points):
        """Compute the interpolated value at given points using linear interpolation."""
        points = points.unsqueeze(-1)  # Ensure shape compatibility
        
        # Find indices of the right interval
        indices = torch.searchsorted(self.t, points, right=True)
        indices = torch.clamp(indices, 1, len(self.t) - 1)
        
        # Get left and right points
        t_left = self.t[indices - 1]
        t_right = self.t[indices]
        x_left = self.x[..., indices - 1, :]
        x_right = self.x[..., indices, :]
        
        # Linear interpolation formula
        weights = (points - t_left) / (t_right - t_left)
        interpolated_values = x_left + weights.unsqueeze(-1) * (x_right - x_left)
        
        return interpolated_values