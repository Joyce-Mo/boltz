import torch
import numpy as np

from boltz.data.types import Structure
from pathlib import Path
from dataclasses import replace
from boltz.data.write.mmcif import to_mmcif

def write_mmcif(
    coords: torch.Tensor,
    structure: Structure,
    output_path: Path,
    elements: torch.Tensor = None,
    return_structure: bool = False,
) -> None | Structure:
    """
    Write one or more mmCIF files from a batch of coordinates and a structure.

    Parameters
    ----------
    coords : torch.Tensor
        Tensor of shape [batch, n_atoms, 3] or [n_atoms, 3] containing atomic coordinates.
    structure : Structure
        A Structure object containing atom, residue, and chain information.
    output_path : Path
        The base file path for the output mmCIF files. For a batch, a suffix
        '_{batch_index}' will be appended to the base name.
    elements : torch.Tensor, optional
        An optional tensor containing atomix numbers for each atom.
        This can either be of shape [n_atoms] (to be used for all structures) or
        [batch, n_atoms] (to provide distinct elements per structure).
    return_structure : bool, optional
        Whether to return the modified Structure object. Defaults to False.
    """
    base_path = Path(output_path)
    batch_size = coords.shape[0] if coords.ndim == 3 else 1
    elements_batch_size = elements.shape[0] if elements is not None and elements.ndim == 2 else 1

    if elements is not None and batch_size != elements_batch_size:
        raise ValueError(
            f"Batch size or number of atoms in coords and elements must match. coords: {coords.shape} elements: {elements.shape}"
        )

    for i in range(batch_size):
        model_coords = coords[i] if batch_size > 1 else coords
        model_coords_np = (
            model_coords.cpu().numpy() if model_coords.is_cuda else model_coords.numpy()
        )

        atoms = structure.atoms.copy()
        atoms["coords"] = model_coords_np
        atoms["is_present"] = True

        # If element information is provided, update the atom table accordingly
        if elements is not None:
            # If elements are provided for each model separately, use those elements
            if elements.ndim == 2:
                elem_tensor = elements[i]
            else:
                elem_tensor = elements
            model_elements_np = (
                elem_tensor.cpu().numpy()
                if elem_tensor.is_cuda
                else elem_tensor.numpy()
            )
            atoms["element"] = model_elements_np

        new_structure = replace(structure, atoms=atoms)

        mmcif_str = to_mmcif(new_structure)

        # Construct the output file path by appending _{i} before the extension
        new_output_path = base_path.with_name(f"{base_path.stem}_{i}{base_path.suffix}") if batch_size > 1 else base_path

        # Make sure the directory exists
        new_output_path.parent.mkdir(parents=True, exist_ok=True)

        with new_output_path.open("w") as f:
            f.write(mmcif_str)

    if return_structure:
        return new_structure