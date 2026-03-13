import torch
import os
from pathlib import Path
import pickle

import diffusion_stepper as ds
from export import write_mmcif
from mmcif import parse_mmcif
from boltz.data.types import Structure
from boltz.data.pad import pad_dim
from boltz.model.loss.diffusion import weighted_rigid_align


# filepath options
wts_path = Path('/Users/joycemo/Documents/GitHub/boltz/boltz1_conf.ckpt') # weights?? 
ccd_path = Path("/Users/joycemo/Documents/GitHub/boltz/ccd.pkl") 
data_path = Path('in') # should contain either 1 fasta file or 1 yaml file
out_dir = Path('out')

# initial structure if doing partial diffusion
init_struct_path = None#Path('A70E_1d3cf_unrelaxed_model_1_rel.pdb')

# parameters
use_msa_server = True
num_images = 1 # number of diffusion trajectories
sampling_steps = 200 # default 200
early_stop = 0 # stop sampling after this many steps, set to 0 to disable
step_scale = 1.0 # default 1.638
inject_noise = True # whether to add noise at each step

# output options
prefix = 'z_mask' # will be added to all outputs
save_structs_every_n = 10 # save structures every n iterations, 0 to disable
save_current = True # save current denoising states
save_perceived = True # save perceived denoising targets

####################################################################################################

# functions to apply to pair rep or single rep

def pair_rep_mask_interdomain(z, range_domain_a, range_domain_b):
    # z will have shape [1, N, N, 128], where N is number of residues/tokens
    z[:,range_domain_a[0]:range_domain_a[1], range_domain_b[0]:range_domain_b[1],:] = 0
    z[:,range_domain_b[0]:range_domain_b[1], range_domain_a[0]:range_domain_a[1],:] = 0
    return z

def pair_rep_explicit_mask(z, mask):
    z[mask.bool()] = 0
    return z





####################################################################################################

input_file = None
with os.scandir(data_path) as d:
    for entry in d:
        if entry.name.endswith('.yaml') or entry.name.endswith('.fasta'):
            input_file = entry.name

with open(ccd_path, "rb") as f:
    ccd = pickle.load(f)

# we want our coordinates in tensors of shape (batch, num_atoms, 3)
if init_struct_path is not None:
    init_struct = torch.Tensor(parse_mmcif(init_struct_path, ccd).data.atoms['coords']).unsqueeze(dim=0)

stepper = ds.DiffusionStepper(checkpoint_path=wts_path,
                              data_path=data_path,
                              out_dir=out_dir,
                              use_msa_server=use_msa_server,
                              pair_rep_func=lambda z: pair_rep_mask_interdomain(z, (10,21), (40,91))
                              )

# may want to use partial diffusion at some point, but currently just using initialize_diffusion()
stepper.initialize_diffusion(num_samples=num_images, sampling_steps=sampling_steps)
stepper.model.structure_module.step_scale = step_scale

npz_name = input_file.replace('.yaml','').replace('.fasta','')
structure = Structure.load(out_dir / f"processed/structures/{npz_name}.npz")
#print(stepper.data_module.manifest.records[0].id)

coords = stepper.cached_diffusion_init['atom_coords'] # shape = (batch, num_atoms, 3)
#coords = coords.repeat_interleave(repeats=2, dim=0)

pad = stepper.cached_representations["feats"]["atom_pad_mask"]
pad_mask = stepper.cached_representations["feats"]["atom_pad_mask"].squeeze().bool()
pad_shape = pad_mask.shape
unpad_coords = coords[:, pad_mask, :]  # unpad the coords to B, N_unpad, 3

# pad coords
#coords = pad_dim(init_struct, 1, pad_shape[0] - init_struct.shape[1])

stepper.cached_diffusion_init['atom_coords'] = coords

if early_stop:
    sampling_steps = early_stop

counter = 0
for i in range(sampling_steps):

    coords, perceived = stepper.step(atom_coords=coords, 
                                     align_to_input=False, 
                                     return_denoised=True,
                                     inject_noise=False,
                                    # gamma_override=0,
                                  #   augmentation=False,
                                  )
    
    if ((save_structs_every_n) and (counter % save_structs_every_n == 0)) or (i == sampling_steps-1):
        if save_current:
            write_mmcif(stepper.diffusion_trajectory[f"step_{counter}"]["coords"],
                        structure, Path(out_dir / f"{prefix}_{counter}_current.cif"),)
        if save_perceived:
            write_mmcif(perceived[:, pad_mask, ...],
                        structure, Path(out_dir / f"{prefix}_{counter}_perceived.cif"),)
    counter += 1