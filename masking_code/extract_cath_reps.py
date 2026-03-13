# extract_cath_reps.py 
import torch
from pathlib import Path
import diffusion_stepper as ds
from test_reps import make_yaml_input

cath_sequences = {}  # load from CATH FASTA/domain list
# e.g. {"1abcA00": {"A": "MKTL..."}, "2defB01": {"B": "ARND..."}, ...}

wts_path = Path("boltz1_conf.ckpt")
save_dir = Path("cath20_reps")
save_dir.mkdir(exist_ok=True)

stepper = None
predict_args = ds.PredictArgs(recycling_steps=0)

for domain_id, chains in cath_sequences.items():
    work_dir = save_dir / f"_work_{domain_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = work_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write YAML with msa: empty (no MSA needed for representation extraction)
    input_dir = make_yaml_input(domain_id, chains, work_dir)

    if stepper is None:
        stepper = ds.DiffusionStepper(
            checkpoint_path=wts_path,
            data_path=input_dir,
            out_dir=out_dir,
            use_msa_server=False,
            predict_args=predict_args,
            device=torch.device("mps"),  # or cuda
        )
    else:
        stepper.setup(data_path=input_dir, out_dir=out_dir, use_msa_server=False)

    batch = stepper.prepare_feats_from_datamodule_batch()
    stepper.compute_representations(batch, recycling_steps=0)
    stepper.save_representations(save_dir, domain_id)

    # Clean up working files
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)