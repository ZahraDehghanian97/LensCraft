
import numpy as np
from pathlib import Path

def export_et_trajectories(simulations, output_dir):
    output_dir = Path(output_dir) / "et_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sim in enumerate(simulations):
        sample_dir = output_dir / f"sample_{i}"
        sample_dir.mkdir(exist_ok=True)
        
        np.save(
            sample_dir / "original_camera.npy", 
            sim["camera"].cpu().numpy()
        )
        np.save(
            sample_dir / "subject.npy", 
            sim["subject"].cpu().numpy()
        )
        
        np.save(
            sample_dir / "reconstructed_camera.npy", 
            sim["rec"]["reconstructed"][0].detach().cpu().numpy()
        )
        
        if "prompt_gen" in sim:
            np.save(
                sample_dir / "prompt_gen_camera.npy", 
                sim["prompt_gen"]["reconstructed"][0].detach().cpu().numpy()
            )
        
        if "hybrid_gen" in sim:
            np.save(
                sample_dir / "hybrid_gen_camera.npy", 
                sim["hybrid_gen"]["reconstructed"][0].detach().cpu().numpy()
            )
        
        if "caption" in sim:
            with open(sample_dir / "caption.txt", "w") as f:
                f.write(sim["caption"])
        
        if "padding_mask" in sim:
            np.save(
                sample_dir / "padding_mask.npy", 
                sim["padding_mask"].cpu().numpy()
            )
    
    print(f"ET inference results saved to {output_dir}")
    return output_dir

