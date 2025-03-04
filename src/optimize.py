import os
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
load_dotenv()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def optimize(cfg: DictConfig) -> float:
    is_sweep_run = True
    sweep_max_epochs = 20
    
    from train import main
    
    os.environ['IS_SWEEP_RUN'] = 'true'
    os.environ['SWEEP_MAX_EPOCHS'] = str(sweep_max_epochs)
    
    try:
        val_loss = main(cfg)
        
        if val_loss is None:
            print("Warning: Training returned None as validation loss. Using a high value.")
            val_loss = float(1e10)
        else:
            val_loss = float(val_loss)
            
        print(f"Trial completed with validation loss: {val_loss}")
        
        return val_loss
        
    except Exception as e:
        print(f"Error during trial: {str(e)}")
        return float(1e10)


if __name__ == "__main__":
    optimize()