import uuid
import hydra
import wandb
import pathlib

from src.loaders import get_dataloaders
from src.models import get_model
from src.optimisers import get_optimiser
from src.schedulers import get_scheduler
from src.losses import get_loss
from src.run import run
from src.utils import set_random_seed, get_device, load_config, save_config, to_dict


@hydra.main(config_path="config", config_name="config.yaml")
def train(cfg):
    # set random seed for reproducibility
    set_random_seed(seed=cfg.train.seed, 
                    is_gpu=cfg.train.is_gpu)
    
    # get training backend
    device = get_device(is_gpu=cfg.train.is_gpu, 
                        gpu_number=cfg.train.gpu_number)
    
    # unique id
    experiment_id = cfg.experiment.id if cfg.experiment.id is not None else uuid.uuid4().hex[:8]
 
    # initialise logging
    if cfg.logging.wb_logging: wandb.init(project=cfg.logging.wb_project, id=experiment_id)

    model_name, ensemble = cfg.model.name, cfg.model.ensemble
    models_dir = pathlib.Path("./models") / ((model_name + "_" + ensemble) if ensemble is not None else model_name) / ("run_" + str(cfg.experiment.run)) / experiment_id
    models_dir.mkdir(parents=True)

    # initalise dataloaders
    dataloaders = get_dataloaders(**to_dict(cfg.data),
                                  seed=cfg.train.seed + cfg.train.run,
                                  device=device)
    
    # initialise model
    model = get_model(**to_dict(cfg.model)).to(device)

    # initialise loss
    loss = get_loss(ensemble=ensemble, **to_dict(cfg.loss))

    # initialise optimiser
    optimiser = get_optimiser(model=model, **to_dict(cfg.optimiser))
    
    # initialise scheduler
    scheduler = get_scheduler(optimiser=optimiser, **to_dict(cfg.scheduler))

    # train model
    run(model=model, 
        train_loader=dataloaders["train"],
        valid_loader=dataloaders["valid"],
        criterion=loss,  
        optimiser=optimiser,
        scheduler=scheduler,
        num_epochs=cfg.train.num_epochs,
        save_dir=models_dir,
        device=device,
        wb_logging=cfg.logging.wb_logging)
     
    # save hyperparameters
    save_config(cfg, models_dir)


if __name__ == "__main__":
    train()