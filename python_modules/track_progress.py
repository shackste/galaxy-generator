import os
import wandb

#os.system("wandb login")
wandb.login(key="834835ffb309d5b1618c537d20d23794b271a208")

## general usage
#wandb.init(**kwargs) ## initialize tracking
#wandb.log({"loss":loss}) ## add loss to track record
#wandb.finish()  ## finalize tracking

def track_progress(func, wandb_kwargs: dict =None) -> None:
    """
    track progress of pytorch model using wandb

    Parameters
    ----------
    func : callable : progress to be tracked, usually use model.train
    wandb_kwargs : keyword arguments to be passed to wandb


    Usage
    -----
    >>> model = pytorch.Model()  ## must contain wandb.log calls
    >>> track_progress(model.train, wandb_kwargs=wandb_kwargs)
    """
    assert type(wandb_kwargs) is dict, "wandb_kwargs must be dict with wandb.init kwargs"
    wandb.init(**wandb_kwargs)
    func()
    wandb.finish()


## standard dicts
hyperparameter_dict = {
    "lr_init" : 0.04,
    "lr_gamma" : 0.995
}

kwargs = {
    "project" : "galaxy classifier", ## top level identifier
    "group" : "parameter search", ## secondary identifier
    "job_type" : "training", ## third level identifier
    "tags" : ["training", "parameter search"],  ## tags for organizing tasks
    "name" : "test", ## bottom level identifier, label of graph in UI
    "config" : hyperparameter_dict, ## dictionary of used hyperparameters
}

