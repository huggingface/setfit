import importlib.util
from typing import TYPE_CHECKING

from .utils import BestRun


if TYPE_CHECKING:
    from .trainer import SetFitTrainer


def is_optuna_available():
    return importlib.util.find_spec("optuna") is not None


def default_hp_search_backend():
    if is_optuna_available():
        return "optuna"


def run_hp_search_optuna(trainer: "SetFitTrainer", n_trials: int, direction: str, **kwargs) -> BestRun:
    import optuna

    # Heavily inspired by transformers.integrations.run_hp_search_optuna
    # https://github.com/huggingface/transformers/blob/cbb8a37929c3860210f95c9ec99b8b84b8cf57a1/src/transformers/integrations.py#L160
    def _objective(trial):
        trainer.objective = None
        trainer.train(trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)
        return trainer.objective

    timeout = kwargs.pop("timeout", None)
    n_jobs = kwargs.pop("n_jobs", 1)
    study = optuna.create_study(direction=direction, **kwargs)
    study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
    best_trial = study.best_trial
    return BestRun(str(best_trial.number), best_trial.value, best_trial.params, study)
