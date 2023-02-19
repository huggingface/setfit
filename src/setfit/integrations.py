import importlib.util
from typing import TYPE_CHECKING, Optional

from .utils import BestRun
from .trainer import SetFitTrainer


def is_optuna_available() -> bool:
    """Check if Optuna is available."""
    return importlib.util.find_spec("optuna") is not None


def default_hp_search_backend() -> Optional[str]:
    """Get the default backend for hyperparameter search."""
    return "optuna" if is_optuna_available() else None


def run_hp_search_optuna(
    trainer: SetFitTrainer, 
    n_trials: int, 
    direction: str, 
    timeout: Optional[int] = None,
    n_jobs: int = 1,
    **kwargs,
) -> BestRun:
    """Run hyperparameter search with Optuna.

    Args:
        trainer: SetFitTrainer object to train and evaluate the model.
        n_trials: Number of trials to run for hyperparameter search.
        direction: Optimization direction for the study.
        timeout: Maximum duration in seconds for running the trials.
        n_jobs: Number of parallel jobs to run.
        **kwargs: Additional arguments to pass to `optuna.create_study`.

    Returns:
        A BestRun object containing the best hyperparameters and results.

    Raises:
        ImportError: If Optuna is not installed.
    """
    if not is_optuna_available():
        raise ImportError("Optuna is not available. Please install it to run hyperparameter search with Optuna.")

    import optuna

    def _objective(trial):
        """Objective function for the Optuna study."""
        trainer.objective = None
        trainer.train(trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)
        return trainer.objective

    study = optuna.create_study(direction=direction, **kwargs)
    study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
    best_trial = study.best_trial
    return BestRun(str(best_trial.number), best_trial.value, best_trial.params, study)
