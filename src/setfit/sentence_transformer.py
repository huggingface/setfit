import time
from typing import Callable, Dict, Iterable, Tuple, Type
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer
import torch
from tqdm.autonotebook import trange, tqdm
from transformers.trainer_callback import TrainerState, TrainerControl, CallbackHandler
from transformers.trainer_utils import speed_metrics

from setfit.training_args import TrainingArguments


def log(args: TrainingArguments, callback_handler: CallbackHandler, state: TrainerState, control: TrainerControl, logs: Dict[str, float]) -> None:
    """
    Log `logs` on the various objects watching training.

    Subclass and override this method to inject custom behavior.

    Args:
        logs (`Dict[str, float]`):
            The values to log.
    """
    if state.epoch is not None:
        logs["epoch"] = round(state.epoch, 2)

    output = {**logs, **{"step": state.global_step}}
    state.log_history.append(output)
    return callback_handler.on_log(args, state, control, logs)


def fit(
    model_body: SentenceTransformer,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    loss_func: nn.Module,
    args: TrainingArguments,
    callback_handler: CallbackHandler,
    state: TrainerState,
    control: TrainerControl,
    # evaluator: SentenceEvaluator = None,  # <- remove
    # epochs: int = 1,  # <- remove
    # steps_per_epoch=None,  # <- remove?
    scheduler: str = "WarmupLinear",
    warmup_steps: int = 10000,
    optimizer_class: Type[Optimizer] = torch.optim.AdamW,
    optimizer_params: Dict[str, object] = {"lr": 2e-5},
    weight_decay: float = 0.01,
    output_path: str = None,
    save_best_model: bool = True,
    max_grad_norm: float = 1,
    use_amp: bool = False,
    # callback: Callable[[float, int, int], None] = None,  # <- remove
    show_progress_bar: bool = True,
    checkpoint_path: str = None,  # <- remove
    checkpoint_save_steps: int = 500,  # <- remove
    checkpoint_save_total_limit: int = 0,  # <- remove
):
    """
    Train the model with the given training objective
    Each training objective is sampled in turn for one batch.
    We sample only as many batches from each objective as there are in the smallest one
    to make sure of equal training with each dataset.

    :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
    :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
    :param epochs: Number of epochs for training
    :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
    :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
    :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
    :param optimizer_class: Optimizer
    :param optimizer_params: Optimizer parameters
    :param weight_decay: Weight decay for model parameters
    :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
    :param output_path: Storage path for the model and evaluation files
    :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
    :param max_grad_norm: Used for gradient normalization.
    :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
    :param callback: Callback function that is invoked after each evaluation.
            It must accept the following three parameters in this order:
            `score`, `epoch`, `steps`
    :param show_progress_bar: If True, output a tqdm progress bar
    :param checkpoint_path: Folder to save checkpoints during training
    :param checkpoint_save_steps: Will save a checkpoint after so many steps
    :param checkpoint_save_total_limit: Total number of checkpoints to store
    """

    """
    ##Add info to model card
    # info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
    info_loss_functions = []
    for dataloader, loss in train_objectives:
        info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
    info_loss_functions = "\n\n".join([text for text in info_loss_functions])

    info_fit_parameters = json.dumps(
        {
            "evaluator": fullname(evaluator),
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "scheduler": scheduler,
            "warmup_steps": warmup_steps,
            "optimizer_class": str(optimizer_class),
            "optimizer_params": optimizer_params,
            "weight_decay": weight_decay,
            "evaluation_steps": evaluation_steps,
            "max_grad_norm": max_grad_norm,
        },
        indent=4,
        sort_keys=True,
    )
    self._model_card_text = None
    self._model_card_vars["{TRAINING_SECTION}"] = ModelCardTemplate.__TRAINING_SECTION__.replace(
        "{LOSS_FUNCTIONS}", info_loss_functions
    ).replace("{FIT_PARAMETERS}", info_fit_parameters)
    """
    # TODO: Loading best model
    # TODO: Saving/checkpointing
    # TODO: args.gradient_accumulation_steps
    # TODO: fp16/bf16, etc.

    state.epoch = 0
    start_time = time.time()
    # TODO: Add max_steps via args.max_steps here?
    state.max_steps = len(train_dataloader) * args.embedding_num_epochs
    control = callback_handler.on_train_begin(args, state, control)

    if use_amp:
        from torch.cuda.amp import autocast

        scaler = torch.cuda.amp.GradScaler()

    model_body.to(model_body._target_device)

    # Use smart batching
    train_dataloader.collate_fn = model_body.smart_batching_collate
    if eval_dataloader:
        eval_dataloader.collate_fn = model_body.smart_batching_collate

    loss_func.to(model_body._target_device)

    model_body.best_score = -9999999

    steps_per_epoch = len(train_dataloader)
    num_train_steps = int(steps_per_epoch * args.embedding_num_epochs)

    # Prepare optimizers
    param_optimizer = list(loss_func.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
    scheduler_obj = model_body._get_scheduler(
        optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
    )

    data_iterator = iter(train_dataloader)

    skip_scheduler = False
    for epoch in range(args.embedding_num_epochs):
        control = callback_handler.on_epoch_begin(args, state, control)

        training_steps = 0

        loss_func.zero_grad()
        loss_func.train()

        for step in range(steps_per_epoch):
            control = callback_handler.on_step_begin(args, state, control)

            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_dataloader)
                data = next(data_iterator)

            features, labels = data
            labels = labels.to(model_body._target_device)
            features = list(map(lambda batch: batch_to_device(batch, model_body._target_device), features))

            if use_amp:
                with autocast():
                    loss_value = loss_func(features, labels)

                scale_before_step = scaler.get_scale()
                scaler.scale(loss_value).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(loss_func.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                skip_scheduler = scaler.get_scale() != scale_before_step
            else:
                loss_value = loss_func(features, labels)
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_func.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()

            if not skip_scheduler:
                scheduler_obj.step()

            training_steps += 1

            state.global_step += 1
            state.epoch = epoch + (step + 1) / steps_per_epoch
            control = callback_handler.on_step_end(args, state, control)

            if control.should_log:
                learning_rate = scheduler_obj.get_last_lr()[0]
                metrics = {"embedding_loss": round(loss_value.item(), 4), "learning_rate": learning_rate}
                control = log(args, callback_handler, state, control, metrics)

            if control.should_evaluate:
                # self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)
                eval_loss = evaluate_with_loss(model_body, eval_dataloader, loss_func, show_progress_bar, use_amp)
                learning_rate = scheduler_obj.get_last_lr()[0]
                metrics = {"eval_embedding_loss": round(eval_loss, 4), "learning_rate": learning_rate}
                control = log(args, callback_handler, state, control, metrics)
                control = callback_handler.on_evaluate(args, state, control, metrics)
                if state.best_metric is None or eval_loss < state.best_metric:
                    state.best_metric = eval_loss

                loss_func.zero_grad()
                loss_func.train()

            if (
                checkpoint_path is not None
                and checkpoint_save_steps is not None
                and checkpoint_save_steps > 0
                and state.global_step % checkpoint_save_steps == 0
            ):
                model_body._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, state.global_step)

            if control.should_epoch_stop or control.should_training_stop:
                break

        control = callback_handler.on_epoch_end(args, state, control)

        if control.should_training_stop:
            break

    if output_path is not None:  # No evaluator, but output path: save final model version
        model_body.save(output_path)

    if checkpoint_path is not None:
        model_body._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, state.global_step)

    control = callback_handler.on_train_end(args, state, control)

    num_train_samples = state.max_steps * args.embedding_batch_size  # * args.gradient_accumulation_steps
    metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=state.max_steps)
    # TODO: This isn't always printed
    log(args, callback_handler, state, control, metrics)

    # eval_start_time = time.time()
    # num_eval_samples = len(eval_dataloader)  # args.max_steps * args.embedding_batch_size  # * args.gradient_accumulation_steps
    # num_eval_steps = num_eval_samples * args.embedding_num_epochs
    # metrics.update(speed_metrics("eval", eval_start_time, num_samples=num_eval_samples, num_steps=num_eval_steps))


def evaluate_with_loss(model_body: SentenceTransformer, eval_dataloader: DataLoader, loss_func: nn.Module, show_progress_bar: bool, use_amp: bool):
    model_body.eval()

    if use_amp:
        from torch.cuda.amp import autocast

        scaler = torch.cuda.amp.GradScaler()

    losses = []
    for data in tqdm(iter(eval_dataloader), leave=False):
        features, labels = data
        labels = labels.to(model_body._target_device)
        features = list(map(lambda batch: batch_to_device(batch, model_body._target_device), features))

        if use_amp:
            with autocast():
                loss_value = loss_func(features, labels)

            losses.append(scaler.scale(loss_value).item())
        else:
            losses.append(loss_func(features, labels).item())

    model_body.train()
    return sum(losses) / len(losses)
