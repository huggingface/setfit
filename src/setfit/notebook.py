import re

from transformers.utils.notebook import NotebookProgressCallback


class SetFitNotebookProgressCallback(NotebookProgressCallback):
    """
    A variation of NotebookProgressCallback that accepts logs/metrics other than "loss" and "eval_loss".
    In particular, it accepts "embedding_loss", "aspect_embedding_loss", and "polarity_embedding_loss"
    and the corresponding metrics for the validation set.
    """

    def on_log(self, *args, logs=None, **kwargs):
        if logs is not None:
            logs = {key if key != "embedding_loss" else "loss": value for key, value in logs.items()}
        return super().on_log(*args, logs=logs, **kwargs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.training_tracker is not None:
            values = {"Training Loss": "No log", "Validation Loss": "No log"}
            for log in reversed(state.log_history):
                if loss_logs := {
                    key for key in log if key in ("embedding_loss", "aspect_embedding_loss", "polarity_embedding_loss")
                }:
                    values["Training Loss"] = log[loss_logs.pop()]
                    break

            if self.first_column == "Epoch":
                values["Epoch"] = int(state.epoch)
            else:
                values["Step"] = state.global_step
            metric_key_prefix = "eval"
            for k in metrics:
                if k.endswith("_loss"):
                    metric_key_prefix = re.sub(r"\_loss$", "", k)
            _ = metrics.pop("total_flos", None)
            _ = metrics.pop("epoch", None)
            _ = metrics.pop(f"{metric_key_prefix}_runtime", None)
            _ = metrics.pop(f"{metric_key_prefix}_samples_per_second", None)
            _ = metrics.pop(f"{metric_key_prefix}_steps_per_second", None)
            _ = metrics.pop(f"{metric_key_prefix}_jit_compilation_time", None)
            for k, v in metrics.items():
                splits = k.split("_")
                name = " ".join([part.capitalize() for part in splits[1:]])
                if name in ("Embedding Loss", "Aspect Embedding Loss", "Polarity Embedding Loss"):
                    # Single dataset
                    name = "Validation Loss"
                values[name] = v
            self.training_tracker.write_line(values)
            self.training_tracker.remove_child()
            self.prediction_bar = None
            # Evaluation takes a long time so we should force the next update.
            self._force_next_update = True
