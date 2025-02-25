from transformers import Trainer
import wandb
from typing import Dict, Optional

class CuriosityTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_logs = {}


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss = outputs.loss + 0.1 * outputs.primary_loss + 0.05 * outputs.secondary_loss
        
        ## callbackのlogsに記録
        self.custom_logs.update({
            "train/primary_loss": outputs.primary_loss.item() * 0.1,
            "train/secondary_loss": outputs.secondary_loss.item() * 0.05,
            "train/total_loss": total_loss.item(),
            "train/lm_loss": outputs.loss.item() * 0.1,  # lm_loss
        })

        if return_outputs:
            return (total_loss, outputs)
        return total_loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:

        """
        Trainerのlogをオーバーライドし、
        Trainerのlogsと、custom_logsをまとめてwandbに送信
        """
        # stepなどをTrainerのlogsから取得し、custom_logに追加
        self.custom_logs.update(logs)
        if self.is_world_process_zero():
            wandb.log(self.custom_logs, step=self.state.global_step)
        self.custom_logs = {}
        super().log(logs, start_time)