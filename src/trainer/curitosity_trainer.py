from transformers import Trainer

class CuriosityTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        self.log({
            "primary_loss": outputs.primary_loss.item() * 0.1,
            "secondary_loss": outputs.secondary_loss.item() * 0.05,
            "loss": outputs.loss.item(), 
            "total_loss": total_loss.item()
        })

        if return_outputs:
            return (total_loss, outputs)
        return total_loss