from transformers import Trainer

class CuriosityTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, attention_mask, labels = inputs

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss = outputs.loss + 0.1 * outputs.primary_loss + 0.05 * outputs.secondary_loss
        if return_outputs:
            return (total_loss, outputs)
        return total_loss