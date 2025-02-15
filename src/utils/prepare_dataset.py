import torch
import datasets
import warnings
import random

from datasets import Dataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from trl.trainer.utils import ConstantLengthDataset


def prepare_dataset(
    dataset,
    tokenizer,
    packing=True,
    dataset_text_field="text",
    max_seq_length=4096,
    formatting_func=None,
    num_of_sequences=1024,
    chars_per_token=3.6,
    remove_unused_columns=True,
    append_concat_token=True,
    add_special_tokens=True,
    skip_prepare_dataset=False,
    encoder=None,
):
    if dataset is None:
        raise ValueError("The dataset should not be None")

    if skip_prepare_dataset:
        return dataset

    # If the dataset is already preprocessed (tokenized), return as-is. Only works if dataset is
    # a datasets.Dataset or datasets.IterableDataset -- not for torch Dataset
    column_names = (
        dataset.column_names
        if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset))
        else None
    )
    if column_names and "input_ids" in column_names:
        if formatting_func is not None:
            warnings.warn(
                "You passed a dataset that is already processed (contains an `input_ids` field) together with a valid formatting function. Therefore `formatting_func` will be ignored."
            )

        return dataset

    # check if torch dataset / dataloader and do nothing
    # see https://github.com/huggingface/trl/pull/1468 for why datasets.IterableDataset needs a separate check
    if isinstance(
        dataset,
        (
            torch.utils.data.IterableDataset,
            torch.utils.data.Dataset,
        ),
    ) and not isinstance(dataset, datasets.IterableDataset):
        return dataset

    if not packing:
        assert ValueError("not impl")
        # return self._prepare_non_packed_dataloader(
        #     tokenizer,
        #     dataset,
        #     dataset_text_field,
        #     max_seq_length,
        #     formatting_func,
        #     add_special_tokens,
        #     remove_unused_columns,
        # )

    else:
        return _prepare_packed_dataloader(
            tokenizer,
            dataset,
            dataset_text_field,
            max_seq_length,
            num_of_sequences,
            chars_per_token,
            formatting_func,
            append_concat_token,
            add_special_tokens,
            encoder=encoder,
        )


def _prepare_packed_dataloader(
    tokenizer,
    dataset,
    dataset_text_field,
    max_seq_length,
    num_of_sequences,
    chars_per_token,
    formatting_func=None,
    append_concat_token=True,
    add_special_tokens=True,
    encoder=None,
):
    if dataset_text_field is not None or formatting_func is not None:
        if tokenizer is None:
            raise ValueError(
                "You need to pass a tokenizer when using `dataset_text_field` with `SFTTrainer`."
            )

        constant_length_iterator = ConstantLengthDatasetHinshi(
            tokenizer,
            dataset,
            dataset_text_field=dataset_text_field,
            formatting_func=formatting_func,
            seq_length=max_seq_length,
            infinite=False,
            num_of_sequences=num_of_sequences,
            chars_per_token=chars_per_token,
            eos_token_id=tokenizer.eos_token_id,
            append_concat_token=append_concat_token,
            add_special_tokens=add_special_tokens,
            encoder=encoder,
        )
        # print('encoder', encoder)
        if isinstance(dataset, datasets.IterableDataset):
            return constant_length_iterator

        def data_generator(constant_length_iterator):
            yield from constant_length_iterator

        try:
            packed_dataset = Dataset.from_generator(
                data_generator,
                gen_kwargs={"constant_length_iterator": constant_length_iterator},
            )
        except (DatasetGenerationError, SchemaInferenceError) as exc:
            raise ValueError(
                "Error occurred while packing the dataset. "
                "Make sure that your dataset has enough samples to at least yield one packed sequence."
            ) from exc
        return packed_dataset
    else:
        raise ValueError(
            "You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want to use the `ConstantLengthDataset`."
        )


class ConstantLengthDatasetHinshi(ConstantLengthDataset):
    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=0,
        shuffle=True,
        append_concat_token=True,
        add_special_tokens=True,
        encoder=None,
    ):
        super().__init__(
            tokenizer,
            dataset,
            dataset_text_field,
            formatting_func,
            infinite,
            seq_length,
            num_of_sequences,
            chars_per_token,
            eos_token_id,
            shuffle,
            append_concat_token,
            add_special_tokens,
        )
        self.encoder = encoder

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn(
                            "The dataset reached end and the iterator is reset to the start."
                        )
                    else:
                        more_examples = False
                        break

            tokenized_inputs = []

            if self.encoder:
                for i, b in enumerate(buffer):
                    id = self.encoder(b, add_special_tokens=self.add_special_tokens)
                    tokenized_inputs.append(id)
                    # if i < 5:
                    #     print('b:', b)
                    #     print('id:', id)
                    #     print('d:', self.tokenizer.decode(id))
                    #     print('-'*100)
            else:
                tokenized_inputs = self.tokenizer(
                    buffer, add_special_tokens=self.add_special_tokens, truncation=False
                )["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.append_concat_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }
