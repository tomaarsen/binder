import logging
from typing import Any, List, Dict, Union
from dataclasses import dataclass

import torch
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput


logger = logging.getLogger(__name__)


@dataclass
class Span:
    type_id: int
    start: int
    end: int
    start_mask: List[int]
    end_mask: List[int]
    span_mask: List[int]


@dataclass
class BinderDataCollator:
    type_input_ids: torch.Tensor
    type_attention_mask: torch.Tensor
    type_token_type_ids: torch.Tensor
    entity_type_id_to_str: List[str]
    entity_type_str_to_id: Dict[str, int]

    def __post_init__(self):
        self.type_input_ids = torch.tensor(self.type_input_ids)
        self.type_attention_mask = torch.tensor(self.type_attention_mask)
        if self.type_token_type_ids is not None:
            self.type_token_type_ids = torch.tensor(self.type_token_type_ids)

    def __call__(self, features: List) -> Dict[str, Any]:
        batch = {}
        batch['input_ids'] = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        batch['attention_mask'] = torch.tensor([f['attention_mask'] for f in features], dtype=torch.bool)
        if "token_type_ids" in features[0]:
            batch['token_type_ids'] = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)

        batch['type_input_ids'] = self.type_input_ids
        batch['type_attention_mask'] = self.type_attention_mask
        if self.type_token_type_ids is not None:
            batch['type_token_type_ids'] = self.type_token_type_ids

        if 'ner' in features[0]:
            # For training
            ner = {}
            # [batch_size, num_types, seq_length]
            start_negative_mask = torch.tensor([], dtype=torch.bool)
            end_negative_mask = torch.tensor([], dtype=torch.bool)
            # [batch_size, num_types, seq_length, seq_length]
            span_negative_mask = torch.tensor([], dtype=torch.bool)

            # Collate mention span examples.
            for feature_id, feature in enumerate(features):
                entity_types = feature["ner"]["entity_types"]
                entity_start_tokens = feature["ner"]["entity_start_tokens"]
                entity_end_tokens = feature["ner"]["entity_end_tokens"]
                token_start_mask = feature["ner"]["token_start_mask"]
                token_end_mask = feature["ner"]["token_end_mask"]
                default_span_mask = [
                    [
                        (j - i >= 0) * s * e for j, e in enumerate(token_end_mask)
                    ]
                    for i, s in enumerate(token_start_mask)
                ]
                # batch_size x num_classes x seq_length
                local_start_negative_mask = torch.tensor(token_start_mask, dtype=torch.bool)
                local_start_negative_mask[0] = 1
                local_start_negative_mask = local_start_negative_mask.repeat(len(self.entity_type_id_to_str), 1).unsqueeze(0)
                start_negative_mask = torch.concat((start_negative_mask, local_start_negative_mask))

                local_end_negative_mask = torch.tensor(token_end_mask, dtype=torch.bool)
                local_end_negative_mask[0] = 1
                local_end_negative_mask = local_end_negative_mask.repeat(len(self.entity_type_id_to_str), 1).unsqueeze(0)
                end_negative_mask = torch.concat((end_negative_mask, local_end_negative_mask))

                # batch_size x num_classes x seq_length x seq_length
                local_span_negative_mask = torch.tensor(default_span_mask, dtype=torch.bool)
                local_span_negative_mask[0, 0] = 1
                local_span_negative_mask = local_span_negative_mask.repeat(len(self.entity_type_id_to_str), 1, 1).unsqueeze(0)
                span_negative_mask = torch.concat((span_negative_mask, local_span_negative_mask))

                # Create negative masks, i.e. with all gold entities zero'd out
                # annotations = []
                for entity_type_id, start_token_index, end_token_index in zip(entity_types, entity_start_tokens, entity_end_tokens):
                    # 0 => "ignore these scores" in contrastive learning of [CLS]
                    # while 1 => decrease these scores
                    start_negative_mask[feature_id, entity_type_id, start_token_index] = 0
                    end_negative_mask[feature_id, entity_type_id, end_token_index] = 0
                    span_negative_mask[feature_id, entity_type_id, start_token_index, end_token_index] = 0

            feature_ids = [i for i, feature in enumerate(features) for _ in range(len(feature["ner"]["entity_types"]))]
            span_type_ids = [entity_type for feature in features for entity_type in feature["ner"]["entity_types"]]

            ner['start_negative_mask'] = start_negative_mask
            ner['end_negative_mask'] = end_negative_mask
            ner['span_negative_mask'] = span_negative_mask

            ner["example_indices"] = [feature_ids, span_type_ids]
            ner["example_starts"] = [start_token for feature in features for start_token in feature["ner"]["entity_start_tokens"]]
            ner["example_ends"] = [end_tokens for feature in features for end_tokens in feature["ner"]["entity_end_tokens"]]

            batch['ner'] = ner

        return batch


class BinderTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
        )

        predictions = self.post_process_function(eval_examples, eval_dataset, output.predictions)
        metrics = predictions["metrics"]

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        return metrics


    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        output = self.evaluation_loop(
            predict_dataloader,
            description="Prediction",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
        )

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = predictions["metrics"]

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)

        return PredictionOutput(predictions=predictions["predictions"], label_ids=predictions["labels"], metrics=metrics)
