from torch.nn.functional import cosine_similarity
from torchmetrics import Metric

from data.simulation.utils import CLIP_PARAMETERS_DICT
from data.simulation.constants import NumericFeature


class CaptionTop1(Metric):
    def __init__(self, clip_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.clip_embeddings = clip_embeddings

        self.add_state("encoder_features", default=[], dist_reduce_fx="cat")
        self.add_state("param_data", default=[], dist_reduce_fx=None)

    def update(self, encoder_features, batch_params):
        self.encoder_features.append(encoder_features)
        self.param_data.append(batch_params)

    def per_sample_outcomes(self) -> list:
        samples = []
        for batch_idx in range(len(self.encoder_features)):
            for i, encoder_features in enumerate(self.encoder_features[batch_idx]):
                sample = []
                for emb_idx, (prefix, data_value, value_idx, _) in enumerate(self.param_data[batch_idx][i]):
                    if value_idx == -1:
                        continue

                    value_type = CLIP_PARAMETERS_DICT[prefix]
                    if isinstance(value_type, NumericFeature):
                        # Continuous features are supervised directly; top-1
                        # classification has no finite candidate vocabulary.
                        continue
                    param_type = value_type.__name__
                    if param_type == "bool" or param_type == "boolean":
                        param_type = "boolean"

                    similarities = []
                    for embedding_key, ref_embedding in self.clip_embeddings[param_type].items():
                        feature = encoder_features[emb_idx]
                        reference = ref_embedding.to(
                            device=feature.device, dtype=feature.dtype
                        )
                        similarity = cosine_similarity(
                            feature.unsqueeze(0), reference.unsqueeze(0)
                        ).item()
                        similarities.append((embedding_key, similarity))

                    similarities.sort(key=lambda x: x[1], reverse=True)
                    top_match = similarities[0][0]

                    is_correct = False
                    if param_type == "boolean":
                        if isinstance(data_value, bool):
                            is_correct = top_match == data_value
                        else:
                            is_correct = (
                                (top_match is True and str(data_value).lower() == "true") or
                                (top_match is False and str(data_value).lower() == "false")
                            )
                    else:
                        is_correct = str(top_match) == str(data_value)

                    sample.append((param_type, is_correct))
                samples.append(sample)
        return samples

    def compute(self) -> dict:
        total_correct = 0
        total_params = 0

        param_metrics = {}

        for sample in self.per_sample_outcomes():
            for param_type, is_correct in sample:
                if param_type not in param_metrics:
                    param_metrics[param_type] = {"correct": 0, "total": 0}

                param_metrics[param_type]["total"] += 1
                total_params += 1
                if is_correct:
                    param_metrics[param_type]["correct"] += 1
                    total_correct += 1

        overall_accuracy = total_correct / total_params if total_params else 0

        type_accuracies = {
            param_type: metrics["correct"] / metrics["total"] if metrics["total"] else 0
            for param_type, metrics in param_metrics.items()
        }

        result = {"overall": overall_accuracy}
        result.update(type_accuracies)

        return result
