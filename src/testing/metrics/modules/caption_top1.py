from torch.nn.functional import cosine_similarity
from torchmetrics import Metric

from data.simulation.utils import CLIP_PARAMETERS_DICT


class CaptionTop1(Metric):
    def __init__(self, clip_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.clip_embeddings = clip_embeddings
        
        self.add_state("encoder_features", default=[], dist_reduce_fx="cat")
        self.add_state("param_data", default=[], dist_reduce_fx=None)
    
    def get_embedding_name(self, name: str) -> str:
        embedding_name = str(name).split(".")[-1].lower()
        embedding_name = embedding_name.split("_")
        embedding_name = embedding_name[0] + "".join([item.capitalize() for item in embedding_name[1:]])
        return embedding_name
    
    def update(self, encoder_features, batch_params):
        self.encoder_features.append(encoder_features)
        self.param_data.append(batch_params)
    
    def compute(self) -> dict:
        total_correct = 0
        total_params = 0
        
        param_metrics = {}
        
        for batch_idx in range(len(self.encoder_features)):
            for i, encoder_features in enumerate(self.encoder_features[batch_idx]):
                for emb_idx, (prefix, data_value, value_idx, _) in enumerate(self.param_data[batch_idx][i]):
                    if value_idx == -1:
                        continue
                    
                    if prefix.count("_") > 1:
                        prefix = "_".join(prefix.split("_")[-2:])
                    
                    param_type = CLIP_PARAMETERS_DICT[prefix].__name__
                    if param_type == "bool" or param_type == "boolean":
                        param_type = "boolean"
                    
                    if param_type not in param_metrics:
                        param_metrics[param_type] = {"correct": 0, "total": 0}

                    similarities = []
                    for embedding_key, ref_embedding in self.clip_embeddings[param_type].items():
                        similarity = cosine_similarity(encoder_features[emb_idx].unsqueeze(0).to('cuda'), ref_embedding.unsqueeze(0).to('cuda')).item()
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
