from data.simulation.init_embeddings import initialize_all_clip_embeddings
from data.simulation.constants import *
from matplotlib import pyplot as plt
import numpy as np
import random


# CLIP_EMB = initialize_all_clip_embeddings()
# N_CLIP_EMBEDDINGS = 10
# EMBEDDING_DIM = 512
# N_RUNS = 10^6
# N_BINS = 20
# PARAMETERS = [
#     "CameraVerticalAngle",
#     "ShotSize",
#     "SubjectView",
#     "SubjectInFramePosition",
#     "CameraMovementType",
#     "MovementSpeed",

#     "Scale",
#     "MovementEasing",
#     "Direction",
#     "MovementMode",
#     "boolean",
# ]


# def calc_embedding_mean() -> list:
#     means = {}
#     for key, value in CLIP_EMB.items():
#         if key in PARAMETERS:
#             sum = np.zeros(EMBEDDING_DIM)
#             n_emb = 0
#             for _, vector in value.items():
#                 sum += vector.numpy()
#                 n_emb += 1
#             means[key] = sum / n_emb
#     return means


# def calc_embedding_std() -> list:
#     stds = {}
#     means = calc_embedding_mean()
#     for key, value in CLIP_EMB.items():
#         if key in PARAMETERS:
#             sum_squared_error = 0
#             n_emb = 0
#             for _, vector in value.items():
#                 sum_squared_error += (vector.numpy() - means[key]) ** 2
#                 n_emb += 1
#             stds[key] = np.sqrt(sum_squared_error / n_emb)
#     return stds
    

# def get_embedding_name(name: str) -> str:
#     embedding_name = str(name).split(".")[-1].lower()
#     embedding_name = embedding_name.split("_")
#     embedding_name = embedding_name[0] + "".join([item.capitalize() for item in embedding_name[1:]])
#     return embedding_name


# def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# def get_combined_clip_vector(means: list, stds: list, normalize: bool=True) -> np.ndarray:
#     combined_clip_vector = np.zeros(N_CLIP_EMBEDDINGS * EMBEDDING_DIM)
#     for index, (_, parameter) in enumerate(CLIP_PARAMETERS[:10]):    
#         embedding_type = parameter.__name__
#         n_values = len(parameter)
#         random_index = random.randint(0, n_values - 1)
#         parameter_embeddings = list(parameter)
#         selected_embedding_name = get_embedding_name(parameter_embeddings[random_index])
#         selected_embedding_vector = CLIP_EMB[embedding_type][selected_embedding_name]
#         if normalize:
#             key = PARAMETERS[index // 6]
#             mean = means[key]
#             std = stds[key]
#             selected_embedding_vector = (selected_embedding_vector - mean) / std
#         combined_clip_vector[EMBEDDING_DIM * index:EMBEDDING_DIM * (index + 1)] = selected_embedding_vector
#     return combined_clip_vector


# def visualize_histogram(data: list,
#                         title: str,
#                         path_to_save: str,
#                         bins: int=N_BINS,
#                         xlim: tuple=(-1, 1)):
#     plt.figure(figsize=(12, 8))
#     plt.hist(data, bins=bins)
#     plt.xlabel("Cosine Similarities")
#     plt.title(title)
#     plt.grid()
#     if xlim:
#         plt.xlim(xlim)
#     plt.savefig(path_to_save)
#     plt.show()


# def main_combined():
#     means = calc_embedding_mean()
#     stds = calc_embedding_std()
#     similarities = list()
#     for _ in range(N_RUNS):
#         combined_clip_1 = get_combined_clip_vector(means, stds, True)
#         combined_clip_2 = get_combined_clip_vector(means, stds, True)
#         cosine_similarity = compute_cosine_similarity(combined_clip_1, combined_clip_2)
#         similarities.append(cosine_similarity)
#     visualize_histogram(similarities,
#                         "Cosine Similarity Hsitogram",
#                         "cosine_similarities_hist.jpeg")



# def main_single():
#     means = calc_embedding_mean()
#     stds = calc_embedding_std()
#     for key, value in CLIP_EMB.items():
#         if key in PARAMETERS:
#             similarities = list()
#             for index_out, (_, value_n_out) in enumerate(value.items()):
#                 for index_in, (_, value_n_in) in enumerate(value.items()):
#                     if index_in == index_out:
#                         continue
#                     mean = means[key]
#                     std = stds[key]
#                     embedding_1 = (value_n_out - mean) / std
#                     embedding_2 = (value_n_in  - mean) / std
#                     cosine_similarity = compute_cosine_similarity(embedding_1, embedding_2)
#                     similarities.append(cosine_similarity)
#             visualize_histogram(similarities,
#                                 "Cosine Similarity Hsitogram - {}".format(key),
#                                 "cosine_similarities_hist_{}.jpeg".format(key),
#                                 bins=10)













CLIP_EMB = initialize_all_clip_embeddings()
N_CLIP_EMBEDDINGS = 10
EMBEDDING_LENGTH = 512
N_RUNS = 10^7
N_BINS = 20


def get_embedding_name(name: str) -> str:
    embedding_name = str(name).split(".")[-1].lower()
    embedding_name = embedding_name.split("_")
    embedding_name = embedding_name[0] + "".join([item.capitalize() for item in embedding_name[1:]])
    return embedding_name


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_combined_clip_vector():
    combined_clip_vector = np.zeros(N_CLIP_EMBEDDINGS * EMBEDDING_LENGTH)
    for index, (key, parameter) in enumerate(CLIP_PARAMETERS[:10]):    
        embedding_type = parameter.__name__
        n_values = len(parameter)
        random_index = random.randint(0, n_values - 1)
        parameter_embeddings = list(parameter)
        selected_embedding_name = get_embedding_name(parameter_embeddings[random_index])
        selected_embedding_vector = CLIP_EMB[embedding_type][selected_embedding_name]
        combined_clip_vector[EMBEDDING_LENGTH * index:EMBEDDING_LENGTH * (index + 1)] = selected_embedding_vector
    return combined_clip_vector



def visualize_histogram(data: list):
    plt.figure(figsize=(12, 8))
    plt.hist(data, bins=N_BINS)
    plt.xlabel("Cosine Similarities")
    plt.title("Cosine Similarities Histogram")
    plt.xlim([-1, 1])
    plt.grid()
    plt.savefig("cosine_similarities_hist.jpeg")
    plt.show()


def main():
    similarities = list()
    for i in range(N_RUNS):
        combined_clip_1 = get_combined_clip_vector()
        combined_clip_2 = get_combined_clip_vector()
        cosine_similarity = compute_cosine_similarity(combined_clip_1, combined_clip_2)
        similarities.append(cosine_similarity)
    visualize_histogram(similarities)




if __name__ == "__main__":
    main()
    # main_combined()
    # main_single()