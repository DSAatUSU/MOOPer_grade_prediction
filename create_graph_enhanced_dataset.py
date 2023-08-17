from utils import *
import pandas as pd


mooc_df = pd.read_csv('./data/grade_prediction_mooc.csv')

node2vec_embedding = create_node2vec_embeddings(mooc_df)
deepwalk_embedding = create_deepwalk_embeddings(mooc_df)

mooc_with_graph = combine_dataset_with_graph_properties(mooc_df)

combine_dataset_with_embedding(mooc_with_graph, node2vec_embedding)
combine_dataset_with_embedding(mooc_with_graph, deepwalk_embedding, embedding_type='deepwalk')
