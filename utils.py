import numpy as np
from node2vec import Node2Vec
import networkx as nx
import pandas as pd
from gensim.models import KeyedVectors
from karateclub import DeepWalk
from sklearn.preprocessing import StandardScaler
import pickle

EMBEDDING_FILENAME = './data/node2vec_embeddings.emb'
EMBEDDING_MODEL_FILENAME = './data/node2vec_embeddings.model'

DEEPWALK_EMBEDDING_FILENAME = './data/deepwalk_embeddings.pkl'

WINDOW_SIZE = 10
EMBEDDING_DIMENSION = 128
WALK_LENGTH = 10
NUM_WALKS = 100

TRAIN_PROPORTION = 0.8


def create_edge_list(df):
    """Create an edge list DataFrame from user and challenge data."""
    edge_list = pd.DataFrame().assign(user_id=df['user_id'].astype(int),
                                      challenge_id=df['challenge_id'].astype(int))
    return edge_list


def load_or_create_node2vec_model(edge_list, embedding_filename, embedding_dimension, walk_length, num_walks,
                                  window_size, embedding_model_filename):
    try:
        embedding_model = KeyedVectors.load_word2vec_format(embedding_filename)
    except (OSError, IOError):
        graph = nx.from_pandas_edgelist(edge_list, "user_id", "challenge_id")
        node2vec = Node2Vec(graph, dimensions=embedding_dimension, walk_length=walk_length, num_walks=num_walks,
                            workers=4)
        model = node2vec.fit(window=window_size, min_count=1, batch_words=4)
        model.wv.save_word2vec_format(embedding_filename)
        model.save(embedding_model_filename)
        embedding_model = KeyedVectors.load_word2vec_format(embedding_filename)
    return embedding_model


def create_node2vec_embeddings(df):
    """Create node2vec embeddings for the given DataFrame."""
    edge_list = create_edge_list(df)
    embedding_model = load_or_create_node2vec_model(edge_list, EMBEDDING_FILENAME, EMBEDDING_DIMENSION,
                                                    WALK_LENGTH, NUM_WALKS, WINDOW_SIZE, EMBEDDING_MODEL_FILENAME)
    return embedding_model


def create_deepwalk_embeddings(mooc_df):
    try:
        embedding_dict = pickle.load(open(DEEPWALK_EMBEDDING_FILENAME, 'rb'))
    except (OSError, IOError):

        edge_list = create_edge_list(mooc_df)

        # create Graph
        G = nx.from_pandas_edgelist(edge_list, 'user_id', 'challenge_id')

        # train model and generate embedding
        model = DeepWalk(walk_length=WALK_LENGTH, walk_number=NUM_WALKS, dimensions=EMBEDDING_DIMENSION,
                         window_size=WINDOW_SIZE)
        model.fit(G)
        embedding = model.get_embedding()
        # inv_node_dict = {v: k for k, v in node_dict.items()}

        embedding_dict = {i: embedding[i] for i in range(len(embedding))}
        pickle.dump(embedding_dict, open(DEEPWALK_EMBEDDING_FILENAME, 'wb'))

    return embedding_dict


def calculate_graph_properties(graph_edge_list):
    graph = nx.from_pandas_edgelist(graph_edge_list, "user_id", "challenge_id")

    eig_c = nx.eigenvector_centrality(graph, max_iter=6000)
    graph_edge_list['eig_c_user'] = graph_edge_list['user_id'].map(eig_c)
    graph_edge_list['eig_c_challenge'] = graph_edge_list['challenge_id'].map(eig_c)

    graph_edge_list['degree_user'] = graph_edge_list['user_id'].apply(lambda x: nx.degree(graph, x))
    graph_edge_list['degree_challenge'] = graph_edge_list['challenge_id'].apply(lambda x: nx.degree(graph, x))

    return graph_edge_list


def preprocess_graph_edge_list(graph_edge_list):
    graph_edge_list['user_id'] = graph_edge_list['user_id'].replace('stu_', '', regex=True).astype(int)
    graph_edge_list['challenge_id'] = graph_edge_list['challenge_id'].replace('ch_', '', regex=True).astype(int)
    return graph_edge_list


def combine_dataset_with_graph_properties(mooc_df):
    graph_edge_list = pd.DataFrame().assign(user_id=mooc_df['user_id'].astype(int),
                                            challenge_id=mooc_df['challenge_id'].astype(int))
    graph_properties = calculate_graph_properties(graph_edge_list)

    mooc_with_graph = pd.merge(mooc_df, graph_properties, on=['user_id', 'challenge_id'])
    return mooc_with_graph


def create_embedding(row, embedding_model):
    user_embedding = embedding_model[row['user_id'].astype(int)]
    challenge_embedding = embedding_model[row['challenge_id'].astype(int)]
    return np.concatenate((user_embedding, challenge_embedding))


def combine_dataset_with_embedding(df, embedding_model, embedding_type='node2vec'):
    x = df.copy()

    embedding_matrix = []
    for _, row in x.iterrows():
        embedding = create_embedding(row, embedding_model)
        embedding_matrix.append(embedding)
    embedding_df = pd.DataFrame(embedding_matrix)

    df_with_embedding = pd.concat([x, embedding_df], axis=1)
    df_with_embedding.drop(columns=['user_id', 'challenge_id'], inplace=True)

    filename = f'./data/mooc_with_{embedding_type}_embedding.csv'
    df_with_embedding.to_csv(filename, index=False)

    return df_with_embedding


def split_dataset(df):
    train_end = int(TRAIN_PROPORTION * len(df))

    df_train = df.head(train_end).reset_index(drop=True)
    df_test = df.loc[train_end:len(df)].reset_index(drop=True)

    scaler = StandardScaler()
    df_train['duration'] = scaler.fit_transform(df_train[['duration']])
    df_test['duration'] = scaler.transform(df_test[['duration']])

    if 'degree_user' in df_train.columns:
        df_train['degree_user'] = scaler.fit_transform(df_train[['degree_user']])
        df_test['degree_user'] = scaler.transform(df_test[['degree_user']])
    if 'degree_challenge' in df_train.columns:
        df_train['degree_challenge'] = scaler.fit_transform(df_train[['degree_challenge']])
        df_test['degree_challenge'] = scaler.transform(df_test[['degree_challenge']])

    X_train = df_train.drop('final_score', axis=1)
    y_train = df_train['final_score']

    X_test = df_test.drop('final_score', axis=1)
    y_test = df_test['final_score']

    return X_train, X_test, y_train, y_test
