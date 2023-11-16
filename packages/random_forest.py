import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def dataSetUp():
  # load in training data on each potential synapse
  data = pd.read_csv("./data/train_data.csv")

  # load in additional features for each neuron
  feature_weights = pd.read_csv("./data/feature_weights.csv")
  morph_embeddings = pd.read_csv("./data/morph_embeddings.csv")

  # Merge Data
  # join all feature_weight_i columns into a single np.array column
  feature_weights["feature_weights"] = (
      feature_weights.filter(regex="feature_weight_")
      .sort_index(axis=1)
      .apply(lambda x: np.array(x), axis=1)
  )
  # delete the feature_weight_i columns
  feature_weights.drop(
      feature_weights.filter(regex="feature_weight_").columns, axis=1, inplace=True
  )

  # join all morph_embed_i columns into a single np.array column
  morph_embeddings["morph_embeddings"] = (
      morph_embeddings.filter(regex="morph_emb_")
      .sort_index(axis=1)
      .apply(lambda x: np.array(x), axis=1)
  )
  # delete the morph_embed_i columns
  morph_embeddings.drop(
      morph_embeddings.filter(regex="morph_emb_").columns, axis=1, inplace=True
  )

  data = (
      data.merge(
          feature_weights.rename(columns=lambda x: "pre_" + x),
          how="left",
          validate="m:1",
          copy=False,
      )
      .merge(
          feature_weights.rename(columns=lambda x: "post_" + x),
          how="left",
          validate="m:1",
          copy=False,
      )
      .merge(
          morph_embeddings.rename(columns=lambda x: "pre_" + x),
          how="left",
          validate="m:1",
          copy=False,
      )
      .merge(
          morph_embeddings.rename(columns=lambda x: "post_" + x),
          how="left",
          validate="m:1",
          copy=False,
      )
  )

  # generate the fw_similarity feature
  # cosine similarity function
  def row_feature_similarity(row):
      pre = row["pre_feature_weights"]
      post = row["post_feature_weights"]
      return (pre * post).sum() / (np.linalg.norm(pre) * np.linalg.norm(post))

      # compute the cosine similarity between the pre- and post- feature weights
  data["fw_similarity"] = data.apply(row_feature_similarity, axis=1)

  # generate projection group as pre->post
  data["projection_group"] = (
      data["pre_brain_area"].astype(str)
      + "->"
      + data["post_brain_area"].astype(str)
  )

  # encoding Non-numerical features
  label_encoders = {}
  for column in ['compartment', 'pre_brain_area', 'post_brain_area', 'projection_group']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])


  return data

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

excluded_feature_indices = [0, 30, 31, 32, 33]

def select_features(data, excluded_feature_indices):
  data = data.drop(data.columns[excluded_feature_indices], axis=1)
  return data

def train_test_data_set_up(data):
  train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)
  # Define the label column name
  label_column = 'connected'
  train_data_x = train_data.drop(label_column,axis=1)
  train_data_y = train_data[label_column]

  test_data_x = test_data.drop(label_column, axis=1)
  test_data_y = test_data[label_column]

  return train_data_x, train_data_y, test_data_x, test_data_y, train_data, test_data

def overSampling(data_x, data_y):
  # oversample connected neuron pairs
  ros = RandomOverSampler(random_state=0)
  X_resampled, y_resampled = ros.fit_resample(data_x, data_y)
  return X_resampled, y_resampled


import logging
import datetime
import os

def setup_logging(experiment_name):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_directory = "./logging"  # Replace with your desired path
    os.makedirs(log_directory, exist_ok=True)

    log_filename = f'{log_directory}/{experiment_name}_{current_time}_metrics_log.log'

    logger = logging.getLogger('test')
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.INFO)
    # stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)
    
    return logger


from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

def metric(test_data, logger):
    
    # compute accuracy
    accuracy = accuracy_score(test_data['connected'], test_data['pred'] > .5)
    logger.info(f'Accuracy: {accuracy}')
    print(f'Accuracy: {accuracy}')
    # confusion matrix
    cm = confusion_matrix(test_data['connected'], test_data['pred'] > .5)

    # Extracting TN, FP, FN, TP from the confusion matrix
    TN, FP, FN, TP = cm.ravel()
    logger.info(f'Confusion Matrix: TN={TN}, FP={FP}, FN={FN}, TP={TP}')
    print(f'Confusion Matrix: TN={TN}, FP={FP}, FN={FN}, TP={TP}')

    # Calculating Sensitivity (True Positive Rate)
    sensitivity = TP / (TP + FN)
    logger.info(f'Sensitivity: {sensitivity}')
    print(f'Sensitivity: {sensitivity}')

    # Calculating Specificity (True Negative Rate)
    specificity = TN / (TN + FP)
    logger.info(f'Specificity: {specificity}')
    print(f'Specificity: {specificity}')

    # compute balanced accuracy
    balanced_accuracy = balanced_accuracy_score(test_data['connected'], test_data['pred'] > .5)
    logger.info(f'Balanced Accuracy: {balanced_accuracy}')
    print(balanced_accuracy)


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# create pipeline
def create_pipe(model, train_data_x, train_data_y, test_data_x, test_data_y, test_data):
  pipe = Pipeline(
      [("scaler", StandardScaler()), ("model", model)]
  )
  pipe.fit(train_data_x, train_data_y)
  test_data["pred"] = pipe.predict_proba(test_data_x)[:,1]

  return pipe, test_data


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import logging
from imblearn.ensemble import BalancedRandomForestClassifier

experiment_name = "random_forests"
logger = setup_logging(experiment_name)

data = dataSetUp()
excluded_feature_indices = [0, 30, 31, 32, 33]
data = select_features(data, excluded_feature_indices)
train_data_x, train_data_y, test_data_x, test_data_y, train_data, test_data = train_test_data_set_up(data)
train_data_x, train_data_y = overSampling(train_data_x, train_data_y)

# model = LogisticRegression(random_state=2)
model = RandomForestClassifier(random_state=42)
model = BalancedRandomForestClassifier(random_state=42)
pipe, test_data = create_pipe(model, train_data_x, train_data_y, test_data_x, test_data_y, test_data)

metric(test_data, logger=logger)