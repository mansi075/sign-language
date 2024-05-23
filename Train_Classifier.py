import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_and_preprocess_data(data_path):
  """
  Loads data from pickle file and preprocesses for training.

  Args:
      data_path (str): Path to the pickle file containing data.

  Returns:
      tuple: (X_train, X_test, y_train, y_test) - Training and testing data.
  """
  data_dict = pickle.load(open(data_path, 'rb'))
  data_list = data_dict['data']
  labels = np.asarray(data_dict['labels'])

  # Define a consistent shape for each sample (example: assuming 21 landmarks * 2 coordinates)
  sample_shape = (20, 2)

  X = []
  for sample in data_list:
      # Truncate longer samples to match expected shape
      truncated_sample = np.array(sample[:sample_shape[0]])  # Truncate to sample_shape[0] elements
      X.append(truncated_sample)

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, shuffle=True, stratify=labels)

  return X_train, X_test, y_train, y_test


def train_and_save_model(X_train, X_test, y_train, y_test, model_path):
  """
  Trains a Random Forest Classifier and saves it to a pickle file.

  Args:
      X_train (np.ndarray): Training data features.
      X_test (np.ndarray): Testing data features.
      y_train (np.ndarray): Training data labels.
      y_test (np.ndarray): Testing data labels.
      model_path (str): Path to save the trained model.
  """
  # Train the model
  model = RandomForestClassifier()
  model.fit(X_train, y_train)

  # Evaluate model performance
  y_predict = model.predict(X_test)
  score = accuracy_score(y_predict, y_test)
  print('{}% of samples were classified correctly !'.format(score * 100))

  # Save the model
  with open(model_path, 'wb') as f:
      pickle.dump({'model': model}, f)


if __name__ == '__main__':
  data_path = './data.pickle'  # Replace with your data path
  model_path = 'hand_gesture_classifier.p'  # Replace with desired model filename

  X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
  train_and_save_model(X_train, X_test, y_train, y_test, model_path)
