from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class Logistic():
    def __init__(self, data):
        self.data = data
        self.model = LogisticRegression(random_state=2)

    def select_features(self):
        excluded_feature_indices = [0, 30, 31, 32, 33]
        data = self.data.drop(self.data.columns[excluded_feature_indices], axis=1)
        return data
    
    def overSampling(self, data_x, data_y):
        # oversample connected neuron pairs
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(data_x, data_y)
        return X_resampled, y_resampled

    def train_test_data_set_up(self, data):
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)
        # Define the label column name
        label_column = 'connected'
        train_data_x = train_data.drop(label_column,axis=1)
        train_data_y = train_data[label_column]

        test_data_x = test_data.drop(label_column, axis=1)
        test_data_y = test_data[label_column]
        
        train_data_x, train_data_y = self.overSampling(train_data_x, train_data_y)

        return train_data_x, train_data_y, test_data_x, test_data_y, train_data, test_data

    def data_processing(self):
        data = self.select_features()
        train_data_x, train_data_y, test_data_x, test_data_y, train_data, test_data = self.train_test_data_set_up(data)
        
        return train_data_x, train_data_y, test_data_x, test_data_y, train_data, test_data