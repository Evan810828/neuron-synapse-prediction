from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

class RandomForest():
    def __init__(self, data):
        self.data = data
        # self.model = BalancedRandomForestClassifier(random_state=42, max_depth=25, n_estimators= 150)
        self.model = BalancedRandomForestClassifier()
        # self.excluded_feature_indices = [0, 30, 31, 32, 33]
        self.excluded_feature_indices = [0, 30, 31, 32, 33, 35, 20, 19]

    def select_features(self, data, excluded_feature_indices):
        data = data.drop(data.columns[excluded_feature_indices], axis=1)
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

        return train_data_x, train_data_y, test_data_x, test_data_y, train_data, test_data

    def feature_importance(self, data_x):
        feature_importance = self.model.feature_importances_
        feature_importance_index = sorted(range(len(feature_importance)), key=lambda k: feature_importance[k], reverse=True)
        print('Feature Importance Index:', feature_importance_index)

        for i in feature_importance_index:
            # Format the coefficient to two decimal places
            formatted_feature_importance = "{:.4f}".format(feature_importance[i])
            print(data_x.columns[i], formatted_feature_importance)

        return feature_importance_index

    def data_processing(self):
        data = self.select_features(self.data, self.excluded_feature_indices)
        train_data_x, train_data_y, test_data_x, test_data_y, train_data, test_data = self.train_test_data_set_up(data)
        
        return train_data_x, train_data_y, test_data_x, test_data_y, train_data, test_data, data
    
    def hyperparamter_tuning(self):
        train_data_x, train_data_y, test_data_x, test_data_y, train_data, test_data = self.data_processing()
        rf_search = RandomForestParameterSearch()
        rf_model = rf_search.search_parameters(train_data_x, train_data_y, test_data_x, test_data_y)
        all_performances = rf_search.get_all_performances()

        return rf_model, all_performances
    

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class RandomForestParameterSearch:
    def __init__(self, n_estimators_range=None, max_depth_range=None, 
                 min_samples_split_range=None, min_samples_leaf_range=None):
        self.n_estimators_range = n_estimators_range or [100, 150, 300]
        self.max_depth_range = max_depth_range or [23, 25, 27, 30]
        self.min_samples_split_range = min_samples_split_range or [2, 4, 6]
        self.min_samples_leaf_range = min_samples_leaf_range or [1, 2, 3]

        self.pipe = Pipeline([("scaler", StandardScaler()), ("model", BalancedRandomForestClassifier())])
        self.all_performances = []

    def search_parameters(self, train_data_x, train_data_y, test_data_x, test_data_y):
        best_score = 0
        best_params = {}

        for n_estimators in self.n_estimators_range:
            for max_depth in self.max_depth_range:
                for min_samples_split in self.min_samples_split_range:
                    for min_samples_leaf in self.min_samples_leaf_range:
                        params = {
                            "model__n_estimators": n_estimators,
                            "model__max_depth": max_depth,
                            "model__min_samples_split": min_samples_split,
                            "model__min_samples_leaf": min_samples_leaf
                        }
                        self.pipe.set_params(**params)
                        self.pipe.fit(train_data_x, train_data_y)
                        predictions = self.pipe.predict(test_data_x)
                        score = balanced_accuracy_score(test_data_y, predictions > 0.5)

                        self.all_performances.append({'params': params, 'score': score})
                        print("Parameters: ", params, "Score: ", score)
                        if score > best_score:
                            best_score = score
                            best_params = params

        print("Best score: ", best_score)
        print("Best parameters: ", best_params)
        self.pipe.set_params(**best_params)

        return self.pipe

    def get_all_performances(self):
        return self.all_performances

# Usage Example:
# rf_search = RandomForestParameterSearch()
# rf_model = rf_search.search_parameters(train_data_x, train_data_y, test_data_x, test_data_y)
# all_performances = rf_search.get_all_performances()
