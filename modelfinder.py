from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


class Model_Finder:
    """
    This class shall  be used to find the model with best accuracy and AUC score.
    Written By: Higher Ed Lab
    Version: 3.8.0
    Revisions: None

    """

    def __init__(self):
        self.clf = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')

    def get_best_params_for_random_forest(self, train_x, train_y):
        """
                Method Name: get_best_params_for_random_forest
                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                             Use Hyper Parameter Tuning.
                Output: The model with the best parameters
                On Failure: Raise Exception

                Written By: Higher Ed Lab
                Version: 3.8.0
                Revisions: None

        """
        self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                           "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}
        self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5, verbose=3)
        self.grid.fit(train_x, train_y)
        self.criterion = self.grid.best_params_['criterion']
        self.max_depth = self.grid.best_params_['max_depth']
        self.max_features = self.grid.best_params_['max_features']
        self.n_estimators = self.grid.best_params_['n_estimators']

        # creating a new model with the best parameters
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                          max_depth=self.max_depth, max_features=self.max_features)
        # training the mew model
        self.clf.fit(train_x, train_y)
        return self.clf

    def get_best_params_for_xgboost(self, train_x, train_y):

        """
                Method Name: get_best_params_for_xgboost
                Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                             Use Hyper Parameter Tuning.
                Output: The model with the best parameters
                On Failure: Raise Exception

                Written By: Higher Ed Lab
                Version: 3.8.0
                Revisions: None

        """
        self.param_grid_xgboost = {

            'learning_rate': [0.5, 0.1, 0.01, 0.001],
            'max_depth': [3, 5, 10, 20],
            'n_estimators': [10, 50, 100, 200]

        }
        self.grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), self.param_grid_xgboost, verbose=3,
                                 cv=5)
        self.grid.fit(train_x, train_y)
        self.learning_rate = self.grid.best_params_['learning_rate']
        self.max_depth = self.grid.best_params_['max_depth']
        self.n_estimators = self.grid.best_params_['n_estimators']
        self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth,
                                 n_estimators=self.n_estimators)
        self.xgb.fit(train_x, train_y)
        return self.xgb


    def get_best_model(self, train_x, train_y, test_x, test_y):
        """
                Method Name: get_best_model
                Description: Find out the Model which has the best AUC score.
                Output: The best model name and the model object
                On Failure: Raise Exception

                Written By: Higher Ed Lab
                Version: 3.8.0
                Revisions: None

        """
        self.xgboost = self.get_best_params_for_xgboost(train_x, train_y)
        self.prediction_xgboost = self.xgboost.predict(test_x)

        if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
            self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
        else:
            self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost)  # AUC for XGBoost

        # create best model for Random Forest
        self.random_forest = self.get_best_params_for_random_forest(train_x, train_y)
        self.prediction_random_forest = self.random_forest.predict(test_x)  # prediction using the Random Forest Algorithm

        if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
            self.random_forest_score = accuracy_score(test_y, self.prediction_random_forest)
        else:
            self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest)  # AUC for Random Forest

        if (self.random_forest_score < self.xgboost_score):
            return 'XGBoost', self.xgboost
        else:
            return 'RandomForest', self.random_forest
