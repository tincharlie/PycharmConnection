
# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from modelfinder import Model_Finder
from file_operations import file_methods
from application_logging import logger
import pandas as pd

#Creating the common Logging object
class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):
        # Logging the start of Training
        self.path = pd.read_csv("./Data_For_Prediction/contact_data_2022-09-02_102457.csv")
        self.X = self.path[["Number_of_Pageviews", "Number_of_Sessions", "Average_Pageviews", "Google_ad_click_id", 'Recent_Conversion',
     'BBB_START_CLASS_Total', 'VIEW_PAGE_ANALYTICS_Total','Signin_Total', 'Signup_Total', 'BBB_CREATE_CLASS_Total','VIEW_PAGE_UPGRADE_Total']]
        self.Y = self.path[["LifecycleStage"]]

