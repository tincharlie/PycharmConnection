import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

path = "G:/Customer Subscription prediction/PyCharmSalePred/Raw_data/Contact_2.xlsx"

add_path = "G:/Customer Subscription prediction/PyCharmSalePred/Raw_data/"
def data_read(path):
    """
    Method name: data reader
    Description: Reading the data by using pandas read excel

    Created by: HigheredLab
    Versions: 3.8
    Output: We can read the data in the form of excel in python output.
    """
    df_final = pd.read_excel(path)
    return df_final


df_final = data_read(path)


def ext_data():
    """
    Method name: external data
    Description: Taking this data from mixpanel for doing the prediction.
                 We also have hubspot data set But that is not sufficient
                 for prediction.

    Created by: HigheredLab
    Versions: 3.8
    Output: We can read the data in the form of excel in python output.
            All mix panel excel will be there with the hubspot data.
    """
    a = pd.read_excel(add_path+"View_page_upgrade.xlsx")
    b = pd.read_excel(add_path+"View_page_analytics.xlsx")
    c = pd.read_excel(add_path+"BBB_create_class.xlsx")
    d = pd.read_excel(add_path+"Contact_2.xlsx")
    e = pd.read_excel(add_path+"Signin-Total.xlsx")
    f = pd.read_excel(add_path+"Signup-Total.xlsx")
    g = pd.read_excel(add_path+"BB_start.xlsx")
    return a, b, c, d, e, f, g


a, b, c, d, e, f, g = ext_data()


# def new_ext_data():
#     main_up = pd.merge(df_final, f, on = ['email', 'email'])
#     main_in = pd.merge(main_up, e, on = ['email', 'email'])
#     BBB_create = pd.merge(main_in, c, on = ['email', 'email'])
#     View_analytic = pd.merge(BBB_create, b, on = ['email', 'email'])
#     View_upgrade= pd.merge(View_analytic, a, on = ['email', 'email'])
#     BBB_start = pd.merge(View_upgrade, g , on = ['email', 'email'])
#     return main_up, main_in, BBB_create, View_analytic, View_upgrade, BBB_start
# main_up, main_in, BBB_create, View_analytic, View_upgrade, BBB_strt = new_ext_data()

def latest_ext_data():
    """
    Method name: latest external data
    Description: Join all the hubspot and mixpanel data by using merge function.
                 Prepared for further analysis and will complete the prediction by the help
                 of this data.
    Created by: HigheredLab
    Versions: 3.8
    Output: Return all data merged with the main data.
    """
    BBB_start = pd.merge(df_final, g, on=["email", "email"]).drop(["Event", "Unnamed: 3"], axis=1)
    BBB_viewpage = pd.merge(df_final, a, on=["email", "email"]).drop(["Event"], axis=1)
    BBB_analytics = pd.merge(df_final, b, on=["email", "email"]).drop(["Event"], axis=1)
    BBB_createclass = pd.merge(df_final, c, on=["email", "email"]).drop(["Event"], axis=1)
    BBB_signin = pd.merge(df_final, e, on=["email", "email"]).drop(["Event"], axis=1)
    BBB_signup = pd.merge(df_final, f, on=["email", "email"]).drop(["Event"], axis=1)
    return BBB_start, BBB_viewpage, BBB_analytics, BBB_createclass, BBB_signin, BBB_signup


BBB_start, BBB_viewpage, BBB_analytics, BBB_createclass, BBB_signin, BBB_signup = latest_ext_data()


def replacer(df):
    """
    Method name: replacer
    Description: To replace the value categorical and continous both.
                 For that I am creating this method to replace continous
                 value with mean and median, and categorical values with mode.
    Created by: HigheredLab
    Versions: 3.8
    Output: None :- Bcz we will not use any return statement here,
            so if we will pass the data it will automatically replacer
            apply no need to create the new variable for that.
    """
    import pandas as pd
    Q = pd.DataFrame(df.isna().sum(), columns=['mv'])
    R = Q[Q.mv > 0]
    for i in R.index:
        if df[i].dtypes == 'object':
            mode = df[i].mode()[0]
            df[i] = df[i].fillna(mode)
        else:
            mean = df[i].mean()
            df[i] = df[i].fillna(mean)


# replacer(BBB_start)

def ANOVA(df, cat, con):
    """
    Method name: anova
    Description: In all the columns how many features are usefull for predicition.
                 That we will know using this method.
    Created by: HigheredLab
    Versions: 3.8
    Output: Return p value which should be less than 0.05 for each column so that
            we will say that is the best for our feature.
    Params: df-dataFrame or excel , cat- categorical and con- continous values we have to pass through this method.
    """
    from pandas import DataFrame
    from statsmodels.api import OLS
    from statsmodels.formula.api import ols
    rel = con + " ~ " + cat
    model = ols(rel, df).fit()
    from statsmodels.stats.anova import anova_lm
    anova_results = anova_lm(model)
    Q = DataFrame(anova_results)
    a = Q['PR(>F)'][cat]
    return round(a, 3)


def preprocessing(df):
    """
    Method name: preprocessing
    Description: This method is applicable when we are ready with our features.
                 It will take the columns for prediction and convert that into
                 standardized format, and label format. So that we can understand
                 properly.
    Created by: HigheredLab
    Versions: 3.8
    Output: Feature Selected Standardized data.
    """
    import pandas as pd
    cat = []
    con = []
    for i in df.columns:
        if (df[i].dtypes == "object"):
            cat.append(i)
        else:
            con.append(i)
    X1 = pd.get_dummies(df[cat])
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X2 = pd.DataFrame(ss.fit_transform(df[con]), columns=con)
    X3 = X2.join(X1)
    return X3


def Created_DF(df):
    """
    Method name: created df
    Description: Here it is creating the main features in the right format.
                 Means convert categorical to continous and binary. So in
                 this if lead generated from google ad this will right (Yes-1)
                 else (No-0)
                 Same for recent conversion with the leads will come in Y and else
                 part will come in N.
    Created by: HigheredLab
    Versions: 3.8
    Output: Convert Big string to y and n & 1 and 0.
    """
    df[["Recent_Conversion"]] = df[["Recent_Conversion"]].replace(np.nan, 0)
    df[["Google_ad_click_id"]] = df[["Google_ad_click_id"]].replace(np.nan, 0)
    Y = df[["Google_ad_click_id"]]
    Q = []
    for i in Y.Google_ad_click_id:
        if (i == 0):
            Q.append(0)
        else:
            Q.append(1)
    df[["Google_ad_click_id"]] = Q
    Y = df[["Recent_Conversion"]]
    k = []
    for i in Y.Recent_Conversion:
        if (i == "Trial Registration demo.higheredlab.com"):
            k.append(1)
        else:
            k.append(0)
    df[["Recent_Conversion"]] = k
    return df


def model(BBB_start, ycol):
    """
    Method name: model
    Description: In this part of the code we can see BBB_start class it means that
                 we already created in the latest external method. That data we are using here
                 bcz we want to predict the columns here for the main lead generation predicton.
    Created by: HigheredLab
    Versions: 3.8
    Output: Return lr- linear regreassion model to predict the main values.
    Params: It wants every mixpanel merge with hubspot dataset, and ycolumn.
    """

    A = BBB_start[['Number_of_Pageviews', 'Number_of_Sessions', 'Google_ad_click_id',
                   'Recent_Conversion', ycol]]

    X = A[['Number_of_Pageviews', 'Number_of_Sessions', 'Google_ad_click_id',
           'Recent_Conversion']]
    Y = A[[ycol]]

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X1 = pd.DataFrame(ss.fit_transform(X[['Number_of_Pageviews', 'Number_of_Sessions']]),
                      columns=['Number_of_Pageviews', 'Number_of_Sessions'])
    X2 = X[['Google_ad_click_id', 'Recent_Conversion']]
    Xnew = X1.join(X2)
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(Xnew, Y, random_state=23)

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr = lr.fit(xtrain, ytrain)
    return lr


def Test_Model(df1, model):
    """
    Method name: test model
    Description: There we will test our data and get the predicted columns for further prediction.
                 Also used some previous code to convert the categorical to continous, standardized
                 the testing data, then moving to model creation part to get the prediction in the integer
                 form.
    Created by: HigheredLab
    Versions: 3.8
    Output: Return dataframe to make the prediction.
    Params: It needs testing data and model which is created for predicition.
    """
    df = pd.read_excel(df1)
    df = df[["Google ad click id", "Recent Conversion", "Number of Pageviews", "Number of Sessions"]]
    df.columns = ["Google_ad_click_id", "Recent_Conversion", "Number_of_Pageviews", "Number_of_Sessions"]
    df[["Recent_Conversion"]] = df[["Recent_Conversion"]].replace(np.nan, 0)
    df[["Google_ad_click_id"]] = df[["Google_ad_click_id"]].replace(np.nan, 0)
    Y = df[["Google_ad_click_id"]]
    Q = []
    for i in Y.Google_ad_click_id:
        if (i == 0):
            Q.append(0)
        else:
            Q.append(1)
    df[["Google_ad_click_id"]] = Q
    Y = df[["Recent_Conversion"]]
    k = []
    for i in Y.Recent_Conversion:
        if (i == "Trial Registration demo.higheredlab.com"):
            k.append(1)
        else:
            k.append(0)
    df[["Recent_Conversion"]] = k

    A = df[['Number_of_Pageviews', 'Number_of_Sessions', 'Google_ad_click_id',
            'Recent_Conversion']]

    X = A[['Number_of_Pageviews', 'Number_of_Sessions', 'Google_ad_click_id',
           'Recent_Conversion']]

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X1 = pd.DataFrame(ss.fit_transform(X[['Number_of_Pageviews', 'Number_of_Sessions']]),
                      columns=['Number_of_Pageviews', 'Number_of_Sessions'])
    X2 = X[['Google_ad_click_id', 'Recent_Conversion']]
    Xnew = X1.join(X2)
    pred_model = model.predict(Xnew).astype("int32")
    return pred_model


def BB_start(BBB_start, path):
    """
    Method name: bbb start class
    Description: In the  bbb start class used last method created df for google ad and recent conversion,
                 thus we have add model method here to create the bbb start linear regression model. All important columns
                 are converted in the form for best features to predict the BBB_start class on the basis of those colums as following:
                 'Number_of_Pageviews','Number_of_Sessions', 'Google_ad_click_id','Recent_Conversion'
    Created by: HigheredLab
    Versions: 3.8
    Output: Return dataframe of BBB_start_class.
    Params: Only testing path and BBB_start_class created file.
    """
    BBB_start = Created_DF(BBB_start)
    lr_start = model(BBB_start, "BBB_START_CLASS - Total")
    BB_start_pred = Test_Model(path, lr_start)
    BBB_start = pd.DataFrame(BB_start_pred)
    BBB_start.columns = ["BBB_START_CLASS_Total"]
    return BBB_start


def BB_analytics(BBB_analytics, path):
    """
    Method name: bbb analytics
    Description: In the  bbb view page analytics used last method created df for google ad and recent conversion,
                 thus we have add model method here to create the bbb view page analytics linear regression model.
                 All important columns are converted in the form for best features to predict the bbb view page analytics
                 on the basis of those colums as following:
                 'Number_of_Pageviews','Number_of_Sessions', 'Google_ad_click_id','Recent_Conversion'
    Created by: HigheredLab
    Versions: 3.8
    Output: Return dataframe of view page analytics.
    Params: Only testing path and bbb_analytics created file.
    """
    BBB_analytics = Created_DF(BBB_analytics)
    lr_analytics = model(BBB_analytics, "VIEW_PAGE_ANALYTICS - Total")
    BBB_analytics_pred = Test_Model(path, lr_analytics)
    BBB_analytics = pd.DataFrame(BBB_analytics_pred)
    BBB_analytics.columns = ["VIEW_PAGE_ANALYTICS_Total"]
    return BBB_analytics


def BB_signin(BBB_signin, path):
    """
    Method name: bb signin
    Description: In the  bbb signin used last method created df for google ad and recent conversion,
                 thus we have add model method here to create the bbb signin linear regression model.
                 All important columns are converted in the form for best features to predict the bbb sign in
                 on the basis of those colums as following:
                 'Number_of_Pageviews','Number_of_Sessions', 'Google_ad_click_id','Recent_Conversion'
    Created by: HigheredLab
    Versions: 3.8
    Output: Return dataframe of signin total.
    Params: Only testing path and bbb_signin created file.
    """
    BBB_signin = Created_DF(BBB_signin)
    lr_signin = model(BBB_signin, "Signin - Total")
    BBB_signin_pred = Test_Model(path, lr_signin)
    BBB_signin = pd.DataFrame(BBB_signin_pred)
    BBB_signin.columns = ["Signin_Total"]
    return BBB_signin


def BB_signup(BBB_signup, path):
    """
    Method name: bb signup

    Description: In the  bbb signup used last method created df for google ad and recent conversion,
                 thus we have add model method here to create the bbb signup linear regression model.
                 All important columns are converted in the form for best features to predict the bbb signup
                 on the basis of those colums as following:
                 'Number_of_Pageviews','Number_of_Sessions', 'Google_ad_click_id','Recent_Conversion'

    Created by: HigheredLab

    Versions: 3.8

    Output: Return dataframe of signup total.

    Params: Only testing path and bbb_signup created file.
    """
    BBB_signup = Created_DF(BBB_signup)
    lr_signup = model(BBB_signup, "Signup - Total")
    BBB_signup_pred = Test_Model(path, lr_signup)
    BBB_signup = pd.DataFrame(BBB_signup_pred)
    BBB_signup.columns = ["Signup_Total"]
    return BBB_signup


def BB_createclass(BBB_createclass, path):
    """
    Method name: bb createclass

    Description: In the  bbb createclass used last method created df for google ad and recent conversion,
                 thus we have add model method here to create the bbb createclass linear regression model.
                 All important columns are converted in the form for best features to predict the bbb createclass
                 on the basis of those colums as following:
                 'Number_of_Pageviews','Number_of_Sessions', 'Google_ad_click_id','Recent_Conversion'

    Created by: HigheredLab

    Versions: 3.8

    Output: Return dataframe of bbb createclass.

    Params: Only testing path and bbb_createclass created file.
    """
    BBB_createclass = Created_DF(BBB_createclass)
    lr_createclass = model(BBB_createclass, "BBB_CREATE_CLASS - Total")
    BBB_createclass_pred = Test_Model(path, lr_createclass)
    BBB_createclass = pd.DataFrame(BBB_createclass_pred)
    BBB_createclass.columns = ["BBB_CREATE_CLASS_Total"]
    return BBB_createclass


def BB_viewpage(BBB_viewpage, path):
    """
    Method name: bb viewpage

    Description: In the  bbb viewpage used last method created df for google ad and recent conversion,
                 thus we have add model method here to create the bbb viewpage linear regression model.
                 All important columns are converted in the form for best features to predict the bbb viewpage
                 on the basis of those colums as following:
                 'Number_of_Pageviews','Number_of_Sessions', 'Google_ad_click_id','Recent_Conversion'


    Created by: HigheredLab

    Versions: 3.8

    Output: Return dataframe of bbb viewpage.

    Params: Only testing path and bbb_viewpage created file.
    """

    BBB_viewpage = Created_DF(BBB_viewpage)
    lr_viewpage = model(BBB_viewpage, "VIEW_PAGE_UPGRADE - Total")
    BBB_viewpage_pred = Test_Model(path, lr_viewpage)
    BBB_viewpage = pd.DataFrame(BBB_viewpage_pred)
    BBB_viewpage.columns = ["VIEW_PAGE_UPGRADE_Total"]
    return BBB_viewpage


path = "Raw_data/Cust_Sub_2022-07-02.xlsx"


def mix_panel_features(path):
    """
    Method name: mix panel features

    Description: Here we will attached all the created method and this method merge all the external
                 feature into a single dataframe. The mix panel feature contains predicted columns and
                 merge it all together and convert it into the main external data frame.

    Created by: HigheredLab

    Versions: 3.8

    Output: Return all mixpanel dataframe.

    Params: Only testing path.
    """

    BB_start_test1 = BB_start(BBB_start, path)
    BB_analytics_test1 = BB_analytics(BBB_analytics, path)
    BB_signin_test1 = BB_signin(BBB_signin, path)
    BB_signup_test1 = BB_signup(BBB_signup, path)
    BB_create_test1 = BB_createclass(BBB_createclass, path)
    BB_viewpage_test1 = BB_viewpage(BBB_viewpage, path)

    A1 = BB_start_test1.join(BB_analytics_test1)
    # A1 = BB_analytics_test1

    A2 = A1.join(BB_signin_test1)
    A3 = A2.join(BB_signup_test1)
    A4 = A3.join(BB_create_test1)
    A5 = A4.join(BB_viewpage_test1)
    return A5


def data_test_joiner(path):
    """
    Method name: data test joiner

    Description: We are passing our test data for further data and join only those features
                 which really need for main prediction of subscription. We have Google ad, Recent Conversion,
                 Contact ID, Email, Average Pageviews , Number of Pageviews , Number of Sessions , Lifecycle Stage.

    Created by: HigheredLab

    Versions: 3.8

    Output: Return Main dataframe with predicted external features.

    Params: Only testing path.
    """

    mix_panel_data = mix_panel_features(path)
    df_final = pd.read_excel(path)
    df_final = df_final[
        ["Contact ID", "Email", "Average Pageviews", 'Number of Pageviews', 'Number of Sessions', 'Google ad click id',
         'Recent Conversion', "Lifecycle Stage"]]
    df_final.columns = ["Contact_ID", "email", "Average_Pageviews", 'Number_of_Pageviews', 'Number_of_Sessions',
                        'Google_ad_click_id',
                        'Recent_Conversion', "Lifecycle_Stage"]
    df_final = df_final.join(mix_panel_data)
    df_final[["Recent_Conversion"]] = df_final[["Recent_Conversion"]].replace(np.nan, 0)
    df_final[["Google_ad_click_id"]] = df_final[["Google_ad_click_id"]].replace(np.nan, 0)
    Y = df_final[["Google_ad_click_id"]]
    Q = []
    for i in Y.Google_ad_click_id:
        if (i == 0):
            Q.append(0)
        else:
            Q.append(1)
    df_final[["Google_ad_click_id"]] = Q
    Y = df_final[["Recent_Conversion"]]
    k = []
    for i in Y.Recent_Conversion:
        if (i == "Trial Registration demo.higheredlab.com"):
            k.append(1)
        else:
            k.append(0)
    df_final[["Recent_Conversion"]] = k

    Y = df_final[["Lifecycle_Stage"]]
    Q1 = []
    for i in Y.Lifecycle_Stage:
        if (i == "Customer" or i == 'Subscriber' or i == 'In progress' or i == "Sales qualified lead"):
            Q1.append("Y")
        else:
            Q1.append("N")
    df_final["Lifecycle_Stage"] = Q1
    return df_final


df_final = data_test_joiner(path)

X = df_final[["Number_of_Pageviews", "Number_of_Sessions", "Average_Pageviews", "Google_ad_click_id", 'Recent_Conversion',
     'BBB_START_CLASS_Total', 'VIEW_PAGE_ANALYTICS_Total','Signin_Total', 'Signup_Total', 'BBB_CREATE_CLASS_Total','VIEW_PAGE_UPGRADE_Total']]
X


def data_Contact(X, Y,df_final):

    from datetime import datetime
    now = datetime.now()
    date = now.date()
    time = now.strftime("%H%M%S")

    nameFile = "contact_data_"+str(date)+"_"+str(time)+".csv"
    Data = X.join(Y)
    Contacts = df_final[['Contact_ID', 'email']].join(Data)
    Contacts.to_csv("./Data_For_Prediction/"+nameFile)

Cd = df_final
C = X

import time
lst = [1400]
x = 0
for i in range(len(lst)):
    df_final = Cd.head(lst[x])
#     print(lst[x])
    X = C.head(lst[x])
    Y = Cd[['Lifecycle_Stage']].head(lst[x])
    data_Contact(X, Y, df_final)
    x += 1
    time.sleep(2)
