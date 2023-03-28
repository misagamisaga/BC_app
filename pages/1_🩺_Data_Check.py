import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import auc
# from sklearn.metrics import RocCurveDisplay
# from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import os
import time

if_run = False

time_now = time.asctime()
st.image(r'treeandawn_title.jpg')
st.title(' 🩺 Data Check')
st.text("""
Have data but don't know if there is a connection?
Have a guess but don't know if it's worth doing?
Upload your data here and let the machine learning models answer your questions!
""")

@st.cache_data
def get_data(file):
    data = pd.read_csv(file)
    return data
upload_file = None

upload_file = st.file_uploader(" ", accept_multiple_files=False, label_visibility="collapsed", key=100)
if upload_file is None:
    st.write('## 📤 Upload your `.csv` file above')
    st.markdown('---')
else:
    data = get_data(upload_file)  # 上传文件和读取表格不要放在一起，不然表格未上传时，read_csv读取到的内容为空白，系统会报错
    
    columns_proc = data.columns
    index_01 = []
    index_SS = []
    for col in columns_proc:
        col_now = data[col]
        if len(col_now[col_now==0]) + len(col_now[col_now==1]) == len(col_now):
            # 如果是01变量
            index_01.append(col)
        else:
            index_SS.append(col)
    
    st.markdown('---')
    
    col1, col2 = st.columns([2,3])
    col1.header('Data uploaded!')
    col2.write('''
    View and check your data below
    
    You can modify the contents of the table as you wish
    ''')
    edited_df = st.experimental_data_editor(data, key=101)
    st.markdown('---')
    
    col1, col2 = st.columns([2,3])
    with col1:
        st.header('Setting Params')
        if_SS = st.checkbox('Standardization')
        if_01trans = st.checkbox('transform 0-1 variable into float format')
        st.markdown('''
        
        ''')
        if_run = st.button('Run !', key=102)
    with col2:
        aim = st.selectbox(
            'Which variable would you like to be predicted? (0-1 only)',
            index_01
        )
        st.markdown('')  # make an empty here
        cut_rate = st.slider('testset split rage?', 0.1, 0.5, 0.3)

    st.markdown('---')



if if_run:
    # 变量处理
    index_01.remove(aim)
    data_unprocess = edited_df.loc[:,[aim]]
    # data_toprocess = data.drop(unproc_list, axis=1)
    data_01 = edited_df.loc[:, index_01]
    data_SS = edited_df.loc[:, index_SS]
    if if_01trans:
        data_01ed = 2*(data_01-0.5)
    else:
        data_01ed = data_01
    
    if if_SS:
        column_SS = data_SS.columns
        sccc = StandardScaler()
        data_SSed = sccc.fit_transform(data_SS)  # 得到的data_SS是numpy.array格式
        data_SSed_df = pd.DataFrame(data_SSed, columns=column_SS, index=data.index)
    else:
        data_SSed_df = data_SS
    
    data_as_input = pd.concat([data_unprocess, data_SSed_df, data_01], axis=1)

    df0 = data_as_input.loc[data_as_input[aim]==0,:]
    df1 = data_as_input.loc[data_as_input[aim]!=0,:]
    
    random_st = 0
    data_0_in, data_0_test_out = train_test_split(df0, test_size=cut_rate, random_state=random_st)
    data_1_in, data_1_test_out = train_test_split(df1, test_size=cut_rate, random_state=random_st)
    data_train = pd.concat([data_0_in, data_1_in])
    data_test = pd.concat([data_0_test_out, data_1_test_out])
    X = data_train.drop([aim], axis=1)
    y = data_train[aim]
    X_test = data_test.drop([aim], axis=1)
    y_test = data_test[aim]

    st.title('Machine Learning Progress')
        
    models = [
        "LogisticRegression(max_iter=5000)",
        # "DecisionTreeClassifier()",
        "RandomForestClassifier()",
        "MLPClassifier()",
        "GaussianNB()",
        "SVC(probability=True)",
        # "LGBMClassifier()",
        "XGBClassifier(max_depth=5, learning_rate=0.1, objective='binary:logistic', nthread=-1, scale_pos_weight = len(y[y == 0])/len(y[y == 1]))",
        "KNeighborsClassifier(n_neighbors=5)"
    ]

    model_names = [
        'Logistic',
        # 'Decision Tree',
        'Random Forest',
        'Neural Network',
        'Bayes',
        'SVM',
        # 'LightGBM',
        'XGBoost',
        'KNeighbor',
    ]

    outcome = pd.DataFrame()
    
    L = len(models)
    process_text = 'Processing... ' + '{:.1%}'.format(0/L) + ' (0/' + str(L) + ')'
    bar = st.progress(0.0, text=process_text)

    columns_input = X.columns
    for j in range(0, len(models)):
        classifier = eval(models[j])
        model_name = model_names[j]
        
        classifier.fit(X, y)
        
        # 模型测试结果
        f_y_pred = classifier.predict(X_test)
        f_y_proba = classifier.predict_proba(X_test)[:, 1]
        f_y_true = y_test
        
        # 计算模型指标
        f_acc = accuracy_score(f_y_true, f_y_pred)
        tn, fp, fn, tp = confusion_matrix(f_y_true, f_y_pred).ravel()
        f_auc = roc_auc_score(f_y_true, f_y_proba)
        f_sec = tp / (tp + fn)
        f_scf = tn / (fp + tn)
        f_pcs = tp / (tp + fp)
        f_npv = tn / (fn + tn)

        if f_auc < 0.5:
            f_auc = 1 - f_auc
        
        # 保存模型指标值
        audf = pd.DataFrame((f_acc, f_auc, f_sec, f_scf, f_pcs, f_npv), columns=[model_names[j]])
        outcome = pd.concat([outcome, audf], axis=1)  # 这里是保存了每个模型的指标值的，如果需要可以拿来输出
        
        process_text = 'Processing... ' + '{:.1%}'.format((j+1)/L) + ' (' + str(j+1)+ '/' + str(L) + ')'
        bar.progress((j+1)/L, text=process_text)

        if model_name == 'XGBoost':
            importance = classifier.feature_importances_
    outcome.index = ['ACC', 'AUC', 'SEC', 'SCF', 'PCS', 'NPV']

    st.table(outcome)
    
    st.markdown('---')

    aucs = outcome.loc[['AUC'], :]
    best_auc = aucs.max(axis=1).item()
    best_model = data.stack().idxmax()[1]
    
    if best_auc >= 0.95:
        col1, col2 = st.columns(2)
        with col1:
            st.header('Incredible !')
        with col2:
            st.metric(
                label='Best AUC Outcome', 
                value=best_auc, 
                # delta_color='inverse'
            )

        st.write('''
        The dependent variable you provided almost perfectly predicts the target variable, 
        which means that there is a great overlap of information between the data. 
        We are sure you are very happy to get such information, 
        and we congratulate you here. 
        
        However, please do not be too happy. 
        Out of responsibility, we have to remind you that such high prediction accuracy in nature is very rare, 
        so you need to carefully check your data and the process of data collection: 
        whether there is information related to the target variable mixed in with the existing variables 
        (e.g., an extra copy of the target variable was accidentally made in the data during pre-processing), 
        and whether there might be improper selection or preference in the process of data collection. 
        And, since this is a primary screening, our data pre-processing is done throughout the data, 
        not just in the test set, which can have an impact on the results.
        Once all of the above have been checked, 
        you can breathe a sigh of relief and open a bottle of champagne to celebrate your excellent results. 
        
        While the above is anachronistic, it is clear that scientific rigor is more important. 
        We sincerely hope that you will achieve outstanding results based on scientifically rigorous experiments. 
        But finally, at least for now, we would like to congratulate you on your good results. Congratulations!
        ''')
    elif best_auc >= 0.90:
        col1, col2 = st.columns(2)
        with col1:
            st.header('Congratulations !')
        with col2:
            st.metric(
                label='Best AUC Outcome', 
                value=best_auc, 
                # delta_color='inverse'
            )
        st.write('''
        The dependent variable predicts the target variable well, 
        which means that there is a great overlap of information between the data. 
        We are sure you are very happy to get such information, 
        and we congratulate you here. 
        
        However, please do not be too happy. 
        Out of responsibility, we have to remind you that such high prediction accuracy in nature is very rare, 
        so you need to carefully check your data and the process of data collection: 
        whether there is information related to the target variable mixed in with the existing variables 
        (e.g., an extra copy of the target variable was accidentally made in the data during pre-processing), 
        and whether there might be improper selection or preference in the process of data collection. 
        And, since this is a primary screening, our data pre-processing is done throughout the data, 
        not just in the test set, which can have an impact on the results.
        Once all of the above have been checked, 
        you can breathe a sigh of relief and open a bottle of champagne to celebrate your excellent results. 
        
        While the above is anachronistic, it is clear that scientific rigor is more important. 
        We sincerely hope that you will achieve outstanding results based on scientifically rigorous experiments. 
        But finally, at least for now, we would like to congratulate you on your good results. Congratulations!
        ''')

    elif best_auc >= 0.80:
        col1, col2 = st.columns(2)
        with col1:
            st.header('A Good Prediction')
        with col2:
            st.metric(
                label='Best AUC Outcome', 
                value=best_auc, 
                # delta_color='inverse'
            )
        st.write('''
        We have found predictive efficacy between the target variable and other variables you specified. 
        We are proud to bring recognition to your work with the expected results. 
        With this result, your study has been initially successful. 
        If this was part of the preliminary work of your study, then the potential of this result would be even greater. 
        
        Such a result is not particularly high, but it is exactly the result that most research will encounter. 
        Such a situation is very common in nature: partially predictable, but not decisively so. 
        So even if the result is not particularly high, you do not have to be discouraged, 
        it may not be the final result, but it is a good start. 
        And, since this is a primary screening, our data pre-processing is done throughout the data, 
        not just in the test set, which can have an impact on the results.
        We know that you will want to build on this foundation for research purposes, so, here are the tips:
        
        - Remove variables that are not so important. They are likely to introduce unnecessary noise into the data and affect the prediction results
        - Consider including more variables. Existing univariate screening methods have limitations, and the process may filter out variables that are actually useful
        - Consider a larger amount of data. Adding more samples can help a lot in training a better model
        - Pre-process the data. The distribution of data in your data can be difficult to handle, which can hinder the training of machine learning models. The right data pre-processing efforts can greatly improve the model results

        These are some common ways to boost. With such a good foundation, we sincerely wish you success in the end.
        ''')

    elif best_auc >= 0.75:
        col1, col2 = st.columns(2)
        with col1:
            st.header("They're Relevant")
        with col2:
            st.metric(
                label='Best AUC Outcome', 
                value=best_auc, 
                # delta_color='inverse'
            )
        st.write('''
        We have found predictive efficacy between the target variable and other variables you specified. 
        We are proud to bring recognition to your work with the expected results. 
        With this result, your study has been initially successful. 
        If this was part of the preliminary work of your study, then the potential of this result would be even greater. 
        
        Such a result is not particularly high, but it is exactly the result that most research will encounter. 
        Such a situation is very common in nature: partially predictable, but not decisively so. 
        So even if the result is not particularly high, you do not have to be discouraged, 
        it may not be the final result, but it is a good start. 
        And, since this is a primary screening, our data pre-processing is done throughout the data, 
        not just in the test set, which can have an impact on the results.
        We know that you will want to build on this foundation for research purposes, so, here are the tips:
        
        - Remove variables that are not so important. They are likely to introduce unnecessary noise into the data and affect the prediction results
        - Consider including more variables. Existing univariate screening methods have limitations, and the process may filter out variables that are actually useful
        - Consider a larger amount of data. Adding more samples can help a lot in training a better model
        - Pre-process the data. The distribution of data in your data can be difficult to handle, which can hinder the training of machine learning models. The right data pre-processing efforts can greatly improve the model results

        These are some common ways to boost. With such a good foundation, we sincerely wish you success in the end.
        ''')

    else:
        col1, col2 = st.columns(2)
        with col1:
            st.header("Unfortunately")
        with col2:
            st.metric(
                label='Best AUC Outcome', 
                value=best_auc, 
                # delta_color='inverse'
            )
        st.write('''
        Your data is not a good source of predictions for your target variable. 
        
        This may be because the other variables you provided do not provide enough information 
        about the target variable you want to predict,
        or because the data were collected with non-negligible errors or too much invalid noise. 
        You may need to revisit the data to check for gaps, duplicates and anomalies, 
        and to consider the factors in the data that may be affecting the predictions: 
        the charts we give below may help you. 
        And, since this is a primary screening, our data pre-processing is done throughout the data, 
        not just in the test set, which can have an impact on the results.
        If the problems in the data are too difficult to eliminate, 
        you may need to re-collect the data and try to avoid unnecessary errors and selective preferences. 
        
        Finally, we would like to remind you that not just any data is related to each other, 
        and not just any idea can be successful. 
        Some data just don't correlate with each other, 
        so no matter how powerful the analysis method is, it won't help. 
        Even if you end up with a usable result through some fancy trick, 
        its scientific validity is still up for debate. 
        If you have been stuck with invalid results for a long time, 
        it is prudent to look at the situation and sometimes it is wise to abandon the current thinking. 
        
        We are sorry for such results and as researchers. we can empathize with your feelings and situation. 
        We wish you the best of luck in getting over this hurdle.
        ''')
    
    st.markdown('---')
    st.header('Supplementary')

    # importance
    imp_df = pd.DataFrame(importance)
    imp_df.index = columns_input
    imp_df.columns = ['Feature Importance']
    list_imp = []
    for i in range(len(importance)):
        list_imp.append([importance[i], columns_input[i]])
    
    list_imp = sorted(list_imp, key=lambda s: s[0])
    new_imp_list = [num[0] for num in list_imp]
    new_name_list = [num[1] for num in list_imp]

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set()

    plt.figure()
    st.table(imp_df)
    sns.barplot(
        x=new_imp_list, 
        y=new_name_list, 
        orient='h', 
    ).set_title('Feature Importance')
    st.pyplot()
    
    plt.figure()
    sns.boxplot(
        data=data_SSed_df, 
        orient="h"
    )
    st.pyplot()

    plt.figure()
    sns.clustermap(
        edited_df.corr(), 
        center=0, 
        cmap="vlag", 
        linewidths=.75
    )
    st.pyplot()

    st.markdown('---')
