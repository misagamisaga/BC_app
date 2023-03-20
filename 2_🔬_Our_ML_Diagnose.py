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
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import auc
# from sklearn.metrics import RocCurveDisplay
# from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import os
import time
from radiomics import featureextractor
import SimpleITK as sitk

# def read_uploaded_file(uploaded_file):
#     try:
#         # Read the uploaded file as bytes
#         bytes_data = uploaded_file.read()
#         # Convert the bytes data to SimpleITK image
#         sitk_image = sitk.ReadImage(sitk.GetArrayFromImage(sitk.ImageFileReader(bytes_data)))
#         # Extract features using pyradiomics
#         feature_extractor = pyradiomics.featureextractor.RadiomicsFeatureExtractor()
#         features = feature_extractor.execute(sitk_image)
#         return features
#     except Exception as e:
#         st.write("Error:", e)

# 检查是否含有目标列
# 输入？要按照训练集进行预处理变成我们要的格式
# 输出预测的概率（metric）
# 输出预测的概率为柱状图

if_run = False

time_now = time.asctime()
st.image(r'road_title.jpg')
st.title(' 🔬 Our ML Diagnose')
st.text("""
Want to get a better outcome as a clinician? 
Have patient test results but are still hesitant to confirm the diagnosis? 
Upload data and let scientific research help you make precise decisions!
""")

st.header('Please upload your `nii` files here')
col1, col2 = st.columns(2)
with col1:
    file_A= st.file_uploader("upload the Arterial clinic `.nii` file here", type="nii", key=210)
    mask_A= st.file_uploader("upload the Arterial mask `.nii` file here", type="nii", key=220)
with col2:
    file_V= st.file_uploader("upload the Venous clinic `.nii` file here", type="nii", key=230)
    mask_V= st.file_uploader("upload the Venous mask `.nii` file here", type="nii", key=240)
if_run = st.button('Process', key=201)

if if_run:
    st.header('Processing')
    if (file_A is not None) and (mask_A is not None):
        file_A_path = os.path.join(os.getcwd(), "file_A.nii")
        with open(file_A_path, "wb") as f:
            f.write(file_A.read())
        mask_A_path = os.path.join(os.getcwd(), "mask_A.nii")
        with open(mask_A_path, "wb") as f:
            f.write(mask_A.read())

    if (file_V is not None) and (mask_V is not None):
        file_V_path = os.path.join(os.getcwd(), "file_V.nii")
        with open(file_V_path, "wb") as f:
            f.write(file_V.read())
        mask_V_path = os.path.join(os.getcwd(), "mask_V.nii")
        with open(mask_V_path, "wb") as f:
            f.write(mask_V.read())
    
    etr = featureextractor.RadiomicsFeatureExtractor()
    try:
        fv_V = etr.execute(file_V_path, mask_V_path, label=2)
        df_add_V = pd.DataFrame([fv_V])
    except:
        try:
            fv_V = etr.execute(file_V_path, mask_V_path)
            df_add_V = pd.DataFrame([fv_V])
        except Exception as e:
            st.text('Sorry, We got an error')
            st.text(e.__class__.__name__)
            st.text(e)
    
    try:
        fv_A = etr.execute(file_A_path, mask_A_path, label=2)
        df_add_A = pd.DataFrame([fv_A])
    except:
        try:
            fv_A = etr.execute(file_A_path, mask_A_path)
            df_add_A = pd.DataFrame([fv_A])
        except Exception as e:
            st.text('Sorry, We got an error')
            st.text(e.__class__.__name__)
            st.text(e)
    
    # 列名更新，区别一下A和V
    col_A = list(df_add_A.columns)
    col_V = list(df_add_V.columns)
    for i in range(len(col_A)):
        col_A[i] = 'A_' + col_A[i]
        col_V[i] = 'V_' + col_V[i]
    df_add_A.columns = col_A
    df_add_V.columns = col_V

    st.dataframe(df_add_A)
    st.dataframe(df_add_V)

    data_all = pd.concat([df_add_A, df_add_V], axis=1)
    st.dataframe(data_all)
            


    

    

















    # st.text(time.asctime())
    # st.text('...OK')

    # st.text('Preparing feature extractor...')
    # bar = st.progress(0.1, text='Preparing feature extractor...')
    # etr = featureextractor.RadiomicsFeatureExtractor()
    # st.text(time.asctime())
    # st.text('...OK')

    # st.text('Get Arterial features...')
    # bar.progress(0.2, text='Generating Arterial Features...')
    # try:
    #     st.text('Trying...(1/2)')
    #     fv_A = etr.execute(image_A, label_A, label=2)
    # except:
    #     try:
    #         st.text('Failed.Retrying...(2/2)')
    #         fv_A = etr.execute(image_A, label_A)
    #     except Exception as e:
    #         st.text('Sorry, we got an Error here')
    #         st.text(e.__class__.__name__)
    #         st.text(e)
    # data_A = pd.DataFrame([fv_A])
    # st.text(time.asctime())
    # st.text('...OK')

    # st.text('Get Venous features...')
    # bar.progress(0.3, text='Generating Venous Features...')
    # try:
    #     st.text('Trying...(1/2)')
    #     fv_V = etr.execute(image_V, label_V, label=2)
    # except:
    #     try:
    #         st.text('Failed.Retrying...(2/2)')
    #         fv_V = etr.execute(image_V, label_V)
    #     except Exception as e:
    #         st.text('Sorry, we got an Error here')
    #         st.text(e.__class__.__name__)
    #         st.text(e)
    # data_V = pd.DataFrame([fv_V])
    # st.text(time.asctime())
    # st.text('...OK')

    # st.dataframe(data_A)
    # st.dataframe(data_V)





