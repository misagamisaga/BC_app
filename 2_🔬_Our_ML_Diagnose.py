import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import time
from radiomics import featureextractor

# 检查是否含有目标列
# 输入？要按照训练集进行预处理变成我们要的格式
# 输出预测的概率（metric）
# 输出预测的概率为柱状图


def SS_both(df_ref, df, li_unp=[]):
    data_ref_2SS = df_ref.drop(['Group']+li_unp, axis=1)
    aim_df_ref = df_ref.loc[:,['Group']+li_unp]

    sccc = StandardScaler()
    sccc.fit(data_ref_2SS)

    df_ref_SSed = sccc.transform(data_ref_2SS)  # 得到的data_SS是numpy.array格式
    df_ref_SSed2 = pd.DataFrame(df_ref_SSed, columns=data_ref_2SS.columns, index=data_ref_2SS.index)
    
    data_ref_SSed = pd.concat([aim_df_ref, df_ref_SSed2], axis=1)

    df_2SS = df.drop(li_unp, axis=1)
    aim_df = df.loc[:,li_unp] 

    df_SSed = sccc.transform(df_2SS)  # 得到的data_SS是numpy.array格式
    df_SSed2 = pd.DataFrame(df_SSed, columns=df_2SS.columns, index=df_2SS.index)
    data_SSed = pd.concat([aim_df, df_SSed2], axis=1)

    X_SS = data_ref_SSed.drop(['Group'], axis=1)
    y_SS = data_ref_SSed['Group']

    return X_SS, y_SS, data_SSed

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

if_run = False
if_run_continue = False

time_now = time.asctime()
st.image(r'road_title.jpg')
st.title(' 🔬 Our ML Diagnose')
st.text("""
Want to get a better outcome as a clinician? 
Have patient test results but are still hesitant to confirm the diagnosis? 
Upload data and let scientific research help you make precise decisions!
""")

st.markdown('---')
st.header('Please upload your `nii` files here')
col1, col2 = st.columns(2)
with col1:
    file_A= st.file_uploader("upload the Arterial clinic `.nii` file here", type="nii", key=210)
    mask_A= st.file_uploader("upload the Arterial mask `.nii` file here", type="nii", key=220)
with col2:
    file_V= st.file_uploader("upload the Venous clinic `.nii` file here", type="nii", key=230)
    mask_V= st.file_uploader("upload the Venous mask `.nii` file here", type="nii", key=240)
st.markdown('')
col11, col12, col13 = st.columns(3)
num1 = int(col11.selectbox('Histology', (0,1)))
num2 = int(col12.selectbox('TNM_Stge', (0,1)))
num3 = col13.number_input('CA199')
st.markdown(' ')
st.markdown(' ')
if_run = st.button('Process', key=201)
st.markdown('---')

if if_run:
    st.header('Program Run Log')
    st.subheader('Processing `.nii` files')
    proc_bar = st.progress(0, text='Read nii files...')
    if (file_A is not None) and (mask_A is not None):
        file_A_path = os.path.join(os.getcwd(), "file_A.nii")
        with open(file_A_path, "wb") as f:
            f.write(file_A.read())
        mask_A_path = os.path.join(os.getcwd(), "mask_A.nii")
        with open(mask_A_path, "wb") as f:
            f.write(mask_A.read())
    
    proc_bar.progress(0.1, text='Read nii files...')
    if (file_V is not None) and (mask_V is not None):
        file_V_path = os.path.join(os.getcwd(), "file_V.nii")
        with open(file_V_path, "wb") as f:
            f.write(file_V.read())
        mask_V_path = os.path.join(os.getcwd(), "mask_V.nii")
        with open(mask_V_path, "wb") as f:
            f.write(mask_V.read())
    
    proc_bar.progress(0.2, text='Preparing Radiomic Feature Extractor...')
    etr = featureextractor.RadiomicsFeatureExtractor()
    
    proc_bar.progress(0.35, text='Radiomic Feature Extracting...')
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
    
    proc_bar.progress(0.55, text='Radiomic Feature Extracting...')
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
    proc_bar.progress(0.75, text='Generating Radiomic Feature DataFrame...')
    col_A = list(df_add_A.columns)
    col_V = list(df_add_V.columns)
    for i in range(len(col_A)):
        col_A[i] = 'A_' + col_A[i]
        col_V[i] = 'V_' + col_V[i]
    df_add_A.columns = col_A
    df_add_V.columns = col_V

    proc_bar.progress(0.80, text='Show Radiomic Feature DataFrame...')
    data_all = pd.concat([df_add_A, df_add_V], axis=1)
    proc_bar.progress(0.85, text='Get Radiomic Logictic Score...')
    data_csv = convert_df(data_all)
    st.markdown(' ')
    st.subheader('Your Radiomic Dataframe')
    st.dataframe(data_all)
    st.download_button(
        label = 'Download your radiomic dataframe', 
        data = data_csv, 
        file_name = 'Radiomic_Dataframe.csv'
    )

    
    name_file_ref = 'yuanshi.csv'
    cols_need_ori = ['V_original_shape_SurfaceArea', 'V_original_glcm_Idm', 'A_original_shape_Maximum2DDiameterColumn', 'V_original_ngtdm_Busyness', 'V_original_shape_Maximum2DDiameterColumn', 'V_original_glrlm_GrayLevelNonUniformity', 'V_original_shape_VoxelVolume', 'V_original_glszm_LargeAreaEmphasis', 'V_original_firstorder_90Percentile', 'V_original_shape_MeshVolume', 'V_original_gldm_GrayLevelNonUniformity', 'V_original_shape_LeastAxisLength', 'V_original_glszm_ZoneVariance', 'V_original_gldm_DependenceNonUniformity', 'V_original_gldm_SmallDependenceEmphasis', 'V_original_shape_MajorAxisLength', 'V_original_glrlm_RunLengthNonUniformity', 'V_original_glrlm_LongRunEmphasis', 'V_original_shape_MinorAxisLength', 'V_original_firstorder_RootMeanSquared', 'V_original_shape_Maximum2DDiameterRow', 'V_original_glrlm_RunVariance', 'V_original_glszm_ZonePercentage', 'V_original_shape_SurfaceVolumeRatio', 'V_original_firstorder_Mean', 'V_original_glszm_LargeAreaLowGrayLevelEmphasis', 'V_original_glszm_GrayLevelNonUniformity', 'V_original_firstorder_Median', 'V_original_glszm_SizeZoneNonUniformity', 'V_original_glcm_DifferenceAverage', 'V_original_ngtdm_Coarseness', 'V_original_shape_Maximum2DDiameterSlice', 'V_original_glrlm_ShortRunEmphasis', 'V_original_firstorder_TotalEnergy', 'V_original_glcm_Id', 'V_original_firstorder_Energy', 'V_original_glrlm_RunPercentage', 'V_original_glrlm_RunLengthNonUniformityNormalized', 'V_original_firstorder_10Percentile', 'V_original_gldm_DependenceNonUniformityNormalized']
    cols_need = cols_need_ori[:35]

    data = data_all.loc[:,cols_need]  # read in it
    data_ref = pd.read_csv(name_file_ref).loc[:,cols_need+['Group']]

    X_SS, y_SS, data_SSed = SS_both(data_ref, data)

    classifier = LogisticRegression(
        penalty = 'l2', 
        C =994,
        max_iter=5000
    )

    classifier.fit(X_SS, y_SS)

    logis_out = classifier.predict_proba(data_SSed)[:, 1].item()
    proc_bar.progress(0.95, text='Show Radiomic Logictic Score...')
    st.markdown(' ')
    st.subheader('Your Radiomic Logistic Score')
    st.metric(
        label=" ", 
        value = logis_out, 
        label_visibility='collapsed'
    )
    proc_bar.progress(1.00, text='Done')
    
    st.markdown(' ')
    st.subheader('Machine Learning Dignosis')
    diag_bar = st.progress(0, text='Loading Supplementary Data...')
    
    list_input = [num1, num2, num3, logis_out]
    data_clinic_ref = pd.read_csv('logis_score.csv').drop(['ID'], axis=1)
    input_data = pd.DataFrame([list_input], columns=[
                            'Histology', 'TNM_Stge', 'CA199', 'logis_score'])
    
    diag_bar.progress(0.1, text='Preprocessing...')
    X_clinic, y_clinic, input_data_SSed = SS_both(data_clinic_ref, input_data, li_unp=['Histology', 'TNM_Stge'])

    models = [
        "LogisticRegression(penalty = 'l2', C = 27.0, max_iter=5000)", 
        'RandomForestClassifier(n_estimators=52, min_samples_split=15, max_features=0.999, max_depth=5, random_state=2)', 
        "MLPClassifier(hidden_layer_sizes=tuple([330, 240]),solver='lbfgs',alpha = 12.589254117941675, max_iter=65, verbose=True)", 
        'GaussianNB()', 
        'SVC(C=0.004086256180706494, gamma=0.00767147013532765, random_state=2, probability=True)', 
        "XGBClassifier(objective='binary:logistic', learning_rate=0.3099376425986338, gamma=3.760043767458776, max_depth=2, seed=0, nthread=-1, scale_pos_weight = len(y_clinic[y_clinic == 0])/len(y_clinic[y_clinic == 1]))"
    ]

    model_names = [
        'LogisticRegression', 
        'RandomForest', 
        'Neural Network',
        'GaussianNB',
        'Support Vector Machine',
        'XGBoost',
    ]

    nn = 0.1
    th_list = []
    proba_list = []
    for ii in range(len(models)):
        model_name = model_names[ii]
        nn += 0.1
        text_bar = 'Running ' + model_name + ' Model...'
        diag_bar.progress(nn, text=text_bar)
        model = eval(models[ii])
        model.fit(X_clinic, y_clinic)
        probs = model.predict_proba(X_clinic)[:, 1]

        best_threshold = 0.5
        best_score = 0
        for threshold in range(1, 100):
            threshold = threshold / 100
            preds = (probs > threshold).astype(int)
            score = f1_score(y_clinic, preds)
            if score > best_score:
                best_threshold = threshold
                best_score = score

        out_proba = model.predict_proba(input_data_SSed)[:, 1].item()
        th_list.append(best_threshold)
        proba_list.append(out_proba)

    out = 0
    for i in range(6):
        if proba_list[i] > th_list[i]:
            out += 1
    if out >= 3:
        diag = True
    else:
        diag = False
    
    st.markdown('---')
    diag_bar.progress(0.8, text='Generating diagnostic report')
    if diag:
        st.write('''
        # Relapse: True
        **We regret to inform you that your illness may recur.**
        ''')
    else:
        st.write('''
        # Relapse: False
        **We are pleased to inform you that your disease is unlikely to recur.**
        ''')
    
    st.markdown('''
    Here are the specific recurrence probabilities 
    and probability differences relative to the model's recommended cut-off values 
    provided by each model: 
    ''')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label = 'LogisticRegression',
            value = '{:.2%}'.format(proba_list[0]), 
            delta = '{:.2%}'.format(proba_list[0] - th_list[0]), 
            delta_color="inverse"
        )
        st.metric(
            label = 'RandomForest',
            value = '{:.2%}'.format(proba_list[1]), 
            delta = '{:.2%}'.format(proba_list[1] - th_list[1]), 
            delta_color="inverse"
        )
    with col2:
        st.metric(
            label = 'Neural Network',
            value = '{:.2%}'.format(proba_list[2]), 
            delta = '{:.2%}'.format(proba_list[2] - th_list[2]), 
            delta_color="inverse"
        )
        st.metric(
            label = 'GaussianNB',
            value = '{:.2%}'.format(proba_list[3]), 
            delta = '{:.2%}'.format(proba_list[3] - th_list[3]), 
            delta_color="inverse"
        )
    with col3:
        st.metric(
            label = 'Support Vector Machine',
            value = '{:.2%}'.format(proba_list[4]), 
            delta = '{:.2%}'.format(proba_list[4] - th_list[4]), 
            delta_color="inverse"
        )
        st.metric(
            label = 'XGBoost',
            value = '{:.2%}'.format(proba_list[5]), 
            delta = '{:.2%}'.format(proba_list[5] - th_list[5]), 
            delta_color="inverse"
        )
    
    if diag:
        st.write("""
        We are sorry to receive such a result.
        With such a result, further medical diagnosis and preparation for treatment is almost always necessary 
        and we wish the patient a better outcome and as long a life as possible. 
        However, our website is still not 100% accurate and it can be wrong. 
        Therefore, even if our model gives a bad diagnosis, 
        there is still the possibility that the patient will have a good prognosis with less medical expenditure.

        Finally, as we are happy to help you with your diagnosis, 
        please note that the diagnoses on this website are for information purposes only 
        and the final diagnosis needs to be supported by more medical evidence. 
        We are continuing to improve our models and algorithms, 
        so please feel free to contact us and collaborate if you wish.
        """)
    else:
        st.write("""
        We would like to congratulate you on your result. 
        However, we must also inform you of the potential risks. 
        It is important to remind you that even if the machine learning model gives a low probability, 
        it is wise to seek additional medical evidence to confirm the diagnosis, just to be on the safe side. 
        In fact, in medical practice, if the cut-off value given by machine learning is 0.5, 
        then we would normally ask for more evidence above a probability of 0.1. 
        This is because an additional test is at most costly, 
        but in the event of a missed diagnosis, the patient could lose his or her life.

        Finally, as we are happy to help you with your diagnosis, 
        please note that the diagnoses on this website are for information purposes only 
        and the final diagnosis needs to be supported by more medical evidence. 
        We are continuing to improve our models and algorithms 
        and welcome all forms of contact and collaboration.
        """)
    diag_bar.progress(1.0, text='Done')
    st.markdown('---')
