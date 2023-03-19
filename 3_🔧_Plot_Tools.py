import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.image(r'images\title_cyber.jpg')
st.title(' 🔧 Plot Tools')
st.text('''
Need to draw a picture but only have data? 
Want to visualize but struggle with programming? 
Upload your data and let the site do it automatically!
''')

st.markdown('---')


@st.cache_data
def get_data(file):
    data = pd.read_csv(file)
    return data

st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set()

# --- Box Plot ---
st.header('Box Plot')
if_plot1 = False
upload_file1 = st.file_uploader("", accept_multiple_files=False, key=1)
if upload_file1 is None:
    st.write('## 📤 Upload your `.csv` file above')
    st.markdown('---')
else:
    data1 = get_data(upload_file1)
    edited_df1 = st.experimental_data_editor(data1, key=11)
    if_plot1 = st.button('Plot', key=12)
if if_plot1:
    plt.figure()
    sns.boxplot(
        data=edited_df1, 
        orient="h"
    )
    st.pyplot()
st.markdown('---')

# --- Line Plot ---
st.header('Line Plot')
if_plot2 = False
upload_file2 = st.file_uploader("", accept_multiple_files=False, key=2)
if upload_file2 is None:
    st.write('## 📤 Upload your `.csv` file above')
    st.markdown('---')
else:
    data2 = get_data(upload_file2)
    edited_df2 = st.experimental_data_editor(data2, key=21)
    col1, col2 = st.columns(2)
    aim1 = col1.selectbox(
        'X-axis',
        edited_df2.columns
    )
    aim2 = col2.selectbox(
        'Y-axis',
        edited_df2.columns
    )
    if_plot2 = st.button('Plot', key=22)
if if_plot2:
    plt.figure()
    sns.jointplot(
        x=aim1, y=aim2, data=edited_df2,
        kind="reg", color="m")
    st.pyplot()
st.markdown('---')

# --- Corr Heatmap ---
st.header('Corr Heatmap')
if_plot3 = False
upload_file3 = st.file_uploader("", accept_multiple_files=False, key=3)
if upload_file3 is None:
    st.write('## 📤 Upload your `.csv` file above')
    st.markdown('---')
else:
    data3 = get_data(upload_file3)
    edited_df3 = st.experimental_data_editor(data3, key=31)
    if_plot3 = st.button('Plot', key=32)
if if_plot3:
    corr = edited_df3.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure()
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap=cmap, 
        center=0,
        square=True, 
        linewidths=.75, 
        cbar_kws={"shrink": .5}
        )
    st.pyplot()
st.markdown('---')

# --- Cluster Map ---
st.header('Cluster Plot')
if_plot4 = False
upload_file4 = st.file_uploader("", accept_multiple_files=False, key=4)
if upload_file4 is None:
    st.write('## 📤 Upload your `.csv` file above')
    st.markdown('---')
else:
    data4 = get_data(upload_file4)
    edited_df4 = st.experimental_data_editor(data4, key=41)
    if_plot4 = st.button('Plot', key=42)
if if_plot4:
    plt.figure()
    sns.clustermap(
        edited_df4.corr(), 
        center=0, 
        cmap="vlag", 
        linewidths=.75
    )
    st.pyplot()
st.markdown('---')

