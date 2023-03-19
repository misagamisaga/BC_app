import streamlit as st
import os



# title
st.image(r'images\title_moun.jpg')
st.title('🎉 Welcome to Bio-Clic Lab!')


# consice introduction
col1, col2 = st.columns([2,1])

with col1:
    st.write('''
    Bio-Clic Lab is a cutting-edge research laboratory founded by Bo Chen in 2020. 
    We focuses on molecular biology and genetics, and aims to uncover the secrets of life and disease. 
    We employs various advanced techniques, such as GWAS analysis, Mendelian Randomisation, Machine Learning, and Single Cell Analysis, to study the structure and function of biomolecules and their interactions. 
    The lab has published numerous high-quality papers with a total impact factor of over 100 in prestigious journals. 
    The lab is also committed to fostering the next generation of scientists through education and outreach activities. 
    Bio-Clic Lab is one of the leading biology labs in the world, and strives to make significant contributions to science and society.
    ''')

with col2:
    st.image(r'images\lightitem.jpg')


st.header('Our Work')
tab1, tab2, tab3, tab4 = st.tabs(["🩺 Data Check", "🔬 Our ML Diagnose", "🔧 Plot Tools", "🧷 Basic Knowledge"])

with tab1:
    st.subheader("🩺 Data Check")
    st.text("""
    Have data but don't know if there is a connection?
    Have a guess but don't know if it's worth doing?
    Upload your data here and let the machine learning models answer your questions!
    """)

with tab2:
    st.subheader("🔬 Our ML Diagnose")
    st.text("""
    Want to get a better outcome as a clinician? 
    Have patient test results but are still hesitant to confirm the diagnosis? 
    Enter data into our website and let scientific research help you make precise decisions!
    """)

with tab3:
    st.subheader("🔧 Plot Tools")
    st.text('''
    Need to draw a picture but only have data? 
    Want to visualize but struggle with programming? 
    Upload your data and let the site do it automatically!
    ''')

with tab4:
    st.subheader("🧷 Basic Knowledge")
    st.text('''
    Think the site works well and want to know how it works? 
    As a clinician, want to use our results for your research? 
    Read this page to gain a better understanding!
    ''')

# contact us
st.header('Contact us')
with st.expander('⇲'):
    st.text('Github: xxxxxxxx')

st.header('FAQs')
with st.expander('⇲'):
    st.write('''
    ## 
    您的问题已收到，我们将在60个工作日内给您回复，如果没有回复就去别的部门问问。
    学生出了问题就是学生有问题和我们没关系，但是学生的荣誉都是我们教务处指导有方。
    我知道你很急但是你先别急，但是，温医大是一所历史悠久的，拥有强劲教务处的医学专科（大学），一直为学生提供最为科学高效的课程设置与最强劲有力的科研支持。
    在我们提供的优渥科研导向环境中，学生们综合素养极强，虽然某些专业执医通过率已经不足50%，但是果然本校在挑战杯与互联网+比赛中屡获佳绩。

    ——温医教务处
    ''')

st.header('References')
with st.expander('⇲'):
    st.write('''
    - Sklearn: [Scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/)
    - Pandas: [Pandas: fast, powerful, flexible and easy to use open source data analysis](https://pandas.pydata.org/)
    - Numpy: [Numpy: The fundamental package for scientific computing with Python](https://numpy.org/)
    - Scipy: [Scipy: Fundamental algorithms for scientific computing in Python](https://scipy.org/)
    - Streamlit: [Streamlit: The fastest way to build and share data apps](https://streamlit.io/)
    ''')