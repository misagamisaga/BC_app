import streamlit as st
import os



# title
st.image(r'title_moun.jpg')
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
    st.image(r'lightitem.jpg')


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
    Any responses received will be displayed here. We are looking forward to your use and feedback😊
    ''')

st.header('References')
with st.expander('⇲'):
    st.write('''
    - Sklearn: [Scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/)
    - Pandas: [Pandas: fast, powerful, flexible and easy to use open source data analysis](https://pandas.pydata.org/)
    - Numpy: [Numpy: The fundamental package for scientific computing with Python](https://numpy.org/)
    - Scipy: [Scipy: Fundamental algorithms for scientific computing in Python](https://scipy.org/)
    - Streamlit: [Streamlit: The fastest way to build and share data apps](https://streamlit.io/)
    - Wikipedia: [Wikipedia: The free encyclopedia that anyone can edit](https://en.wikipedia.org/wiki/Machine_learning)
    ''')