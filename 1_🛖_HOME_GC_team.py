import streamlit as st
import os



# title
st.image(r'title_moun.jpg')
st.title('🎉 Welcome to Gang Chen Lab!')


# consice introduction
col1, col2 = st.columns([2,1])

with col1:
    st.write('''
    Gang Chen Lab is a cutting-edge research laboratory in Whenzhou Medical University. 
    We have long focused on liver and gallbladder science. Discovering the mechanisms behind diseases. Providing best treatment for patients.
    The research areas of the mentor team members are related to basic research in hepatobiliary surgery, gastrointestinal malignancies, 
    transplantation immunology, experimental surgery, experimental diagnostics, treatment of metastatic colorectal cancer, and the role of environmental and genetic interactions on tumorigenesis and development. He has also worked in the field of basic and translational medicine for liver cancer. He has made a series of research achievements in the field of basic and translational medicine of liver cancer: 1. Basic research 1. In basic research, we have demonstrated the important role of SUL F2-TGF β 1-SMAD-POSTN signaling pathway in the process of hepatocellular carcinoma angiogenesis, and We have clarified the molecular mechanism of the nesting of hepatocellular carcinoma neovascularization by endothelial progenitor cells of bone marrow origin, and clarified the molecular mechanism of the long-chain chain angiogenesis.
    Gang Chen Lab is one of the leading liver and gallbladder Lab in the world, and strives to make significant contributions to science and society.
    ''')

with col2:
    st.image(r'lightitem.jpg')


st.header('Our Work')
tab1, tab2, tab3, tab4 = st.tabs(["🩺 Data Check", "🔬 Our ML Diagnose", "🔧 Plot Tools", "🧷 Basic Knowledge"])

with tab1:
    st.subheader("🔬 Our ML Diagnose")
    st.text("""
    Want to get a better outcome as a clinician? 
    Have patient test results but are still hesitant to confirm the diagnosis? 
    Enter data into our website and let scientific research help you make precise decisions!
    """)

with tab2:
    st.subheader("🩺 Data Check")
    st.text("""
    Have data but don't know if there is a connection?
    Have a guess but don't know if it's worth doing?
    Upload your data here and let the machine learning models answer your questions!
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
    st.write('''
    This Web APP is currently being tested internally. Please concat Bo Chen in our lab if you are interested.
    ''')

st.header('FAQs')
with st.expander('⇲'):
    st.write('''
    This Web APP is currently being tested internally and has not been officially launched to any community members or organisations.

    Please click [here](https://github.com/misagamisaga/BC_app/issues) to give us feedback on any bugs you've encountered, features you'd like to see added, or even any text or language comments.
    Any responses received will be displayed here. We are looking forward to your use and feedback😊.
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