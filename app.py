# sentiment_analyzer/main.py

import streamlit as st

def main():
    st.title("Sentiment Analysis Dashboard")
    st.sidebar.title("Navigation")
    
    app_mode = st.sidebar.selectbox("Choose Analysis Type:", [
        "Home",
        "Text Analysis",
        "Image Analysis",
        "Video Analysis",
        "Voice Analysis"
    ])

    if app_mode == "Home":  
        #import home
        st.markdown("Welcome to the Sentiment Analysis Dashboard! 🚀")
        st.markdown("Select an analysis type from the sidebar to get started.")
        #home.run()
    
    if app_mode == "Text Analysis":
        import text_analysis
        text_analysis.run()
    elif app_mode == "Image Analysis":
        import image_analysis
        image_analysis.run()
    elif app_mode == "Video Analysis":
        import video_analysis
        video_analysis.run()
    elif app_mode == "Voice Analysis":
        import voice_analysis
        voice_analysis.run()
    
if __name__ == "__main__":
    main()

# The rest of the files will be modularized accordingly.
