import streamlit as st
from text_to_speech import * 


st.header("Demo Text to speech - HCMUS K30")
st.subheader("Phan Minh Toan 20C11057")
text_input = st.text_area("Input Text", "Hi, my name is Minh Toan, welcome to my demo of text to speech")
option_lang = st.selectbox("Select Language", ("English", "Vietnamese not available"))
if option_lang == "Vietnamese":
    speaker_type = st.selectbox("Speaker Type", ("Female", "Male"))
    if speaker_type == "Female":
        speaker = 0
    else:
        speaker = 1


if st.button("Text to speech"):
    with st.spinner(text="In progress..."):
        if option_lang == "Vietnamese not available":
            pass
        
        elif option_lang == "English":
            text_to_speech(text_input)
           
        audio_file = open('result/audio.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
        st.success("Done.")


