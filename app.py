import streamlit as st
from text_to_speech import * 
from google_drive_downloader import GoogleDriveDownloader as gdd
import os



st.header("Demo Text to speech - HCMUS K30")
st.subheader("Phan Minh Toan 20C11057")


if (not os.path.exists('./output/ckpt/LJSpeech/900000.pth.tar')):
    with st.spinner(text="Download model in progress..."):
        gdd.download_file_from_google_drive(file_id='1Mv0LP0jmqijK6zknC9Q0MBYuBBIIUoD6',
                                    dest_path='./output/ckpt/LJSpeech/900000.pth.tar')

if (not os.path.exists('./hifigan_model/generator_LJSpeech.pth.tar')):
    with st.spinner(text="Download model in progress..."):
        gdd.download_file_from_google_drive(file_id='16vCMbc7uNHUdCFuzft4-G-GW_YCUF33m',
                                        dest_path='./hifigan_model/generator_LJSpeech.pth.tar')
    

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


