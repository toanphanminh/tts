import streamlit as st
from text_to_speech import * 
from google_drive_downloader import GoogleDriveDownloader as gdd
import os

import re
import unicodedata
import soundfile as sf

from hifigan.mel2wave import mel2wave
from nat.config import FLAGS
from nat.text2mel import text2mel


def nat_normalize_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.lower().strip()
    sp = FLAGS.special_phonemes[FLAGS.sp_index]
    text = re.sub(r'[\n.,:]+', f' {sp} ', text)
    text = text.replace('"', " ")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[.,:;?!]+', f' {sp} ', text)
    text = re.sub('[ ]+', ' ', text)
    text = re.sub(f'( {sp}+)+ ', f' {sp} ', text)
    return text.strip()

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

if (not os.path.exists('./assets/infore/hifigan/g_01135000')):
    with st.spinner(text="Download model in progress..."):
        gdd.download_file_from_google_drive(file_id='1oD1gg9yAdc1zG0uLK0n8hj6ZlhuhdZS8',
                                    dest_path='./assets/infore/hifigan/g_01135000')

if (not os.path.exists('./assets/infore/hifigan/hk_hifi.pickle')):
    with st.spinner(text="Download model in progress..."):
        gdd.download_file_from_google_drive(file_id='1Cc1AJXHBNAll-tftV0c_ijHiAhyvEbhV',
                                    dest_path='./assets/infore/hifigan/hk_hifi.pickle')  

if (not os.path.exists('./assets/infore/nat/nat_ckpt_latest.pickle')):
    with st.spinner(text="Download model in progress..."):
        gdd.download_file_from_google_drive(file_id='1FqSoXVwVNIjw6XOJYqrTJYsC0XweEZTa',
                                    dest_path='./assets/infore/nat/nat_ckpt_latest.pickle')                                                                          

text_input = st.text_area("Input Text", "Hi, my name is Minh Toan, welcome to my demo of text to speech")
option_lang = st.selectbox("Select Language", ("English", "Vietnamese"))
if option_lang == "Vietnamese":
    speaker_type = st.selectbox("Speaker Type", ("Female", "Male"))
    if speaker_type == "Female":
        speaker = 0
    else:
        speaker = 1


if st.button("Text to speech"):
    with st.spinner(text="In progress..."):
        if option_lang == "Vietnamese":
            text = nat_normalize_text(text_input)
            if speaker_type == "Female":
                mel = text2mel(text, 'assets/infore/lexicon.txt', -1, speaker= 0)
            elif speaker_type == "Male":
                mel = text2mel(text, 'assets/infore/lexicon.txt', -1, speaker= 1)
            else:
                st.error("Please select speaker type")
            wave = mel2wave(mel)
            sf.write(str('result/audio.wav'), wave, samplerate=16000)
        elif option_lang == "English":
            text_to_speech(text_input)
           
        audio_file = open('result/audio.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
        st.success("Done.")


