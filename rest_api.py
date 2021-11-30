from text_to_speech import * 
from google_drive_downloader import GoogleDriveDownloader as gdd
import os

import re
import unicodedata
import soundfile as sf

from hifigan.mel2wave import mel2wave
from nat.config import FLAGS
from nat.text2mel import text2mel

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import uvicorn
import os
from google_drive_downloader import GoogleDriveDownloader as gdd



if (not os.path.exists('./output/ckpt/LJSpeech/900000.pth.tar')):
    print("Download model in progress...")
    gdd.download_file_from_google_drive(file_id='1Mv0LP0jmqijK6zknC9Q0MBYuBBIIUoD6',
                                    dest_path='./output/ckpt/LJSpeech/900000.pth.tar')

if (not os.path.exists('./hifigan_model/generator_LJSpeech.pth.tar')):
    print("Download model in progress...")
    gdd.download_file_from_google_drive(file_id='16vCMbc7uNHUdCFuzft4-G-GW_YCUF33m',
                                        dest_path='./hifigan_model/generator_LJSpeech.pth.tar')

if (not os.path.exists('./assets/infore/hifigan/g_01135000')):
    print("Download model in progress...")
    gdd.download_file_from_google_drive(file_id='1oD1gg9yAdc1zG0uLK0n8hj6ZlhuhdZS8',
                                    dest_path='./assets/infore/hifigan/g_01135000')

if (not os.path.exists('./assets/infore/hifigan/hk_hifi.pickle')):
    print("Download model in progress...")
    gdd.download_file_from_google_drive(file_id='1Cc1AJXHBNAll-tftV0c_ijHiAhyvEbhV',
                                    dest_path='./assets/infore/hifigan/hk_hifi.pickle')  

if (not os.path.exists('./assets/infore/nat/nat_ckpt_latest.pickle')):
    print("Download model in progress...")
    gdd.download_file_from_google_drive(file_id='1FqSoXVwVNIjw6XOJYqrTJYsC0XweEZTa',
                                    dest_path='./assets/infore/nat/nat_ckpt_latest.pickle')                                                                          



class Item(BaseModel):
    text: str
    language: str
    speaker_gender: Optional[str] = "Female"


app = FastAPI()


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

@app.get("/")
async def root():
    return {"message": "Demo Rest API - Text To Speech "}


@app.post("/tts/")
async def text2speech(item: Item):
    input_text = item.text
    language = item.language
    speaker_gender = item.speaker_gender

    response = {}
    error_msg = ''
    response['status'] = ''
    response['err_msg'] = error_msg

    lang_list = ["english", "vietnamese"]
    speaker_gender_list = ['female', 'male']

    if language not in lang_list:
        error_msg = 'Please input language only english or vietnamese'
    
    if language == 'vietnamese' and speaker_gender not in speaker_gender_list:
        error_msg = 'Please input speaker gender only female or male'

    if error_msg != "":
        response['status'] = 'fail'
        response['err_msg'] = error_msg
        return response

    if language == 'english':
        text_to_speech(input_text)
    elif language == 'vietnamese':
        text = nat_normalize_text(input_text)
        if speaker_gender == "female":
            mel = text2mel(text, 'assets/infore/lexicon.txt', -1, speaker= 0)
        elif speaker_gender == "male":
            mel = text2mel(text, 'assets/infore/lexicon.txt', -1, speaker= 1)
        wave = mel2wave(mel)
        sf.write(str('result/audio.wav'), wave, samplerate=16000)

    with open('result/audio.wav', 'rb') as audio_file:
        encoded_string = base64.b64encode(audio_file.read())
        response['status'] = 'success'
        response['err_msg'] = ''
        response['wav_file_path'] = 'result/audio.wav'
        response['wav_file_base64'] = encoded_string

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)