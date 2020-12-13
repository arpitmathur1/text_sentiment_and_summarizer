import speech_recognition as sr

# Initialize recognizer class
r = sr.Recognizer()

# read audio file as source
# listening the audio file and store in audio_text variable

with sr.AudioFile('maybe-next-time.wav') as source:
#with sr.AudioFile('2.wav') as source:
    audio_text = r.listen(source)

    try:
        
        # using google speech recognition
        text = r.recognize_google(audio_text)
        print('The text is:'. text)
     
    except:
         print('Sorry.. run again in some time')