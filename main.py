from record import record_audio
from speech_to_text import convert_speech_to_text
from emotion import detect_emotion
from chat_engine import generate_response
from text_to_voice import speak


def run_voice_assistant():
    # Step 1: record voice
    record_audio()

    # Step 2: speech to text
    text = convert_speech_to_text()
    print("User:", text)

    # Step 3: detect emotion
    emotion = detect_emotion(text)
    print("Emotion:", emotion)

    # Step 4: AI response
    response = generate_response(text, emotion)
    print("Bot:", response)

    # Step 5: speak response
    speak(response)


run_voice_assistant()

