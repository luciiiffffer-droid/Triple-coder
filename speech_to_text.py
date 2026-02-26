import whisper

model = whisper.load_model("base")

def convert_speech_to_text(file="input.wav"):
    result = model.transcribe(file)
    return result["text"]
