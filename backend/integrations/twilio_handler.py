"""
Twilio voice call webhook handler.
"""

from fastapi import APIRouter, Request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from loguru import logger

router = APIRouter(prefix="/api/twilio", tags=["twilio"])


@router.post("/voice")
async def handle_incoming_call(request: Request):
    """Handle incoming Twilio voice call — greets and gathers speech."""
    response = VoiceResponse()
    gather = Gather(
        input="speech",
        action="/api/twilio/process-speech",
        language="en-IN",
        speech_timeout="auto",
        timeout=5,
    )
    gather.say(
        "Hello! Welcome to AI Voice Assistant. How can I help you today?",
        voice="Polly.Aditi",
        language="en-IN",
    )
    response.append(gather)
    response.say("I didn't catch that. Goodbye!")
    response.hangup()

    logger.info("Twilio incoming call handled")
    return Response(content=str(response), media_type="application/xml")


@router.post("/process-speech")
async def process_speech(request: Request):
    """Process gathered speech from Twilio and respond."""
    form_data = await request.form()
    speech_result = form_data.get("SpeechResult", "")

    logger.info(f"Twilio speech: {speech_result}")

    # In production, this would call ai_service.generate_response()
    response = VoiceResponse()
    response.say(
        f"You said: {speech_result}. Let me help you with that.",
        voice="Polly.Aditi",
    )

    gather = Gather(
        input="speech",
        action="/api/twilio/process-speech",
        language="en-IN",
        speech_timeout="auto",
    )
    gather.say("Is there anything else I can help you with?", voice="Polly.Aditi")
    response.append(gather)

    response.say("Thank you for calling. Goodbye!")
    response.hangup()

    return Response(content=str(response), media_type="application/xml")


@router.post("/status-callback")
async def status_callback(request: Request):
    """Handle Twilio call status updates."""
    form_data = await request.form()
    call_status = form_data.get("CallStatus", "unknown")
    logger.info(f"Twilio call status: {call_status}")
    return {"status": "received"}
