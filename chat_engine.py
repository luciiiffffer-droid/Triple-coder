def generate_response(text, emotion):
    if emotion == "angry":
        return "I understand your frustration. Let me help you solve this quickly."

    if "payment" in text.lower():
        return "Your payment issue can be fixed by checking transaction history."

    if "order" in text.lower():
        return "Your order is being processed. It will arrive soon."

    return "Thank you for contacting us. How may I help you?"