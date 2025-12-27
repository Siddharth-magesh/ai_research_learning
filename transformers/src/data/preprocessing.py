def preprocess_text(text, config: dict) -> str:
    if config.get("lowercase", False):
        text = text.lower()
    text = text.strip()
    return text
