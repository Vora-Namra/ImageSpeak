import os
import time
from typing import Any

import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv

# transformers pipeline import wrapped in try/except for clearer error messages
try:
    from transformers import pipeline
except Exception as e:
    pipeline = None
    _TRANSFORMERS_IMPORT_ERROR = e

# google generative ai client - wrapped too
try:
    import google.generativeai as genai
except Exception as e:
    genai = None
    _GENAI_IMPORT_ERROR = e

from utils.custom import css_code

# Load env variables
load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def progress_bar(amount_of_time: int) -> Any:
    progress_text = "Please wait — generative models are hard at work..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(amount_of_time):
        time.sleep(0.02)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(0.3)
    my_bar.empty()


def ensure_transformers_available():
    """Check if transformers.pipeline is available and give actionable error if not."""
    if pipeline is None:
        st.error("Error importing transformers.pipeline. See instructions below.")
        st.write("Import error details:")
        st.code(str(_TRANSFORMERS_IMPORT_ERROR))
        st.info(
            "On Windows, install a compatible transformers and torch version. "
            "Recommended: Python 3.10/3.11, then install torch (CPU wheel) and transformers==4.41.2.\n\n"
            "Example commands (PowerShell):\n"
            "python -m venv venv\n"
            "venv\\Scripts\\activate\n"
            "python -m pip install --upgrade pip setuptools wheel\n"
            "pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2\n"
            "pip install transformers==4.41.2 pillow"
        )
        return False
    return True


def generate_text_from_image_local(path_or_bytes: str) -> str:
    """Generate a short caption/scenario from an image using transformers' BLIP pipeline.

    Accepts a file path or bytes-like object (the pipeline will accept both).
    """
    if not ensure_transformers_available():
        raise RuntimeError("transformers.pipeline not available")

    # pipeline name chosen to match newer alias; if this fails, try 'image-to-text' (older)
    for pipeline_name in ("image-to-image-captioning", "image-to-text", "image-captioning"):
        try:
            image_to_text: Any = pipeline(pipeline_name, model="Salesforce/blip-image-captioning-base")
            output = image_to_text(path_or_bytes)
            # pipeline returns list of dicts; best readable key is "generated_text"
            if isinstance(output, list) and len(output) > 0:
                txt = output[0].get("generated_text") or output[0].get("caption") or str(output[0])
            else:
                txt = str(output)
            return txt
        except Exception:
            # try next pipeline alias
            continue

    # if none of the pipeline aliases worked, raise informative error
    raise RuntimeError(
        "Could not run image captioning pipeline. Try installing transformers==4.41.2 and compatible torch, "
        "or review the pipeline name for your installed transformers version."
    )


def generate_story_from_text_gemini(scenario: str) -> str:
    """Generate a short story using Google Gemini (via google.generativeai).

    This function attempts to call the google.generativeai client if installed and configured.
    If the client isn't present or the API fails, we return a helpful message or fallback story.
    """
    if not GEMINI_API_KEY:
        # no key provided: return helpful fallback text (so app stays responsive)
        return (
            "No GEMINI_API_KEY provided. Install and set GEMINI_API_KEY in your .env to get an AI-generated story.\n\n"
            "Fallback: " + (scenario[:200] + "..." if len(scenario) > 200 else scenario)
        )

    if genai is None:
        # client not installed
        return (
            "google.generativeai package not installed or failed to import. "
            "Install google-generativeai and restart app.\n"
            "Fallback scenario: " + (scenario[:200] + "..." if len(scenario) > 200 else scenario)
        )

    # configure client
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        # Some versions use genai.configure; if it fails, continue — we'll try other calls below
        pass

    # attempt to call the typical client method(s)
    try:
        # Preferred current usage in some SDKs: genai.generate or genai.generate_text
        # We attempt multiple plausible APIs and pick the first that works.
        # 1) Try modern/simple genai.generate (may vary by SDK version)
        try:
            resp = genai.generate(model="gemini-1.5-flash", prompt=f"Create a short story (<=50 words) from: {scenario}")
            text = getattr(resp, "text", None) or (resp.get("candidates")[0].get("content") if isinstance(resp, dict) else None)
            if text:
                return text.strip()
        except Exception:
            pass

        # 2) Try older-style genai.messages or genai.models.generate_content if available
        try:
            # genai may expose models or a messages API
            if hasattr(genai, "models") and hasattr(genai.models, "generate"):
                resp = genai.models.generate(model="gemini-1.5-flash", prompt=f"Create a short story (<=50 words) from: {scenario}")
                # resp shape may vary:
                text = resp.text if hasattr(resp, "text") else (resp.get("candidates")[0]["content"] if isinstance(resp, dict) and resp.get("candidates") else None)
                if text:
                    return text.strip()
        except Exception:
            pass

        # 3) As last attempt, try a "generate_content" style API
        try:
            if hasattr(genai, "GenerativeModel"):
                model = genai.GenerativeModel("gemini-1.5-flash")
                resp = model.generate_content(f"Create a short story (<=50 words) from: {scenario}")
                text = getattr(resp, "text", None)
                if text:
                    return text.strip()
        except Exception:
            pass

    except Exception:
        # fall through to fallback
        pass

    # final fallback: produce a naive short story locally (guaranteed to run)
    words = scenario.split()
    short = " ".join(words[:30]).strip()
    if not short:
        short = "A lively scene unfolds as colors and faces tell a small, quiet story."
    return f"(Fallback short story) {short}."


def generate_speech_from_text_hf(message: str) -> str:
    """Call Hugging Face TTS model via inference API, save file, and return filename.

    Returns the path to the saved audio file.
    """
    if not HUGGINGFACE_API_TOKEN:
        raise RuntimeError("HUGGINGFACE_API_TOKEN not set in .env")

    API_URL: str = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers: dict[str, str] = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payloads: dict[str, str] = {"inputs": message}

    response = requests.post(API_URL, headers=headers, json=payloads, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"HuggingFace inference failed: {response.status_code} {response.text}")

    out_path = "generated_audio.flac"
    with open(out_path, "wb") as f:
        f.write(response.content)
    return out_path


def main() -> None:
    st.set_page_config(page_title="IMAGE TO STORY CONVERTER", page_icon="🖼️")
    st.markdown(css_code, unsafe_allow_html=True)

    # Top-level import check
    if not ensure_transformers_available():
        st.stop()

    with st.sidebar:
        # show a sample if exists
        if os.path.exists("img/gkj.jpg"):
            st.image("img/gkj.jpg", use_container_width=True)
        st.write("---")
        st.write("AI App created with Gemini + HuggingFace")

    st.header("Image-to-Story Converter (Gemini Version)")

    uploaded_file = st.file_uploader("Please choose a JPG file to upload", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Upload an image to begin. If you do not have an image handy, use the sample in img/gkj.jpg")
        return

    # Save uploaded file locally (safe for Windows)
    file_name = uploaded_file.name
    with open(file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(file_name, caption="Uploaded Image", use_container_width=True)
    progress_bar(60)

    # 1) Image -> Text (caption / scenario)
    try:
        scenario = generate_text_from_image_local(file_name)
    except Exception as e:
        st.error("Failed to generate text from image.")
        st.exception(e)
        return

    # 2) Text -> Story (Gemini or fallback)
    try:
        story = generate_story_from_text_gemini(scenario)
    except Exception as e:
        st.error("Failed to generate story from text.")
        st.exception(e)
        # produce a fallback short story so UI continues
        story = "(Fallback) " + (scenario[:160] + "..." if len(scenario) > 160 else scenario)

    # 3) Story -> Speech (HuggingFace TTS)
    audio_file = None
    try:
        audio_file = generate_speech_from_text_hf(story)
    except Exception as e:
        st.error("Failed to generate speech (Hugging Face).")
        st.exception(e)
        # do not return — still show scenario and story

    # Show outputs
    with st.expander("Generated Image scenario"):
        st.write(scenario)

    with st.expander("Generated short story"):
        st.write(story)

    if audio_file and os.path.exists(audio_file):
        st.audio(audio_file)
    else:
        st.info("Audio not available (HuggingFace TTS may have failed). You can still read the generated story above.")


if __name__ == "__main__":
    main()
