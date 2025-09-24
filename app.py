import os
import time
from typing import Any
from transformers import pipeline
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
    """Generate detailed & descriptive caption/scenario from an image using transformers' BLIP pipeline.

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
    """
    Use Gemini to rewrite the extracted scenario in a slightly more descriptive
    and polished way, but WITHOUT inventing new details. 
    The output must remain faithful to the original input.
    """

    if not GEMINI_API_KEY:
        return (
            "No GEMINI_API_KEY provided. Install and set GEMINI_API_KEY in your .env.\n\n"
            "Fallback: " + (scenario[:200] + "..." if len(scenario) > 200 else scenario)
        )

    if genai is None:
        return (
            "google.generativeai package not installed. Install google-generativeai and restart.\n"
            "Fallback scenario: " + (scenario[:200] + "..." if len(scenario) > 200 else scenario)
        )

    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

    try:
        # ✅ STRICT prompt: elaborate descriptively but do not add imaginary content
        prompt = (
            f"Rewrite the following scene in a slightly more descriptive and polished way, "
            f"but do not add, remove, or invent any details. Keep all original content intact. "
            f"Output should be <=50 words.\n\nScene: {scenario}"
        )

        # Preferred new API
        try:
            resp = genai.generate(model="gemini-1.5-flash", prompt=prompt)
            text = getattr(resp, "text", None) or (
                resp.get("candidates")[0].get("content") if isinstance(resp, dict) else None
            )
            if text:
                return text.strip()
        except Exception:
            pass

        # Older API (models.generate)
        try:
            if hasattr(genai, "models") and hasattr(genai.models, "generate"):
                resp = genai.models.generate(model="gemini-1.5-flash", prompt=prompt)
                text = resp.text if hasattr(resp, "text") else (
                    resp.get("candidates")[0]["content"]
                    if isinstance(resp, dict) and resp.get("candidates")
                    else None
                )
                if text:
                    return text.strip()
        except Exception:
            pass

        # generate_content API
        try:
            if hasattr(genai, "GenerativeModel"):
                model = genai.GenerativeModel("gemini-1.5-flash")
                resp = model.generate_content(prompt)
                text = getattr(resp, "text", None)
                if text:
                    return text.strip()
        except Exception:
            pass

    except Exception:
        pass

    # Fallback: just return the scenario (keeps originality)
    return f"(Fallback - original preserved) {scenario}"

def generate_speech_from_text_hf(message: str) -> str:
    """Convert text to speech using HuggingFace API or fallback to pyttsx3."""
    os.makedirs("audio", exist_ok=True)  # ensure audio dir exists
    timestamp = int(time.time())
    out_path = os.path.join("audio", f"story_{timestamp}.wav")

    # 1) Hugging Face API (if token provided)
    if HUGGINGFACE_API_TOKEN:
        try:
            model_name = "facebook/mms-tts-eng"
            API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
            payload = {"inputs": message[:500]}

            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

            if response.status_code == 200 and response.headers.get("content-type", "").startswith("audio"):
                with open(out_path, "wb") as f:
                    f.write(response.content)
                return out_path
            else:
                st.warning(f"HuggingFace TTS failed ({response.status_code}), falling back to local TTS.")
        except Exception as e:
            st.warning(f"HuggingFace TTS error: {e}. Falling back to local TTS.")

    # 2) Local fallback (pyttsx3)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(message[:400], out_path)
        engine.runAndWait()
        return out_path
    except Exception as e:
        raise RuntimeError(f"Both HuggingFace API and local TTS failed. Last error: {e}")

def main() -> None:
    st.set_page_config(page_title="IMAGE TO STORY CONVERTER", page_icon="🖼️")
    st.markdown(css_code, unsafe_allow_html=True)

    # Top-level import check
    if not ensure_transformers_available():
        st.stop()

    # Ensure folders exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("audio", exist_ok=True)

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

    # Save uploaded file inside /uploads with timestamp to avoid overwrite
    file_name = uploaded_file.name
    timestamp = int(time.time())
    safe_name = f"{timestamp}_{file_name}"
    image_path = os.path.join("uploads", safe_name)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(image_path, caption="Uploaded Image", use_container_width=True)
    progress_bar(60)

    # 1) Image -> Text (caption / scenario)
    try:
        scenario = generate_text_from_image_local(image_path)
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
        story = "(Fallback) " + (scenario[:160] + "..." if len(scenario) > 160 else scenario)

    # 3) Story -> Speech (HuggingFace TTS)
    audio_file = None
    try:
        audio_file = generate_speech_from_text_hf(story)
    except Exception as e:
        st.error("Failed to generate speech (Hugging Face).")
        st.exception(e)

    # Show outputs
    with st.expander("Generated Image scenario"):
        st.write(scenario)

    with st.expander("Generated short story"):
        st.write(story)

    if audio_file and os.path.exists(audio_file):
        st.audio(audio_file)
        with open(audio_file, "rb") as f:
            st.download_button(
                "Download Audio",
                f,
                file_name=os.path.basename(audio_file),
                mime="audio/wav"
            )
    else:
        st.info("Audio not available (HuggingFace TTS may have failed). You can still read the generated story above.")


if __name__ == "__main__":
    main()