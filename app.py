# app.py
import streamlit as st
import io
import speech_recognition as sr
from rag import generate_answer
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="Farmer Assistant", layout="wide")
st.title("🌾 KETH")

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- DISPLAY CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- MIC INPUT ----------------
audio = mic_recorder(
    start_prompt="🎤 Speak",
    stop_prompt="⏹ Stop",
    format="wav",
    key="recorder"
)

voice_text = ""

if audio and audio.get("bytes"):
    try:
        r = sr.Recognizer()
        audio_bytes = io.BytesIO(audio["bytes"])
        with sr.AudioFile(audio_bytes) as source:
            audio_data = r.record(source)
        voice_text = r.recognize_google(audio_data)
        st.success(f"🗣 You said: {voice_text}")
    except sr.UnknownValueError:
        st.warning("Could not understand audio. Please speak clearly.")
    except sr.RequestError as e:
        st.error(f"Google Speech API error: {e}")
    except Exception as e:
        st.error(f"Voice recognition failed: {e}")

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask your question...")

if voice_text:
    user_input = voice_text

# ---------------- PROCESS ----------------
if user_input:
    # ✅ Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # ✅ Pass full chat history to Groq
    with st.spinner("Thinking..."):
        response = generate_answer(
            query=user_input,
            chat_history=st.session_state.messages[:-1]  # exclude current message
        )

    if not response:
        response = "Sorry, I could not find the answer."

    # ✅ Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.markdown(response)

# ---------------- CLEAR CHAT BUTTON ----------------
if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()