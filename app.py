import streamlit as st
import os
import base64
import random
from io import BytesIO
from PIL import Image
from openai import OpenAI
import google.generativeai as genai
import anthropic
from audio_recorder_streamlit import audio_recorder

from config import get_prompt_template, load_env, PromptTemplate
from src.vectordb_utils import query_pinecone
from src.conv_db import load_all_sessions, save_session, rename_session, delete_session

# --- Constants ---
ANTHROPIC_MODELS = ["claude-opus-4-6"]
GOOGLE_MODELS = ["gemini-3.1-pro-preview"]
OPENAI_MODELS = ["gpt-5.4"]

TONE_CATEGORIES = {
    "casual": {
        "label": "Friends / Casual",
        "descriptors": "casual, relaxed, and friendly",
        "instruction": "Write in a casual, relaxed tone. Use contractions freely, slang is okay. Sound like texting a friend.",
    },
    "close": {
        "label": "Partner / Close",
        "descriptors": "warm, intimate, and affectionate",
        "instruction": "Write in a warm, affectionate tone. Be personal and caring. Keep it natural and loving without being cheesy.",
    },
    "professional": {
        "label": "Professional / Client",
        "descriptors": "professional, kind, and polished",
        "instruction": "Write in a professional but kind tone. Be polite, clear, and helpful. Avoid being stiff or robotic.",
    },
    "formal": {
        "label": "Formal / Business",
        "descriptors": "formal, respectful, and business-appropriate",
        "instruction": "Write in a formal, business-appropriate tone. Be respectful and structured. Use proper grammar, no contractions, and maintain professional distance.",
    },
}

def _tone_selector(key_suffix):
    """Render a tone/formality category selector. Returns the selected category dict."""
    labels = [v["label"] for v in TONE_CATEGORIES.values()]
    keys = list(TONE_CATEGORIES.keys())
    selected_label = st.selectbox(
        "Tone / Audience",
        labels,
        index=2,  # default to "Professional / Client"
        key=f"tone_select_{key_suffix}",
    )
    selected_key = keys[labels.index(selected_label)]
    return TONE_CATEGORIES[selected_key]

# --- Helper Functions ---

def _copy_button(text, key):
    """Render a small copy-to-clipboard button for the given text."""
    # Base64-encode the text so we don't need to worry about escaping
    # quotes, backticks, newlines, etc. in the JS string.
    text_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    st.components.v1.html(f"""
    <button id="copybtn" style="
        background: none;
        border: 1px solid #ccc;
        border-radius: 6px;
        padding: 4px 12px;
        cursor: pointer;
        font-size: 0.85em;
        color: #888;
    ">📋 Copy</button>
    <script>
    const btn = document.getElementById('copybtn');
    btn.addEventListener('click', function() {{
        const bytes = Uint8Array.from(atob("{text_b64}"), c => c.charCodeAt(0));
        const text = new TextDecoder('utf-8').decode(bytes);
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        btn.textContent = 'Copied!';
        setTimeout(function() {{ btn.textContent = '📋 Copy'; }}, 1500);
    }});
    </script>
    """, height=40)

def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
            elif content["type"] == "video_file":
                gemini_message["parts"].append(genai.upload_file(content["video_file"]))
            elif content["type"] == "audio_file":
                gemini_message["parts"].append(genai.upload_file(content["audio_file"]))

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]
        
    return gemini_messages

def messages_to_anthropic(messages):
    anthropic_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            anthropic_message = anthropic_messages[-1]
        else:
            anthropic_message = {
                "role": message["role"] ,
                "content": [],
            }
        if message["content"][0]["type"] == "image_url":
            anthropic_message["content"].append(
                {
                    "type": "image",
                    "source":{   
                        "type": "base64",
                        "media_type": message["content"][0]["image_url"]["url"].split(";")[0].split(":")[1],
                        "data": message["content"][0]["image_url"]["url"].split(",")[1]
                    }
                }
            )
        else:
            anthropic_message["content"].append(message["content"][0])

        if prev_role != message["role"]:
            anthropic_messages.append(anthropic_message)

        prev_role = message["role"]
        
    return anthropic_messages

def stream_llm_response(model_params, model_type, api_key, messages):
    response_message = ""
    timeout = 300  # 5 minutes timeout

    if model_type == "openai":
        client = OpenAI(api_key=api_key, timeout=timeout)
        for chunk in client.chat.completions.create(
            model=model_params.get("model", "gpt-5.4"),
            messages=messages,
            temperature=model_params.get("temperature", 0.7),
            stream=True,
        ):
            chunk_text = chunk.choices[0].delta.content or ""
            response_message += chunk_text
            yield chunk_text

    elif model_type == "google":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_params["model"],
            generation_config={
                "temperature": model_params.get("temperature", 0.7),
            }
        )
        gemini_messages = messages_to_gemini(messages)

        for chunk in model.generate_content(
            contents=gemini_messages,
            stream=True,
            request_options={'timeout': timeout}
        ):
            chunk_text = chunk.text or ""
            response_message += chunk_text
            yield chunk_text

    elif model_type == "anthropic":
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        with client.messages.stream(
            model=model_params.get("model", "claude-opus-4-6"),
            messages=messages_to_anthropic(messages),
            temperature=model_params.get("temperature", 0.7),
            max_tokens=4096,
        ) as stream:
            for text in stream.text_stream:
                response_message += text
                yield text

    return response_message

# --- State Management ---

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "prev_speech_hash" not in st.session_state:
        st.session_state.prev_speech_hash = None
    if "nav_selection" not in st.session_state:
        st.session_state.nav_selection = "💬 2English"
    # Upwork Response tab: multiple saved job sessions (loaded from SQLite)
    if "conv_sessions" not in st.session_state:
        st.session_state.conv_sessions = load_all_sessions()
    if "conv_active_id" not in st.session_state:
        st.session_state.conv_active_id = None  # currently active session id
    # Counter used to reset the file uploader widget after each follow-up
    if "conv_upload_key_counter" not in st.session_state:
        st.session_state.conv_upload_key_counter = 0

def on_nav_change():
    """
    Callback for navigation change.
    Clears chat history and draft inputs (except for Conversation tab).
    """
    current_tab = st.session_state.nav_selection
    
    # 2. Clear chatting history whenever tab is switched
    st.session_state.messages = []
    
    # 3. Clear draft in text field whenever tab is switched, except tab_conversation
    # We clear the volatile fields (Upwork, etc.) regardless of where we are going,
    # effectively resetting them.
    # The requirement "except tab_conversation" means we DON'T clear Conversation fields.
    volatile_keys = ["upwork_job_description", "screening_questions", "qr_client_message", "qr_reply_context"]
    for key in volatile_keys:
        if key in st.session_state:
            del st.session_state[key]

    # Conversation sessions persist across tab switches -- just clear the LLM messages
    # (sessions are kept in conv_sessions, active one tracked by conv_active_id)

# --- Render Functions ---

def render_sidebar():
    with st.sidebar:
        cols_keys = st.columns(2)
        with cols_keys[0]:
            default_openai = os.getenv("OPENAI_API_KEY") or ""
            with st.popover("🔐 OpenAI"):
                openai_api_key = st.text_input("OpenAI API Key", value=default_openai, type="password")
        
        with cols_keys[1]:
            default_google = os.getenv("GOOGLE_API_KEY") or ""
            with st.popover("🔐 Google"):
                google_api_key = st.text_input("Google API Key", value=default_google, type="password")

        default_anthropic = os.getenv("ANTHROPIC_API_KEY") or ""
        with st.popover("🔐 Anthropic"):
            anthropic_api_key = st.text_input("Anthropic API Key", value=default_anthropic, type="password")

        st.divider()
        
        available_models = [] + (ANTHROPIC_MODELS if anthropic_api_key else []) + \
                             (GOOGLE_MODELS if google_api_key else []) + \
                             (OPENAI_MODELS if openai_api_key else [])
        
        model = st.selectbox("Select a model:", available_models, index=0) if available_models else None
        
        model_type = None
        if model:
            if model.startswith("gpt"): model_type = "openai"
            elif model.startswith("gemini"): model_type = "google"
            elif model.startswith("claude"): model_type = "anthropic"
        
        with st.popover("⚙️ Model parameters"):
            model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

        audio_response = st.toggle("Audio response", value=False)
        tts_voice = "alloy"
        tts_model = "tts-1"
        if audio_response:
            cols = st.columns(2)
            with cols[0]:
                tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
            with cols[1]:
                tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=1)

        st.button("🗑️ Reset conversation", on_click=lambda: st.session_state.pop("messages", None))
        st.divider()

    return {
        "openai": openai_api_key,
        "google": google_api_key,
        "anthropic": anthropic_api_key
    }, {
        "model": model,
        "temperature": model_temp
    }, model_type, audio_response, tts_voice, tts_model

def render_2english(api_keys, model_params, model_type, audio_response, tts_voice, tts_model):
    # Check API keys
    if not model_type:
        st.warning("⬅️ Please introduce an API Key to continue...")
        return

    tone = _tone_selector("2eng")

    # # Image/Video Upload Logic
    # if model_params["model"] in ["gpt-4.1", "gemini-3-pro-preview", "claude-opus-4-5-20251101"]:
    #     st.write(f"### **🖼️ Add an image{' or a video file' if model_type=='google' else ''}:**")
        
    #     def add_image_to_messages():
    #         if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
    #             img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
    #             # Append to messages (this is part of the current "prompt" construction)
    #             if img_type == "video/mp4":
    #                 video_id = random.randint(100000, 999999)
    #                 with open(f"video_{video_id}.mp4", "wb") as f:
    #                     f.write(st.session_state.uploaded_img.read())
    #                 st.session_state.messages.append({
    #                     "role": "user", 
    #                     "content": [{"type": "video_file", "video_file": f"video_{video_id}.mp4"}]
    #                 })
    #             else:
    #                 raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
    #                 img = get_image_base64(raw_img)
    #                 st.session_state.messages.append({
    #                     "role": "user", 
    #                     "content": [{"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img}"}}]
    #                 })

    #     cols_img = st.columns(2)
    #     with cols_img[0]:
    #         with st.popover("📁 Upload"):
    #             st.file_uploader(
    #                 f"Upload an image{' or a video' if model_type == 'google' else ''}:", 
    #                 type=["png", "jpg", "jpeg"] + (["mp4"] if model_type == "google" else []), 
    #                 accept_multiple_files=False,
    #                 key="uploaded_img",
    #                 on_change=add_image_to_messages,
    #             )
    #     with cols_img[1]:                    
    #         with st.popover("📸 Camera"):
    #             if st.checkbox("Activate camera"):
    #                 st.camera_input("Take a picture", key="camera_img", on_change=add_image_to_messages)

    # # Audio Input
    # st.write("#")
    # st.write(f"### **🎤 Add an audio{' (Speech To Text)' if model_type == 'openai' else ''}:**")
    
    # audio_prompt = None
    # audio_file_added = False
    
    # speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395")
    # if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
    #     st.session_state.prev_speech_hash = hash(speech_input)
    #     if model_type != "google":
    #         # Transcribe
    #         client = OpenAI(api_key=api_keys["openai"]) # Whisper uses OpenAI
    #         transcript = client.audio.transcriptions.create(model="whisper-1", file=("audio.wav", speech_input))
    #         audio_prompt = transcript.text
    #     else:
    #         # Upload audio file
    #         audio_id = random.randint(100000, 999999)
    #         with open(f"audio_{audio_id}.wav", "wb") as f:
    #             f.write(speech_input)
    #         st.session_state.messages.append({
    #             "role": "user", 
    #             "content": [{"type": "audio_file", "audio_file": f"audio_{audio_id}.wav"}]
    #         })
    #         audio_file_added = True

    # st.divider()

    # Display Messages
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content["text"])
                    if message["role"] == "assistant":
                        _copy_button(content["text"], f"copy_2eng_{msg_idx}")
                elif content["type"] == "image_url":      
                    st.image(content["image_url"]["url"])
                elif content["type"] == "video_file":
                    st.video(content["video_file"])
                elif content["type"] == "audio_file":
                    st.audio(content["audio_file"])

    # Chat Input
    # if prompt := st.chat_input("Hi! Ask me anything...") or audio_prompt or audio_file_added:
    if prompt := st.chat_input("Hi! Ask me anything..."):
        # 4. Whenever new prompt is inputted, chatting history must be cleared.
        # Strategy: We keep the *pending* user inputs (like the image/audio just added)
        # but remove completed turns.
        # Actually, simplest is to filter the messages to keep only the last consecutive user messages.
        
        current_user_messages = []
        # Check existing messages for pending user inputs (images/files)
        if st.session_state.messages:
            # If the last message was assistant, clear everything.
            if st.session_state.messages[-1]["role"] == "assistant":
                st.session_state.messages = []
            else:
                # If last was user (e.g. image upload), keep it.
                # In fact, we only want to keep the *tail* sequence of user messages.
                # But for simplicity, if there's no assistant response yet, it's all new context.
                pass 
        
        # if not audio_file_added:
            # Prepare 2English prompt
            # final_prompt = prompt or audio_prompt
        final_prompt = prompt
        # 1. Convert user prompt to sentences as if usa native english speakers write/say
        system_instruction = f"Rewrite the following text to sound natural, {tone['descriptors']}, and native-like (USA English), while preserving the original meaning."
        full_text_prompt = f"{system_instruction}\n\nInput Text:\n{final_prompt}"
        
        st.session_state.messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": full_text_prompt}] # Store the full prompt or just display the original?
            # Usually better to display the original to the user, but send instructions to LLM.
            # But to keep it simple and stateless, we'll just append the text.
            # Let's append the raw text for display, but modify what we send to stream_llm_response if possible.
            # Since stream_llm_response takes messages directly, we'll just use the prompt text and rely on the instruction being there.
        })
        
        # Only show the user's original text in the chat (hack: replace the last message content for display vs logic)
        # Actually, let's just be transparent.
        
        with st.chat_message("user"):
            st.markdown(final_prompt)
        # else:
        #     with st.chat_message("user"):
        #         st.audio(f"audio_{audio_id}.wav")

        # Generate Response
        with st.chat_message("assistant"):
            # For 2English, we want to ensure the instruction is clear.
            # If we just appended text, we can modify the last message in the list passed to the API
            # without modifying session state (which is used for display).
            
            api_messages = [m.copy() for m in st.session_state.messages]
            # Inject instruction into the last text message if it's user
            if api_messages and api_messages[-1]["role"] == "user":
                last_content = api_messages[-1]["content"]
                for item in last_content:
                    if item["type"] == "text":
                        # Prepend instruction
                         item["text"] = f"You are a native USA English speaker helper. Rewrite this to sound {tone['descriptors']} and native-like:\n\n{item['text']}"
            
            response_text = ""
            response_container = st.empty()
            
            # Stream
            for chunk in stream_llm_response(model_params, model_type, api_keys[model_type], api_messages):
                response_text += chunk
                response_container.write(response_text)
            
            _copy_button(response_text, "copy_2eng_live")

            # Append assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}]
            })

        # Audio Response
        if audio_response and response_text:
            client = OpenAI(api_key=api_keys["openai"])
            response = client.audio.speech.create(
                model=tts_model,
                voice=tts_voice,
                input=response_text,
            )
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            st.html(f"""<audio controls autoplay><source src="data:audio/wav;base64,{audio_base64}" type="audio/mp3"></audio>""")

def _build_image_content(uploaded_images):
    """Convert uploaded image files into message content parts for the LLM."""
    image_parts = []
    for img_file in uploaded_images:
        raw_img = Image.open(img_file)
        img_b64 = get_image_base64(raw_img)
        img_type = img_file.type or "image/png"
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img_type};base64,{img_b64}"}
        })
    return image_parts

def render_upwork_proposal(api_keys, model_params, model_type, *args):
    job_description = st.text_area("Job Description *", height=200, key="upwork_job_description", placeholder="Paste the job description here...")
    screening_questions = st.text_area("Screening Questions (Optional)", height=150, key="screening_questions", placeholder="Paste any screening questions here...")
    important_points = st.text_area("Important Points (Optional)", height=150, key="important_points", placeholder="Paste any important points here...")

    uploaded_images = st.file_uploader(
        "📎 Attach client images (optional)",
        type=["png", "jpg", "jpeg", "gif", "webp"],
        accept_multiple_files=True,
        key="upwork_images",
        help="Upload screenshots or images the client attached to the job post."
    )

    if uploaded_images:
        cols = st.columns(min(len(uploaded_images), 4))
        for idx, img_file in enumerate(uploaded_images):
            with cols[idx % len(cols)]:
                st.image(img_file.read(), caption=img_file.name, use_container_width=True)

    if st.button("Generate Proposal", type="primary"):
        if not job_description:
            st.error("Please provide a job description")
            return

        with st.spinner("Generating proposal..."):
            prompt = get_prompt_template(PromptTemplate.GENERATE).format(
                experience=query_pinecone(job_description),
                job_description=job_description,
                important_points=important_points,
            )

            user_content = [{"type": "text", "text": prompt}]
            if uploaded_images:
                user_content.extend(_build_image_content(uploaded_images))

            st.session_state.messages.append({"role": "user", "content": user_content})

            with st.chat_message("assistant"):
                response_text = ""
                response_container = st.empty()
                for chunk in stream_llm_response(model_params, model_type, api_keys[model_type], st.session_state.messages):
                    response_text += chunk
                    response_container.write(response_text)
                
                _copy_button(response_text, "copy_proposal")
                st.session_state.messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})

            if screening_questions:
                sq_prompt = get_prompt_template(PromptTemplate.UPWORK_SCREENING_QUESTIONS).format(screening_questions=screening_questions)
                st.session_state.messages.append({"role": "user", "content": [{"type": "text", "text": sq_prompt}]})
                
                with st.chat_message("assistant"):
                    sq_response = ""
                    sq_container = st.empty()
                    for chunk in stream_llm_response(model_params, model_type, api_keys[model_type], st.session_state.messages):
                        sq_response += chunk
                        sq_container.write(sq_response)
                    
                    _copy_button(sq_response, "copy_screening")
                    st.session_state.messages.append({"role": "assistant", "content": [{"type": "text", "text": sq_response}]})

def _build_conv_messages(context, new_client_content=None):
    """
    Build proper multi-turn LLM messages for follow-up conversation.

    Structure:
      1. user: system prompt with job context + original conversation history
      2. assistant: first generated response
      3. user: client's follow-up message (may include images)
      4. assistant: our drafted reply
      ... and so on for each follow-up exchange.
      Last: user message with the new client message (if provided).

    This gives the LLM natural turn-taking instead of one giant prompt.
    """
    # First message: the system context prompt (same as initial generation)
    system_prompt = get_prompt_template(PromptTemplate.CONVERSATION_RESPONSE).format(
        job_description=context["job_description"],
        cover_letter=context["cover_letter"],
        conversation=context["conversation"],
    )
    messages = [
        {"role": "user", "content": [{"type": "text", "text": system_prompt}]}
    ]

    # Replay the accumulated follow-up exchanges as proper turns
    for entry in context["chat_history"]:
        if entry["role"] == "client":
            content = []
            if entry.get("image_parts"):
                content.extend(entry["image_parts"])
            content.append({"type": "text", "text": entry["text"]})
            messages.append({"role": "user", "content": content})
        else:  # assistant
            messages.append({"role": "assistant", "content": [{"type": "text", "text": entry["text"]}]})

    # Append the new client message if provided
    if new_client_content is not None:
        messages.append({"role": "user", "content": new_client_content})

    return messages

def _get_active_session():
    """Return the active session dict, or None."""
    sid = st.session_state.conv_active_id
    if sid and sid in st.session_state.conv_sessions:
        return st.session_state.conv_sessions[sid]
    return None

def _save_active_session(context, chat_history):
    """Save context and chat_history into the active session."""
    sid = st.session_state.conv_active_id
    if sid and sid in st.session_state.conv_sessions:
        st.session_state.conv_sessions[sid]["context"] = context
        st.session_state.conv_sessions[sid]["chat_history"] = chat_history

def _create_session_label(job_description):
    """Generate a short label from the job description (first ~50 chars)."""
    text = job_description.strip().replace("\n", " ")
    return text[:50] + ("..." if len(text) > 50 else "")

def _render_conv_right_panel():
    """Right panel: previous conversations list."""
    sessions = st.session_state.conv_sessions
    active_id = st.session_state.conv_active_id

    st.markdown("##### History")

    # "New" button at the top to start fresh
    if st.button("➕ New", key="conv_new_btn", use_container_width=True):
        st.session_state.conv_active_id = None
        st.session_state.messages = []
        st.rerun()

    if not sessions:
        st.caption("No conversations yet.")
        return

    for sid, sess in sessions.items():
        is_active = sid == active_id
        col_btn, col_edit, col_del = st.columns([5, 1, 1])
        with col_btn:
            label = ("▶ " if is_active else "") + sess["label"]
            if st.button(
                label,
                key=f"conv_sess_{sid}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    st.session_state.conv_active_id = sid
                    st.session_state.messages = []
                    st.rerun()
        with col_edit:
            with st.popover("✏️"):
                new_label = st.text_input(
                    "Rename",
                    value=sess["label"],
                    key=f"conv_rename_{sid}",
                    label_visibility="collapsed",
                )
                if st.button("Save", key=f"conv_rename_save_{sid}"):
                    new_label = new_label.strip()
                    if new_label and new_label != sess["label"]:
                        sess["label"] = new_label
                        rename_session(sid, new_label)
                        st.rerun()
        with col_del:
            if st.button("🗑️", key=f"conv_del_{sid}"):
                delete_session(sid)
                del st.session_state.conv_sessions[sid]
                if active_id == sid:
                    st.session_state.conv_active_id = None
                st.session_state.messages = []
                st.rerun()


def _render_conv_main_panel(api_keys, model_params, model_type):
    """Left/main panel: new conversation form + active chat."""
    active_session = _get_active_session()
    has_active = active_session is not None

    # --- New conversation form (or active session's context viewer) ---
    if not has_active:
        # Full form when no session is active
        st.markdown("##### New Conversation")
        job_description = st.text_area("Job Description *", height=150, key="conv_job_description", placeholder="Paste the original job description here...")
        initial_proposal = st.text_area("Initial Cover Letter/Proposal *", height=150, key="initial_proposal", placeholder="Paste your initial proposal or cover letter here...")
        conversation_history = st.text_area("Conversation History *", height=200, key="conversation_history", placeholder="Paste the conversation between you and the client here...")
        generate_initial = st.button("Generate Response", type="primary", key="generate_response")
    else:
        # Collapsed context when a session is active
        with st.expander("📋 Job Context"):
            job_description = st.text_area("Job Description *", height=150, key="conv_job_description", placeholder="Paste the original job description here...")
            initial_proposal = st.text_area("Initial Cover Letter/Proposal *", height=150, key="initial_proposal", placeholder="Paste your initial proposal or cover letter here...")
            conversation_history = st.text_area("Conversation History *", height=200, key="conversation_history", placeholder="Paste the conversation between you and the client here...")
            generate_initial = st.button("Generate Response", type="primary", key="generate_response")

    # --- Handle initial generation (creates a new session) ---
    if generate_initial:
        if not all([job_description, initial_proposal, conversation_history]):
            st.error("Please provide all required information")
            return

        new_id = f"s_{random.randint(100000, 999999)}"
        context = {
            "job_description": job_description,
            "cover_letter": initial_proposal,
            "conversation": conversation_history,
            "chat_history": [],
        }

        st.session_state.conv_sessions[new_id] = {
            "label": _create_session_label(job_description),
            "context": context,
            "chat_history": [],
        }
        st.session_state.conv_active_id = new_id
        st.session_state.messages = []

        with st.spinner("Generating response..."):
            api_messages = _build_conv_messages(context)
            st.session_state.messages = api_messages

            with st.chat_message("assistant"):
                response_text = ""
                response_container = st.empty()
                for chunk in stream_llm_response(model_params, model_type, api_keys[model_type], st.session_state.messages):
                    response_text += chunk
                    response_container.write(response_text)

                _copy_button(response_text, "copy_conv_initial")

            sess = st.session_state.conv_sessions[new_id]
            sess["context"]["chat_history"].append({"role": "assistant", "text": response_text})
            sess["chat_history"].append({"role": "assistant", "text": response_text})
            save_session(new_id, sess["label"], sess["context"], sess["chat_history"])
            st.rerun()

    # --- Active session: display chat history & follow-up input ---
    active_session = _get_active_session()  # refresh after potential creation
    if active_session and active_session["chat_history"]:
        st.divider()

        # Render all past exchanges
        for entry_idx, entry in enumerate(active_session["chat_history"]):
            if entry["role"] == "client":
                with st.chat_message("user"):
                    st.markdown(entry["text"])
                    if entry.get("images"):
                        img_cols = st.columns(min(len(entry["images"]), 4))
                        for i, img_url in enumerate(entry["images"]):
                            with img_cols[i % len(img_cols)]:
                                st.image(img_url, use_container_width=True)
            else:
                with st.chat_message("assistant"):
                    st.markdown(entry["text"])
                    _copy_button(entry["text"], f"copy_conv_{entry_idx}")

        # Image upload for follow-up messages (dynamic key to clear after each send)
        upload_key = f"conv_followup_images_{st.session_state.conv_upload_key_counter}"
        uploaded_images = st.file_uploader(
            "📎 Attach images from client (optional)",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            accept_multiple_files=True,
            key=upload_key,
            help="Upload screenshots or images the client sent in their latest message."
        )

        if uploaded_images:
            preview_cols = st.columns(min(len(uploaded_images), 4))
            for idx, img_file in enumerate(uploaded_images):
                with preview_cols[idx % len(preview_cols)]:
                    st.image(img_file, caption=img_file.name, use_container_width=True)

        # Follow-up input: paste client's latest message
        client_msg = st.chat_input("Paste client's message to get a draft reply...")
        if client_msg:
            context = active_session["context"]

            image_content_parts = []
            image_urls_for_display = []
            if uploaded_images:
                image_content_parts = _build_image_content(uploaded_images)
                image_urls_for_display = [p["image_url"]["url"] for p in image_content_parts]

            with st.chat_message("user"):
                st.markdown(client_msg)
                if image_urls_for_display:
                    img_cols = st.columns(min(len(image_urls_for_display), 4))
                    for i, img_url in enumerate(image_urls_for_display):
                        with img_cols[i % len(img_cols)]:
                            st.image(img_url, use_container_width=True)

            msg_text = client_msg
            if uploaded_images:
                msg_text += "\n\n(The client attached images above. Reference any relevant details in your response.)"

            chat_display_entry = {"role": "client", "text": client_msg}
            if image_urls_for_display:
                chat_display_entry["images"] = image_urls_for_display
            active_session["chat_history"].append(chat_display_entry)

            context_entry = {"role": "client", "text": msg_text}
            if image_content_parts:
                context_entry["image_parts"] = image_content_parts
            context["chat_history"].append(context_entry)

            api_messages = _build_conv_messages(context)
            st.session_state.messages = api_messages

            with st.chat_message("assistant"):
                response_text = ""
                response_container = st.empty()
                for chunk in stream_llm_response(model_params, model_type, api_keys[model_type], st.session_state.messages):
                    response_text += chunk
                    response_container.write(response_text)

                _copy_button(response_text, "copy_conv_live")

            active_session["chat_history"].append({"role": "assistant", "text": response_text})
            context["chat_history"].append({"role": "assistant", "text": response_text})
            sid = st.session_state.conv_active_id
            save_session(sid, active_session["label"], context, active_session["chat_history"])

            st.session_state.conv_upload_key_counter += 1
            st.rerun()


def render_conversation_response(api_keys, model_params, model_type, *args):
    # Two-column layout: main panel (left) + history panel (right)
    main_col, right_col = st.columns([3, 1])

    with right_col:
        _render_conv_right_panel()

    with main_col:
        _render_conv_main_panel(api_keys, model_params, model_type)


def render_quick_reply(api_keys, model_params, model_type, *args):
    if not model_type:
        st.warning("⬅️ Please introduce an API Key to continue...")
        return

    tone = _tone_selector("qr")

    client_message = st.text_area(
        "What the client said *",
        height=180,
        key="qr_client_message",
        placeholder="Paste what the other person said here...",
    )
    reply_context = st.text_area(
        "Your reply context / notes *",
        height=180,
        key="qr_reply_context",
        placeholder="Describe what you want to say, key points, tone, etc...",
    )

    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content["text"])
                    if message["role"] == "assistant":
                        _copy_button(content["text"], f"copy_qr_{msg_idx}")

    if st.button("Generate Reply", type="primary", key="qr_generate"):
        if not client_message or not reply_context:
            st.error("Please fill in both fields")
            return

        st.session_state.messages = []

        prompt = get_prompt_template(PromptTemplate.QUICK_REPLY).format(
            client_message=client_message,
            reply_context=reply_context,
            tone_instruction=tone["instruction"],
        )

        st.session_state.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        })

        with st.chat_message("assistant"):
            response_text = ""
            response_container = st.empty()
            for chunk in stream_llm_response(
                model_params, model_type, api_keys[model_type], st.session_state.messages
            ):
                response_text += chunk
                response_container.write(response_text)

            _copy_button(response_text, "copy_qr_live")

        st.session_state.messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}],
        })


# --- Main ---

def main():
    st.set_page_config(page_title="The Chai-Chat", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
    load_env()
    init_session_state()

    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>The Chai-Chat</i> 💬</h1>""")

    # Navigation (replaces st.tabs to allow for state management on switch)
    # We use radio button with horizontal=True to mimic tabs
    current_tab = st.radio(
        "Navigation",
        ["💬 2English", "💼 Upwork Proposal", "💬 Upwork Response", "✉️ Conversation Reply"],
        horizontal=True,
        key="nav_selection",
        on_change=on_nav_change,
        label_visibility="collapsed"
    )

    api_keys, model_params, model_type, audio_resp, tts_v, tts_m = render_sidebar()

    if current_tab == "💬 2English":
        render_2english(api_keys, model_params, model_type, audio_resp, tts_v, tts_m)
    elif current_tab == "💼 Upwork Proposal":
        render_upwork_proposal(api_keys, model_params, model_type)
    elif current_tab == "💬 Upwork Response":
        render_conversation_response(api_keys, model_params, model_type)
    elif current_tab == "✉️ Conversation Reply":
        render_quick_reply(api_keys, model_params, model_type)

if __name__=="__main__":
    main()
