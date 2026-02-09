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

# --- Constants ---
ANTHROPIC_MODELS = ["claude-opus-4-6"]
GOOGLE_MODELS = ["gemini-3-pro-preview"]
OPENAI_MODELS = ["gpt-5.2"]

# --- Helper Functions ---

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
            model=model_params.get("model", "gpt-5.2"),
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
    # Conversation Response tab: persistent chat history across follow-ups
    if "conv_context" not in st.session_state:
        st.session_state.conv_context = None  # stores the initial context dict
    if "conv_chat_history" not in st.session_state:
        st.session_state.conv_chat_history = []  # list of {role, text} dicts for display

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
    volatile_keys = ["upwork_job_description", "screening_questions"]
    for key in volatile_keys:
        if key in st.session_state:
            del st.session_state[key]

    # Clear conversation response follow-up chat when leaving its tab
    st.session_state.conv_context = None
    st.session_state.conv_chat_history = []

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
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            for content in message["content"]:
                if content["type"] == "text":
                    st.write(content["text"])
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
        system_instruction = "Rewrite the following text to sound natural, professional, and native-like (USA English), while preserving the original meaning."
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
                         item["text"] = f"You are a native USA English speaker helper. Rewrite this to be native-like:\n\n{item['text']}"
            
            response_text = ""
            response_container = st.empty()
            
            # Stream
            for chunk in stream_llm_response(model_params, model_type, api_keys[model_type], api_messages):
                response_text += chunk
                response_container.write(response_text)
            
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

            # Build user message content: images first, then the text prompt
            user_content = []
            if uploaded_images:
                user_content.extend(_build_image_content(uploaded_images))
                # Add instruction so the LLM knows images are attached
                prompt += "\n\n**Note:** The client attached images to this job post. Analyze them carefully and incorporate any relevant details into your proposal."
            user_content.append({"type": "text", "text": prompt})

            st.session_state.messages.append({"role": "user", "content": user_content})

            with st.chat_message("assistant"):
                response_text = ""
                response_container = st.empty()
                for chunk in stream_llm_response(model_params, model_type, api_keys[model_type], st.session_state.messages):
                    response_text += chunk
                    response_container.write(response_text)
                
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

def render_conversation_response(api_keys, model_params, model_type, *args):
    st.header("Conversation Response Generator")

    has_context = st.session_state.conv_context is not None

    # --- Initial setup form (collapsible once context is set) ---
    with st.expander("📋 Job Context", expanded=not has_context):
        job_description = st.text_area("Job Description *", height=150, key="conv_job_description", placeholder="Paste the original job description here...")
        initial_proposal = st.text_area("Initial Cover Letter/Proposal *", height=150, key="initial_proposal", placeholder="Paste your initial proposal or cover letter here...")
        conversation_history = st.text_area("Conversation History *", height=200, key="conversation_history", placeholder="Paste the conversation between you and the client here...")

        col_gen, col_reset = st.columns([3, 1])
        with col_gen:
            generate_initial = st.button("Generate Response", type="primary", key="generate_response")
        with col_reset:
            if has_context:
                if st.button("🗑️ Reset Chat", key="reset_conv_chat"):
                    st.session_state.conv_context = None
                    st.session_state.conv_chat_history = []
                    st.session_state.messages = []
                    st.rerun()

    # --- Handle initial generation ---
    if generate_initial:
        if not all([job_description, initial_proposal, conversation_history]):
            st.error("Please provide all required information")
            return

        # Store context for follow-up messages
        st.session_state.conv_context = {
            "job_description": job_description,
            "cover_letter": initial_proposal,
            "conversation": conversation_history,
            "chat_history": [],  # will accumulate follow-up exchanges as proper turns
        }
        st.session_state.conv_chat_history = []
        st.session_state.messages = []

        with st.spinner("Generating response..."):
            # Build messages using the proper multi-turn builder (just the initial context)
            api_messages = _build_conv_messages(st.session_state.conv_context)
            st.session_state.messages = api_messages

            with st.chat_message("assistant"):
                response_text = ""
                response_container = st.empty()
                for chunk in stream_llm_response(model_params, model_type, api_keys[model_type], st.session_state.messages):
                    response_text += chunk
                    response_container.write(response_text)

            # Save the first assistant response into both display history and context turns
            st.session_state.conv_chat_history.append({"role": "assistant", "text": response_text})
            st.session_state.conv_context["chat_history"].append({"role": "assistant", "text": response_text})
            st.rerun()

    # --- Display chat history & follow-up input ---
    if has_context and st.session_state.conv_chat_history:
        st.divider()
        st.subheader("💬 Chat")

        # Render all past exchanges
        for entry in st.session_state.conv_chat_history:
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

        # Image upload for follow-up messages
        uploaded_images = st.file_uploader(
            "📎 Attach images from client (optional)",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            accept_multiple_files=True,
            key="conv_followup_images",
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
            # Build image content if images are attached
            image_content_parts = []
            image_urls_for_display = []
            if uploaded_images:
                image_content_parts = _build_image_content(uploaded_images)
                image_urls_for_display = [p["image_url"]["url"] for p in image_content_parts]

            # Show the client message immediately
            with st.chat_message("user"):
                st.markdown(client_msg)
                if image_urls_for_display:
                    img_cols = st.columns(min(len(image_urls_for_display), 4))
                    for i, img_url in enumerate(image_urls_for_display):
                        with img_cols[i % len(img_cols)]:
                            st.image(img_url, use_container_width=True)

            # Build the new user content (images + text)
            new_user_content = []
            if image_content_parts:
                new_user_content.extend(image_content_parts)
            msg_text = client_msg
            if uploaded_images:
                msg_text += "\n\n(The client attached images above. Reference any relevant details in your response.)"
            new_user_content.append({"type": "text", "text": msg_text})

            # Record to display history
            chat_display_entry = {"role": "client", "text": client_msg}
            if image_urls_for_display:
                chat_display_entry["images"] = image_urls_for_display
            st.session_state.conv_chat_history.append(chat_display_entry)

            # Record to context (keeps image_parts so _build_conv_messages can replay them)
            context_entry = {"role": "client", "text": msg_text}
            if image_content_parts:
                context_entry["image_parts"] = image_content_parts
            st.session_state.conv_context["chat_history"].append(context_entry)

            # Build proper multi-turn messages: system context → assistant → user → assistant → ...
            api_messages = _build_conv_messages(st.session_state.conv_context)
            st.session_state.messages = api_messages

            with st.chat_message("assistant"):
                response_text = ""
                response_container = st.empty()
                for chunk in stream_llm_response(model_params, model_type, api_keys[model_type], st.session_state.messages):
                    response_text += chunk
                    response_container.write(response_text)

            # Save to display history + context for next round
            st.session_state.conv_chat_history.append({"role": "assistant", "text": response_text})
            st.session_state.conv_context["chat_history"].append({"role": "assistant", "text": response_text})
            st.rerun()

# --- Main ---

def main():
    st.set_page_config(page_title="The Chai-Chat", page_icon="🤖", layout="centered", initial_sidebar_state="expanded")
    load_env()
    init_session_state()

    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>The Chai-Chat</i> 💬</h1>""")

    # Navigation (replaces st.tabs to allow for state management on switch)
    # We use radio button with horizontal=True to mimic tabs
    current_tab = st.radio(
        "Navigation",
        ["💬 2English", "💼 Upwork Proposal", "💬 Conversation Response"],
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
    elif current_tab == "💬 Conversation Response":
        render_conversation_response(api_keys, model_params, model_type)

if __name__=="__main__":
    main()
