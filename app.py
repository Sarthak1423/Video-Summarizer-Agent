import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

import time
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()

# Configure Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page config
st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .stApp { background-color: #0E1117; }
        .title { color: #ff4b4b; font-size: 2.5em; font-weight: bold; text-align: center; }
        .header { color: #dddddd; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>Video Summarizer AI Agent 🎥</div>", unsafe_allow_html=True)
st.markdown("<h3 class='header'>Powered by Gemini 2.0</h3>", unsafe_allow_html=True)

# Initialize history in session
if "history" not in st.session_state:
    st.session_state["history"] = []

# Initialize Agent (cached)
@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True
    )

multimodal_agent = initialize_agent()

# Tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload Video", "🔍 Query & Analysis", "📚 History"])

# -------------------
# 📤 Tab 1: Upload
# -------------------
with tab1:
    st.subheader("Step 1: Upload your video file")
    video_file = st.file_uploader(
        "Upload a Video File",
        type=["mp4", "mov", "avi"],
        help="Upload a video for AI analysis"
    )

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        st.video(video_path, format='video/mp4', start_time=0)
        st.success("✅ Video uploaded successfully.")
    else:
        st.info("Upload a video to begin analysis.")

# -------------------------
# 🔍 Tab 2: Query & Analyze
# -------------------------
with tab2:
    if "video_path" not in locals():
        st.warning("Please upload a video in the first tab.")
    else:
        st.subheader("Step 2: Ask your query")

        # Suggestions
        with st.expander("💡 Query Suggestions"):
            st.markdown("""
            - What is the summary of this video?
            - Identify the key people and places mentioned.
            - What are the main themes or subjects?
            - Extract important quotes or dialogues.
            - What happens at the end of the video?
            """)

        user_query = st.text_area(
            "Enter your question or insight request:",
            placeholder="e.g., Summarize the main points of the video...",
            help="Ask anything about the uploaded video."
        )

        if st.button("🔍 Analyze", key="analyze_video_button"):
            if not user_query:
                st.warning("Please enter a query before analyzing.")
            else:
                try:
                    with st.spinner("Processing video..."):
                        # Upload video to Gemini
                        processed_video = upload_file(video_path, mime_type="video/mp4")
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = get_file(processed_video.name)

                        # Prompt
                        analysis_prompt = f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        {user_query}
                        Provide a clear, detailed and actionable response.
                        """

                        response = multimodal_agent.run(analysis_prompt, videos=[processed_video])

                    # Store history
                    st.session_state["history"].append({
                        "query": user_query,
                        "response": response.content
                    })

                    # Output
                    st.subheader("Analysis Result")
                    st.markdown(response.content)

                except Exception as error:
                    st.error(f"❌ Error: {error}")
                finally:
                    Path(video_path).unlink(missing_ok=True)

# -------------------
# 📚 Tab 3: History
# -------------------
with tab3:
    st.subheader("🕘 Previous Queries & Results")
    if not st.session_state["history"]:
        st.info("You haven't analyzed any videos yet.")
    else:
        for i, item in enumerate(reversed(st.session_state["history"]), 1):
            st.markdown(f"**{i}. Query:** {item['query']}")
            st.markdown(f"**Answer:** {item['response']}")
            st.markdown("---")
