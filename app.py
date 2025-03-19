import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file,get_file
import google.generativeai as genai

import time
from pathlib import Path

import tempfile

from dotenv import load_dotenv
load_dotenv()

import os

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

#page configuration
st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="🎥",
    layout = "wide"
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f4f4f4;
        }
        .title {
            color: #ff4b4b;
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
        }
        .header {
            color: #333;
            text-align: center;
    </style>
    """,unsafe_allow_html=True)

st.markdown("<div class='title'>Video Summarizer AI Agent 🎥</div>", unsafe_allow_html=True)
st.markdown("<h3 class='header'>Powered by Gemini 2.0</h3>", unsafe_allow_html=True)

@st.cache_resource
def initialize_agent():
    return Agent(
        name = "Video AI summarizer",
        model = Gemini(id="gemini-2.0-flash-exp"),
        tools = [DuckDuckGo()],
        markdown= True
    )

##Initilaizing the Agent
multimodal_agent = initialize_agent()

#File Uploader
video_file = st.file_uploader(
    "Upload a Video File",type=["mp4","mov","avi"], help= "Upload a video for AI analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format='video/mp4',start_time=0)

    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the Video.The Agent will provide the specific information and additional information from the web as well",
        help="Provide specific questions or insights you want to get from the video"
    )

    if st.button("🔍Analyze Button", key= "analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights...."):
                    #Upload and process video file
                    processed_video = upload_file(video_path, mime_type="video/mp4")
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    #prompt generation for analysis
                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research.
                        {user_query}
                        Provide a detailed, user-friendly and actionable response
                        """
                    )

                    #AI Agent processing
                    response = multimodal_agent.run(analysis_prompt, videos=[processed_video])

                #Display the result
                st.subheader("Analysis Result")
                st.markdown(response.content)
            
            except Exception as error:
                st.error(f"An error occured during analysis:{error}")
            finally:
                #clean up temporary video file
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video to begin analysis.")


