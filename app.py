import json
import os
import tempfile

import streamlit as st
from video_analyzer import analyze_video

st.set_page_config(
    page_title="QueueLens AI",
    page_icon="📹",
    layout="wide"
)

st.title("QueueLens AI")
st.subheader("Privacy-conscious video activity analysis")

st.info(
    "Upload a short video to measure motion. "
    "This prototype does not identify faces or directly count people."
)

uploaded = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded is not None:
    st.video(uploaded)

    if st.button("Analyse video", type="primary"):
        temp_path = None

        try:
            suffix = os.path.splitext(uploaded.name)[1] or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as temp:
                temp.write(uploaded.getbuffer())
                temp_path = temp.name

            with st.spinner("Analysing with OpenCV 5..."):
                result = analyze_video(temp_path)

            st.success("Analysis completed!")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Average motion",
                f"{result['average_motion_percent']}%"
            )
            col2.metric(
                "Peak motion",
                f"{result['peak_motion_percent']}%"
            )
            col3.metric(
                "Activity level",
                result["activity_level"]
            )

            st.subheader("Analysis details")
            st.json(result)

            st.download_button(
                "Download results as JSON",
                data=json.dumps(result, indent=2),
                file_name="queuelens_results.json",
                mime="application/json"
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

st.divider()
st.caption("QueueLens AI | OpenCV 5 prototype")
