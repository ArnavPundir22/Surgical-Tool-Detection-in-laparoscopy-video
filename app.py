import streamlit as st
import os
import time
from inference import process_video

st.set_page_config(page_title="Surgical Tool Detection", layout="wide")

# ======== STYLE ========
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ======== HEADER ========
st.markdown("# 🏥 Surgical Tool Detection System")
st.markdown("AI-powered tool recognition from surgical videos")

# ======== SIDEBAR ========
st.sidebar.title("⚙️ Settings")
option = st.sidebar.radio("Choose Input Type", ["Upload Video", "YouTube Link"])

video_path = None

# ======== MAIN UI ========
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input")

    if option == "Upload Video":
        uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

        if uploaded_file:
            with open("input.mp4", "wb") as f:
                f.write(uploaded_file.read())
            video_path = "input.mp4"
            st.success("Video uploaded!")

    elif option == "YouTube Link":
        url = st.text_input("Paste YouTube URL")

        if url:
            st.info("Downloading video...")
            os.system(f"yt-dlp -f best[ext=mp4] -o input.mp4 {url}")
            video_path = "input.mp4"
            st.success("Download complete!")

with col2:
    st.subheader("📊 Output")

    if video_path:
        if st.button("🚀 Run Detection"):
            try:
                with st.spinner("Processing video..."):
                    output_path = process_video(video_path)

                # IMPORTANT FIX (prevents error)
                time.sleep(1)

                st.success("Processing Complete!")

                # FIX: Use bytes instead of file path
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                    st.video(video_bytes)

                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Output",
                        data=f,
                        file_name="output.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"Error: {e}")

# ======== FOOTER ========
st.markdown("---")
st.markdown("Built with ❤️ using Deep Learning")
