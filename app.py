import streamlit as st
import os
import logging
from pathlib import Path
from typing import List

# Import backend functions
from doc_processor_backend import (
    process_documents,
    batch_summarize_documents,
    process_stream_components,
    DOCS_FOLDER
)

# Setup & Configuration
Path(DOCS_FOLDER).mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

# Streamlit UI Configuration
st.set_page_config(layout="wide", page_title="Document Summarization")

# Custom CSS
custom_css = """
body {
    color: #002B5B;
    background-color: #FFFFFF;
}
.stButton>button {
    border: 2px solid #2596be;
    background-color: #2596be;
    color: white;
    padding: 0.5em 1em;
    border-radius: 0.3em;
}
"""
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# Session state initialization
if 'processing_started' not in st.session_state:
    st.session_state.processing_started = False
if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = None
if 'summaries' not in st.session_state:
    st.session_state.summaries = []
if 'uploaded_file_paths' not in st.session_state:
    st.session_state.uploaded_file_paths = []
if 'stream_containers' not in st.session_state:
    st.session_state.stream_containers = {}

# Helper function to clean up files
def cleanup_files(file_paths: List[str]):
    for file_path in file_paths:
        try:
            Path(file_path).unlink()
            logger.info(f"Removed: {file_path}")
        except Exception as e:
            logger.error(f"Error removing {file_path}: {e}")

# UI Elements - Sidebar
st.sidebar.header("⚙️ Configuration")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Documents", 
    type="pdf",
    accept_multiple_files=True
)

workers = st.sidebar.slider(
    "Max Summarization Workers:", 
    1, 8, 2
)

# Main UI
st.title("📄 Document Summarization")
status = st.empty()
progress = st.empty()

start_button = st.button(
    "🚀 Start Processing Documents", 
    type="primary",
    disabled=st.session_state.processing_started,
    use_container_width=True
)

# Process uploaded files
if uploaded_files and not st.session_state.processing_started:
    st.session_state.uploaded_file_paths = []
    for uploaded_file in uploaded_files:
        file_path = Path(DOCS_FOLDER) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file_paths.append(str(file_path))
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded.")

# Core Processing Flow
if start_button and not st.session_state.processing_started:
    if not st.session_state.uploaded_file_paths:
        st.sidebar.warning("No files uploaded. Please upload PDFs.")
    else:
        st.session_state.processing_started = True
        st.session_state.summaries = []
        st.session_state.stream_containers = {}
        
        # Stage 1: Document Extraction
        status.info("⏳ Extracting documents...")
        try:
            st.session_state.extraction_results = process_documents()
            if not st.session_state.extraction_results:
                status.warning("⚠️ No documents found/extracted.")
                st.session_state.processing_started = False
            else:
                status.success(f"✅ Extracted {len(st.session_state.extraction_results)} documents.")
                
                # Stage 2: Summarization
                status.info("⏳ Summarizing documents...")
                progress.progress(0.5)
                
                try:
                    stream_components = batch_summarize_documents(
                        st.session_state.extraction_results, 
                        max_workers=workers
                    )
                    
                    # Create containers for each component before processing
                    for component in stream_components:
                        filename = Path(component.get('filename', 'Unknown')).name
                        comp_name = component.get('comp_name', 'Summary')
                        container_id = f"{filename}-{comp_name}"
                        st.session_state.stream_containers[container_id] = st.empty()
                    
                    # Process streams and update UI in real-time
                    for component in stream_components:
                        filename = Path(component.get('filename', 'Unknown')).name
                        comp_name = component.get('comp_name', 'Summary')
                        container_id = f"{filename}-{comp_name}"
                        container = st.session_state.stream_containers[container_id]
                        
                        if comp_name in component:
                            stream_generator = component[comp_name]
                            content_buffer = []
                            
                            # Handle resource_link component stream differently
                            if comp_name == 'resource_link':
                                for event in stream_generator:
                                    content_buffer.append(str(event))
                                    container.markdown("".join(content_buffer))
                            else:
                                # Handle regular component streams
                                for event in stream_generator:
                                    if event.type == "content-delta":
                                        delta_text = event.delta.message.content.text
                                        content_buffer.append(delta_text)
                                        container.markdown("".join(content_buffer))
                    
                    # Store the processed summaries
                    for component in stream_components:
                        st.session_state.summaries.append(component)
                    
                    status.success(f"✅ Processed {len(st.session_state.summaries)} summaries.")
                    progress.progress(1.0)
                    
                except Exception as e:
                    status.error(f"❌ Summarization error: {str(e)}")
                
                st.session_state.processing_started = False
                cleanup_files(st.session_state.uploaded_file_paths)
                st.session_state.uploaded_file_paths = []
                
        except Exception as e:
            status.error(f"❌ Extraction error: {str(e)}")
            st.session_state.processing_started = False
            cleanup_files(st.session_state.uploaded_file_paths)
            st.session_state.uploaded_file_paths = []

# Display Summaries
st.subheader("📝 Document Summaries")
if st.session_state.summaries:
    for i, summary in enumerate(st.session_state.summaries):
        filename = Path(summary.get('filename', f'Document {i+1}')).name
        comp_name = summary.get('comp_name', 'Summary')
        container_id = f"{filename}-{comp_name}"
        
        with st.expander(f"{comp_name.replace('_', ' ').title()} for: {filename}", expanded=True):
            if comp_name in summary:
                st.markdown(f"### {comp_name.replace('_', ' ').title()}")
                if container_id in st.session_state.stream_containers:
                    # The content is already displayed in the stream container
                    pass
                else:
                    st.write(summary[comp_name])
            else:
                st.info("No summary content available.")
else:
    st.info("Upload documents and click 'Start Processing' to see summaries here.")