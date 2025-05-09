import streamlit as st
import os
import logging
from pathlib import Path
from typing import List
import time
import random
import threading

# Import backend functions
from doc_processor_backend import (
    process_documents,
    batch_summarize_documents,
    # process_stream_components, # This was not used in the original selection's logic
    DOCS_FOLDER
)

# Setup & Configuration
Path(DOCS_FOLDER).mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

# Streamlit UI Configuration
st.set_page_config(layout="wide", page_title="Document Summarization")

# Custom CSS for creative loader
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
.loader-creative {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin: 0 auto;
}
.loader-dots {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 1em;
}
.loader-dot {
    width: 22px;
    height: 22px;
    margin: 0 6px;
    border-radius: 50%;
    background: #2596be;
    opacity: 0.7;
    animation: bounce 1.2s infinite;
}
.loader-dot:nth-child(2) { animation-delay: 0.2s; background: #43c6ac; }
.loader-dot:nth-child(3) { animation-delay: 0.4s; background: #fbb034; }
.loader-dot:nth-child(4) { animation-delay: 0.6s; background: #fd1d1d; }
@keyframes bounce {
    0%, 100% { transform: translateY(0);}
    50% { transform: translateY(-18px);}
}
.loader-book {
    width: 60px;
    height: 40px;
    position: relative;
    margin-bottom: 1em;
}
.loader-book .book {
    width: 60px;
    height: 40px;
    background: #fff;
    border: 2px solid #2596be;
    border-radius: 6px 6px 10px 10px;
    position: absolute;
    left: 0;
    top: 0;
    z-index: 1;
    box-shadow: 0 2px 8px #2596be22;
}
.loader-book .page {
    width: 10px;
    height: 32px;
    background: #e3f2fd;
    position: absolute;
    top: 4px;
    left: 25px;
    border-radius: 2px;
    animation: flip 1.2s infinite;
    z-index: 2;
}
@keyframes flip {
    0% { transform: rotateY(0deg);}
    40% { transform: rotateY(180deg);}
    100% { transform: rotateY(0deg);}
}
.pulse {
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.3; }
    100% { opacity: 1; }
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
if 'loading_messages' not in st.session_state:
    st.session_state.loading_messages = [
        "Analyzing document structure...",
        "Extracting meaningful content...",
        "Parsing document sections...",
        "Preparing for summarization...",
        "Connecting to AI models...",
        "Organizing document insights...",
        "Flipping through digital pages...",
        "Highlighting key concepts...",
        "Summoning the summary wizard...",
        "Brewing up concise insights...",
        "Turning pages at lightning speed...",
        "Scanning for hidden gems...",
        "Synthesizing knowledge nuggets...",
        "Mapping document mindspace..."
    ]
# Removed 'loader_animation_index' as it's not used in the new loader

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
status_message_placeholder = st.empty()
progress_bar_placeholder = st.empty()
loader_placeholder = st.empty() # Single placeholder for the loader

# --- Creative Loader HTML ---
def creative_loader_html(phase_message="Processing..."):
    loader_html = f"""
    <div class="loader-creative">
        <div class="loader-dots">
            <div class="loader-dot"></div>
            <div class="loader-dot"></div>
            <div class="loader-dot"></div>
            <div class="loader-dot"></div>
        </div>
        <div class="loader-book">
            <div class="book"></div>
            <div class="page"></div>
        </div>
        <div style="text-align:center; font-size:1.1em; color:#2596be; margin-top:0.5em;" class="pulse">
            <b>{phase_message}</b>
        </div>
    </div>
    """
    return loader_html

# Thread control for loader
if 'stop_loading_thread_event' not in st.session_state:
    st.session_state.stop_loading_thread_event = threading.Event()
if 'loading_thread_instance' not in st.session_state:
    st.session_state.loading_thread_instance = None

def start_loader_thread():
    st.session_state.stop_loading_thread_event.clear()
    
    def cycle_loader_messages():
        i = 0
        emojis = ["📄", "🔍", "🧠", "✨", "📚", "📝", "⚡", "🔎", "🧩", "🧙‍♂️", "🧪", "📖", "💡", "🗂️"]
        while not st.session_state.stop_loading_thread_event.is_set():
            message = st.session_state.loading_messages[i % len(st.session_state.loading_messages)]
            emoji = emojis[i % len(emojis)]
            full_message = f"{emoji} {message}"
            try:
                # Ensure we are calling markdown on the placeholder from the main thread's context
                # This is generally okay if the placeholder object itself is stable.
                loader_placeholder.markdown(creative_loader_html(full_message), unsafe_allow_html=True)
            except Exception as e:
                logger.warning(f"Error updating loader placeholder from thread: {e}")
                # This might happen if the placeholder is cleared while the thread is trying to update it.
                break 
            time.sleep(2.1)
            i += 1
        # Attempt to clear the placeholder once the loop is done, if not already cleared.
        try:
            loader_placeholder.empty()
        except Exception:
            pass # Placeholder might have been cleared by the main thread already.

    st.session_state.loading_thread_instance = threading.Thread(target=cycle_loader_messages)
    st.session_state.loading_thread_instance.start()

def stop_loader_thread():
    st.session_state.stop_loading_thread_event.set()
    if st.session_state.loading_thread_instance and st.session_state.loading_thread_instance.is_alive():
        st.session_state.loading_thread_instance.join(timeout=3)
    loader_placeholder.empty() # Ensure cleared from main thread
    st.session_state.loading_thread_instance = None

# Process uploaded files (runs on every interaction if files are uploaded and not processing)
if uploaded_files and not st.session_state.processing_started:
    st.session_state.uploaded_file_paths = []
    for uploaded_file in uploaded_files:
        file_path = Path(DOCS_FOLDER) / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file_paths.append(str(file_path))
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded.")
    # No rerun here, just updates paths and shows success.

# Button and Core Processing Flow
if not st.session_state.processing_started:
    # Clear previous messages if returning to this state
    status_message_placeholder.empty()
    progress_bar_placeholder.empty()
    loader_placeholder.empty()

    if st.button("🚀 Start Processing Documents", type="primary", use_container_width=True):
        if not st.session_state.uploaded_file_paths:
            st.sidebar.warning("No files uploaded. Please upload PDFs.")
        else:
            st.session_state.processing_started = True
            st.session_state.summaries = []
            st.session_state.stream_containers = {}

if st.session_state.processing_started:
    # This block runs after the button is clicked and st.experimental_rerun()
    # The button is not rendered due to the `if not st.session_state.processing_started:` condition above.
    
    start_loader_thread() # Start the unified loader

    try:
        # Stage 1: Document Extraction
        status_message_placeholder.info("⏳ Extracting documents...")
        # Loader is running in its thread.

        st.session_state.extraction_results = process_documents()
        
        if not st.session_state.extraction_results:
            status_message_placeholder.warning("⚠️ No documents found/extracted.")
            # This will lead to the finally block which stops the loader and resets state.
            raise Exception("No documents extracted to halt processing.") 
        
        status_message_placeholder.success(f"✅ Extracted {len(st.session_state.extraction_results)} documents.")
        progress_bar_placeholder.progress(0.25) # Initial progress

        # Stage 2: Summarization
        status_message_placeholder.info("⏳ Preparing summaries...") # Updated message
        # Loader thread continues to run.

        stream_components = batch_summarize_documents(
            st.session_state.extraction_results,
            max_workers=workers
        )
        progress_bar_placeholder.progress(0.5) # Progress after batch call

        # Create containers for each component before processing streams
        for component_data in stream_components:
            filename = Path(component_data.get('filename', 'Unknown')).name
            comp_name = component_data.get('comp_name', 'Summary')
            container_id = f"{filename}-{comp_name}"
            st.session_state.stream_containers[container_id] = st.empty()

        status_message_placeholder.info("⏳ Generating summaries stream by stream...")
        total_components = len(stream_components) if stream_components else 0
        processed_components_count = 0

        for component_data in stream_components:
            filename = Path(component_data.get('filename', 'Unknown')).name
            comp_name = component_data.get('comp_name', 'Summary')
            container_id = f"{filename}-{comp_name}"
            container = st.session_state.stream_containers[container_id]

            if comp_name in component_data:
                stream_generator = component_data[comp_name]
                content_buffer = []
                
                # Stream content into its container
                if comp_name == 'resource_link':
                    for event in stream_generator:
                        content_buffer.append(str(event))
                        container.markdown("".join(content_buffer))
                else:
                    for event in stream_generator:
                        if hasattr(event, 'type') and event.type == "content-delta": # Check for attribute
                            if hasattr(event.delta, 'message') and hasattr(event.delta.message.content, 'text'):
                                delta_text = event.delta.message.content.text
                                content_buffer.append(delta_text)
                                container.markdown("".join(content_buffer))
                        elif isinstance(event, str): # Fallback for simpler stream types
                             content_buffer.append(event)
                             container.markdown("".join(content_buffer))


                # Original logic for st.session_state.summaries:
                # It appends the component with the generator. This is kept as per original design
                # unless specified to change how summaries are stored.
                st.session_state.summaries.append(component_data)
            
            processed_components_count += 1
            if total_components > 0:
                progress_bar_placeholder.progress(0.5 + (processed_components_count / total_components) * 0.5)

        if not stream_components:
             status_message_placeholder.info("ℹ️ No summary components to process.")
        else:
            status_message_placeholder.success(f"✅ All {len(stream_components)} summary components processed.")
        progress_bar_placeholder.progress(1.0)

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        # Avoid overwriting "No documents extracted" warning with a generic error.
        if "No documents extracted" not in str(e):
            status_message_placeholder.error(f"❌ An error occurred: {str(e)}")
        # Loader and state reset will be handled in finally.
    
    finally:
        stop_loader_thread() # Stop and clear the loader
        st.session_state.processing_started = False # Reset processing state
        cleanup_files(st.session_state.uploaded_file_paths)
        st.session_state.uploaded_file_paths = []
        # No st.experimental_rerun() here, so the final status/results are shown.
        # Button will reappear on next interaction if needed (e.g., new file upload).

# Display Summaries
if st.session_state.summaries:
    st.subheader("📝 Document Summaries")
    for i, summary_component in enumerate(st.session_state.summaries):
        filename = Path(summary_component.get('filename', f'Document {i+1}')).name
        comp_name = summary_component.get('comp_name', 'Summary')
        container_id = f"{filename}-{comp_name}"

        with st.expander(f"{comp_name.replace('_', ' ').title()} for: {filename}", expanded=True):
            # Content is expected to be in stream_containers, which were populated during streaming.
            # If for some reason it's not (e.g. error before streaming to container but component added to summaries),
            # this logic might need adjustment based on how `summary_component[comp_name]` should behave post-stream.
            # The original code had a fallback:
            # if container_id in st.session_state.stream_containers:
            #     # The content is already displayed in the stream container (by being written to st.empty())
            #     # So, nothing to do here to re-display, it's already there.
            #     pass
            # else:
            #    st.write(summary_component[comp_name]) # This might write an exhausted generator.
            
            # For simplicity, we assume the containers hold the displayed content.
            # If `st.session_state.stream_containers[container_id]` was an `st.empty()` that got content,
            # that content persists. The expander just organizes it.
            # If `summary_component[comp_name]` holds the final string content (it doesn't with current logic),
            # then `st.markdown(summary_component[comp_name])` would be appropriate here.
            # Given the current setup, the content is already in the placeholders.
            # We can add a title within the expander.
            st.markdown(f"### {comp_name.replace('_', ' ').title()}")
            if container_id not in st.session_state.stream_containers:
                 # This case implies the component was in summaries but not streamed to a container.
                 # Or, if stream_containers are cleared/reinstantiated differently.
                 # Fallback to trying to display from summary_component if available and not a generator.
                content_to_display = summary_component.get(comp_name)
                if isinstance(content_to_display, str):
                    st.markdown(content_to_display)
                elif content_to_display is not None:
                     st.info("Summary content was streamed directly. See above if populated.")
                else:
                    st.info("No summary content available for this component.")
            # If container_id IS in st.session_state.stream_containers, its content is already visible on the page.
            # The expander just groups it. No need to re-draw.

