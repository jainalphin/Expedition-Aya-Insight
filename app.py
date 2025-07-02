import os
import shutil
import urllib.parse
import pandas as pd
import streamlit as st
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import uuid
import streamlit.components.v1 as components
from streaming_generator import clear_upload_directory, process_pdf_links
from app.summarization.summarizer import DocumentSummarizer

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(threadName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

if os.path.isdir("temp_uploads"):
    shutil.rmtree("temp_uploads")
st.set_page_config(page_title="AI-Powered Scientific Summarization", layout="wide")


custom_css = """
<style>
    /* --- Overall Font & Body --- */
    body {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif; /* Modern font stack */
        color: #333; /* Darker grey for better readability */
    }

    .main .block-container { /* Targets the main content area */
        padding-top: 2rem; /* Add some space at the top */
    }

    /* --- Markdown Content Styling --- */
    .stMarkdown {
        font-size: 0.92rem; /* Base for markdown text */
        line-height: 1.65;
        color: #4F4F4F; /* Slightly softer than pure black */
    }
    .stMarkdown p, .stMarkdown li {
        margin-bottom: 0.6rem; /* Space between paragraphs/list items */
    }

    /* Markdown Headers in main content */
    .stMarkdown h2 { /* Used by st.subheader and ## markdown */
        font-size: 1.6rem;
        color: #2c3e50; /* Dark blue-grey */
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.4rem;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        font-weight: 600;
    }

    /* Component Titles (e.g., "Basic Paper Information") inside expanders/columns */
    .stMarkdown h3 { /* Used by st.subheader in columns/expanders and ### markdown */
        font-size: 1.20rem; /* Smaller for component titles */
        color: #007BFF; /* Bright blue for emphasis */
        margin-top: 0rem; /* REVISED: Removed margin-top for tighter spacing */
        margin-bottom: 0.6rem;
        font-weight: 600;
        /* border-left: 3px solid #007BFF; */
        /* padding-left: 0.5rem; */
    }
    .stMarkdown h4 {
        font-size: 1.05rem;
        color: #333;
        font-weight: 600;
    }

    /* Progress/Timer text */
    .stApp [data-testid="stText"], .stApp [data-testid="stMarkdownContainer"] p:has(> code) { /* Targeting timer */
        font-size: 0.9rem;
        color: #007BFF; /* Make timer text blue */
        font-weight: 500;
    }

    /* REVISED: Styling for the custom blinking cursor span, directly controlled by Python */
    .blinking-cursor {
        animation: blink 1s step-end infinite;
        color: #007BFF;
        display: inline-block; /* Ensures the cursor is visible next to text */
    }

    @keyframes blink {
        from, to { opacity: 1; }
        50% { opacity: 0; }
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
# --- Helper Functions for Status Display ---

def get_status_display(filename):
    """Get status display text for a file."""
    if filename not in st.session_state.file_status_tracking:
        return "Pending"

    status = st.session_state.file_status_tracking[filename]
    if status == "processing":
        return "Processing"
    elif status == "success":
        return "Complete"
    elif status == "error":
        return "Error"
    elif status == "warning":
        return "Warning"
    else:
        return "Pending"


# --- UI and Data Handling ---

def create_navigation_table_dataframe(pdf_links):
    """Create a navigation dataframe with a conditional download link for each summary."""
    if not pdf_links:
        return None

    table_data = []
    for i, filename in enumerate(pdf_links):
        # Extract a clean display name for the document
        if "https" in filename:
            display_name = filename
        else:
            display_name = filename.split("/")[-1].split("_")[-1] if "_" in filename else filename.split("/")[-1]

        status_text = get_status_display(display_name)
        summary_link = f'<a href="#expander_{i}" target="_self">Go to Summary</a>'
        download_link = "Pending..."
        current_status = st.session_state.file_status_tracking.get(display_name)

        # Show download link only if processing is complete
        if current_status in ["success", "warning"]:
            summary_content = generate_download_content(display_name)
            encoded_content = urllib.parse.quote(summary_content)
            safe_filename = display_name.replace('.pdf', '').replace(' ', '_')
            download_filename = f"{safe_filename}_summary.txt"
            download_link = f'<a href="data:text/plain;charset=utf-8,{encoded_content}" download="{download_filename}" target="_blank">Download Summary</a>'

        elif current_status == "error":
            download_link = "<i>Failed</i>"

        table_data.append({
            "Document Name": display_name,
            "Status": status_text,
            "Link": summary_link,
            "Download": download_link
        })

    return pd.DataFrame(table_data)


def update_navigation_table(pdf_links):
    """Update the navigation table by rendering it as HTML."""
    if st.session_state.navigation_table_placeholder:
        df = create_navigation_table_dataframe(pdf_links)
        if df is not None and not df.empty:
            # Render dataframe as a simple HTML table to allow for clickable links
            html_table = df.to_html(
                escape=False,
                index=False,
                justify='center'
            )
            st.session_state.navigation_table_placeholder.markdown(
                html_table,
                unsafe_allow_html=True
            )
        else:
            st.session_state.navigation_table_placeholder.empty()


def generate_download_content(display_name):
    """Generates a single string with all summary content for download."""
    content = [f"Summary for: {display_name}\n\n"]
    component_content = st.session_state.component_content.get(display_name, {})
    text = component_content.get('summary', "*No content generated for this section.*")
    content.append(f"--- Summary ---\n\n{text}\n\n")
    return "".join(content)


# --- Initialize Session State ---
if 'file_placeholders' not in st.session_state:
    st.session_state.file_placeholders = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'component_content' not in st.session_state:
    st.session_state.component_content = {}
if 'tasks_running' not in st.session_state:
    st.session_state.tasks_running = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'timer_placeholder' not in st.session_state:
    st.session_state.timer_placeholder = None
if 'file_status_tracking' not in st.session_state:
    st.session_state.file_status_tracking = {}
if 'navigation_table_placeholder' not in st.session_state:
    st.session_state.navigation_table_placeholder = None


# --- Sidebar UI ---
with st.sidebar:
    st.markdown("<h2>Multi-File Summary Tool</h2>", unsafe_allow_html=True)
    input_method = st.radio("Choose input method:", ["Upload Files", "PDF URLs"], horizontal=True)
    pdf_links = []

    if input_method == "Upload Files":
        uploaded_files = st.file_uploader("Choose PDF files to analyze:", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    timestamp = int(time.time() * 1000)
                    temp_filename = f"{timestamp}_{uploaded_file.name}"
                    temp_path = os.path.join(temp_dir, temp_filename)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_links.append(temp_path)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
    else:
        pdf_links = []
        st.markdown("**Enter PDF URLs (one per line):**")
        url_input = st.text_area(
            "PDF URLs:",
            height=150,
            placeholder="https://arxiv.org/pdf/1706.03762\nhttps://aclanthology.org/D19-3019.pdf\n...",
            help="Enter each PDF URL on a new line"
        )
        urls = [url.strip() for url in url_input.strip().split('\n') if url.strip()]
        for url in urls:
            pdf_links.append(url)

    if st.button("Clear Upload Cache", help="Removes temporarily saved uploaded files and resets the app state."):
        try:
            clear_upload_directory()
            st.success("Temporary upload directory cleared.")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing directory: {e}")
            logger.error(f"Error during cache clear: {e}", exc_info=True)

    st.session_state.pdf_links = pdf_links

    summarize_button = st.button("Summarize All Files",
                                 type="primary",
                                 key="summarize_all",
                                 disabled=st.session_state.tasks_running > 0 or not pdf_links,
                                 use_container_width=True)

# --- Main Content Area ---

st.title("AI-Powered Scientific Summarization")
st.markdown(
    """
    Welcome! This tool provides an intelligent assistant for dissecting and understanding PDF documents.
    """
)

navigation_container = st.container()
timer_container = st.container()

if st.session_state.navigation_table_placeholder is None:
    with navigation_container:
        st.session_state.navigation_table_placeholder = st.empty()
if st.session_state.timer_placeholder is None:
    with timer_container:
        st.session_state.timer_placeholder = st.empty()


# --- Backend Processing Functions ---

def get_and_summarize_component_task(comp, update_queue):
    component_key = comp['comp_name']
    stream = comp[component_key]
    filename = comp['filename']
    try:
        chunk_count = 0
        for event in stream:
            if event.choices[0].delta.content:
                delta_text = event.choices[0].delta.content
                update_queue.put(('chunk', filename, component_key, delta_text))
                chunk_count += 1
        if chunk_count == 0:
            update_queue.put(('comp_done', filename, component_key, "*No specific content generated for this section.*"))
        else:
            update_queue.put(('comp_done', filename, component_key, None))
        logger.info(f"[{filename}-{component_key}] Finished processing stream. Chunks: {chunk_count}")
    except Exception as e:
        logger.error(f"Error in component task for {filename}-{component_key}: {e}", exc_info=True)
        error_msg = f"Error processing {component_key.replace('_', ' ').title()}: {str(e)[:100]}..."
        update_queue.put(('comp_error', filename, component_key, error_msg))


def process_file_task(doc_data, update_queue):
    filename = doc_data.get('filename', f'unknown_file_{uuid.uuid4()}')
    try:
        logger.info(f"[{filename}] Starting process_file_task.")
        update_queue.put(('status', filename, None, "Initializing analysis engine..."))
        summarizer = DocumentSummarizer()
        update_queue.put(('status', filename, None, "Generating content components..."))
        summarizer_iterator = summarizer._process_component(comp_data=doc_data)
        get_and_summarize_component_task(summarizer_iterator, update_queue)
        logger.info(f"[{filename}] Summary tasks completed.")
        update_queue.put(('file_done', filename))
    except Exception as e:
        logger.error(f"[{filename}] Critical error in process_file_task: {e}", exc_info=True)
        update_queue.put(('file_error', filename, f"Critical processing error: {str(e)[:150]}..."))


# --- Main Application Logic ---
pdf_links = st.session_state.get('pdf_links', [])

if pdf_links:
    for filename in pdf_links:
        display_name = filename.split("/")[-1].split("_")[-1] if "https" not in filename else filename
        if display_name not in st.session_state.file_status_tracking:
            st.session_state.file_status_tracking[display_name] = "pending"

    update_navigation_table(pdf_links)

    if not summarize_button and not st.session_state.tasks_running:
        st.info("Click 'Summarize All Files' in the sidebar to process.")

    if summarize_button:
        st.session_state.start_time = time.time()
        st.session_state.results = {}
        st.session_state.file_placeholders = {}
        st.session_state.component_content = {}
        st.session_state.tasks_running = len(pdf_links)

        for filename in pdf_links:
            display_name = filename.split("/")[-1].split("_")[-1] if "https" not in filename else filename
            st.session_state.file_status_tracking[display_name] = "processing"

        update_navigation_table(pdf_links)
        update_queue = Queue()
        component_info = {'summary': "Summary"}

        for file_index, filename in enumerate(pdf_links):
            display_name = filename.split("/")[-1].split("_")[-1] if "https" not in filename else filename
            st.markdown(f'<div id="expander_{file_index}"></div>', unsafe_allow_html=True)
            with st.expander(f"Document: {display_name}", expanded=True):
                status_ph = st.empty()
                status_ph.info("Queued for processing...")
                component_phs_dict = {}
                component_content_dict = {}
                with st.container(border=True):
                    st.markdown(f"### {component_info['summary']}")
                    component_phs_dict['summary'] = st.empty()
                    component_content_dict['summary'] = ""
                st.session_state.file_placeholders[display_name] = {'status': status_ph, 'components': component_phs_dict}
                st.session_state.component_content[display_name] = component_content_dict

        try:
            extraction_results = process_pdf_links(pdf_links)
        except Exception as e:
            st.error(f"Critical error during initial PDF processing: {e}")
            logger.error(f"Error during process_pdf_links: {e}", exc_info=True)
            st.session_state.tasks_running = 0
            st.stop()

        process_executor = ThreadPoolExecutor(max_workers=min(8, len(pdf_links)))
        submitted_files = set()
        for result_data in extraction_results:
            filename = result_data.get('filename')
            display_name = filename.split("/")[-1].split("_")[-1] if "https" not in filename else filename
            if display_name and display_name in st.session_state.file_placeholders:
                process_executor.submit(process_file_task, result_data, update_queue)
                submitted_files.add(display_name)
                st.session_state.file_placeholders[display_name]['status'].info(f"Processing: {display_name}...")
            else:
                logger.warning(f"Filename {display_name} from extraction not found in placeholders.")
                st.session_state.tasks_running -= 1

        files_done_processing = set()
        component_statuses = {fname: {} for fname in submitted_files}
        last_table_update = 0
        table_update_interval = 2.0

        while st.session_state.tasks_running > 0:
            current_time = time.time()
            if st.session_state.start_time is not None and st.session_state.timer_placeholder:
                elapsed_time = current_time - st.session_state.start_time
                with timer_container:
                    st.session_state.timer_placeholder.markdown(f"**Overall Processing Time:** `{elapsed_time:.2f} seconds`")
            try:
                msg = update_queue.get(timeout=0.1)
                msg_type, filename, *payload = msg
                display_name = filename.split("/")[-1].split("_")[-1] if "https" not in filename else filename

                if display_name not in st.session_state.file_placeholders:
                    logger.warning(f"Received message for unknown file: {display_name}.")
                    update_queue.task_done()
                    continue

                file_placeholders = st.session_state.file_placeholders[display_name]
                file_component_content = st.session_state.component_content[display_name]
                table_needs_update = False

                if msg_type == 'chunk':
                    comp_key, text_delta = payload
                    if comp_key in file_component_content:
                        file_component_content[comp_key] += text_delta
                        display_text = file_component_content[comp_key] + '▌'
                        file_placeholders['components'][comp_key].markdown(display_text, unsafe_allow_html=True)
                elif msg_type == 'comp_done':
                    comp_key, final_message = payload
                    component_statuses.setdefault(display_name, {})[comp_key] = 'done'
                    final_content = file_component_content.get(comp_key, "")
                    if final_message:
                        final_content += f"\n\n*{final_message}*"
                    file_placeholders['components'][comp_key].markdown(final_content)
                elif msg_type == 'comp_error':
                    comp_key, error_msg = payload
                    component_statuses.setdefault(display_name, {})[comp_key] = 'error'
                    file_placeholders['components'][comp_key].error(error_msg)
                elif msg_type == 'status':
                    _, status_msg = payload
                    file_placeholders['status'].info(status_msg)
                elif msg_type == 'file_done':
                    if display_name not in files_done_processing:
                        files_done_processing.add(display_name)
                        st.session_state.tasks_running -= 1
                        final_file_status = 'success'
                        if any(v == 'error' for v in component_statuses.get(display_name, {}).values()):
                            final_file_status = 'error'
                        status_ph = file_placeholders['status']
                        if final_file_status == 'success':
                            status_ph.success(f"Summarization Complete for {display_name}!")
                            st.session_state.file_status_tracking[display_name] = "success"
                        else:
                            status_ph.error(f"Completed with errors for {display_name}.")
                            st.session_state.file_status_tracking[display_name] = "error"
                        table_needs_update = True
                elif msg_type == 'file_error':
                    critical_error_msg, = payload
                    if display_name not in files_done_processing:
                        files_done_processing.add(display_name)
                        st.session_state.tasks_running -= 1
                        file_placeholders['status'].error(f"Critical Error for {display_name}: {critical_error_msg}")
                        st.session_state.file_status_tracking[display_name] = "error"
                        table_needs_update = True

                if table_needs_update and (current_time - last_table_update >= table_update_interval):
                    with navigation_container:
                        update_navigation_table(pdf_links)
                    last_table_update = current_time
                update_queue.task_done()
            except Empty:
                pass
            except Exception as loop_exc:
                logger.error(f"Error in main UI update loop: {loop_exc}", exc_info=True)
                st.error(f"A critical error occurred while updating the UI: {loop_exc}")
                st.session_state.tasks_running = 0

        if st.session_state.start_time is not None and st.session_state.timer_placeholder:
            final_elapsed_time = time.time() - st.session_state.start_time
            with timer_container:
                st.session_state.timer_placeholder.success(f"All processing finished in {final_elapsed_time:.2f} seconds!")
        with navigation_container:
            update_navigation_table(pdf_links)
        st.session_state.start_time = None
        process_executor.shutdown(wait=False)
else:
    st.info("Please upload PDF files or provide PDF URLs using the sidebar to get started.")