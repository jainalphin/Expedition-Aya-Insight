import os

import streamlit as st
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import uuid
from streaming_generator import clear_upload_directory, process_pdf_links
from app.summarization.summarizer import DocumentSummarizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Aya Insight - AI-Powered Scientific Summarization", page_icon="📄", layout="wide")

# --- CUSTOM CSS ---
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

    /* --- Page Title (st.title) --- */
    h1[data-testid="stTitle"] {
        font-size: 2.2rem !important; /* Slightly reduced title size */
        font-weight: 700 !important;
        color: #1E88E5; /* A nice, vibrant blue */
        text-align: center;
        margin-bottom: 1.5rem;
        letter-spacing: -0.5px;
    }

    /* --- Sidebar Styling --- */
    [data-testid="stSidebarUserContent"] {
        padding-top: 1rem;
    }
    [data-testid="stSidebarUserContent"] .stMarkdown h3, /* For your sidebar subheader */
    [data-testid="stSidebarUserContent"] h3 {
        font-size: 1.25rem !important;
        color: #005A9C; /* Darker blue for sidebar title */
        font-weight: 600;
    }
    [data-testid="stSidebarUserContent"] .stButton>button {
        font-size: 0.9rem;
        border-radius: 6px;
        border: 1px solid #007bff;
        padding: 0.4rem 0.8rem;
    }
    [data-testid="stSidebarUserContent"] .stButton>button:hover {
        background-color: #007bff;
        color: white;
    }
    [data-testid="stSidebarUserContent"] .stFileUploader label {
        font-size: 0.95rem;
        font-weight: 500;
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

    /* --- Expander Styling --- */
    .stExpander {
        border: 1px solid #ddd !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
        background-color: #fdfdfd; /* Slightly off-white */
    }
    .stExpander summary {
        font-size: 1.1rem !important; /* Title of the expander */
        font-weight: 600 !important;
        color: #334E68; /* A muted, professional blue */
        padding: 0.6rem 0.8rem !important;
        border-radius: 8px 8px 0 0; /* Match top corners */
    }
    .stExpander summary:hover {
        background-color: #f0f4f8;
    }
    .stExpander div[data-testid="stExpanderDetails"] {
        padding: 0.5rem 1rem 1rem 1rem; /* Padding inside the expander */
    }


    /* --- Placeholders & Info/Status Boxes --- */
    .stAlert { /* Targets st.info, st.success, st.warning, st.error */
        font-size: 0.88rem;
        padding: 0.75rem;
        border-radius: 6px;
        border-width: 1px;
        border-left-width: 5px;
    }
    div[data-testid="stInfo"] {
        border-left-color: #17a2b8; /* Bootstrap info color */
        background-color: #e8f7f9;
    }
    div[data-testid="stSuccess"] {
        border-left-color: #28a745; /* Bootstrap success color */
        background-color: #eaf6eb;
    }
    div[data-testid="stWarning"] {
        border-left-color: #ffc107; /* Bootstrap warning color */
        background-color: #fff8e6;
    }
    div[data-testid="stError"] {
        border-left-color: #dc3545; /* Bootstrap error color */
        background-color: #fdecea;
    }

    /* Text within st.empty() before content loads, or generic st.text */
    [data-testid="stText"] {
        font-size: 0.88rem;
        color: #555;
        font-style: italic;
    }

    /* Progress/Timer text */
    .stApp [data-testid="stText"], .stApp [data-testid="stMarkdownContainer"] p:has(> code) { /* Targeting timer */
        font-size: 0.9rem;
        color: #007BFF; /* Make timer text blue */
        font-weight: 500;
    }

    /* Spinner color change (More advanced, might need JS or specific Streamlit features if this doesn't work well) */
    .stSpinner > div {
        border-top-color: #007BFF !important; /* Spinner color */
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

    /* --- Caption --- */
    .stCaption {
        font-size: 0.8rem;
        color: #777;
        text-align: center;
        margin-top: 2rem;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
# --- END CUSTOM CSS ---



# Initialize session state variables
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
    st.session_state.timer_placeholder = None  # Will be created when needed

with st.sidebar:
    st.markdown("<h3>Aya Multi-File Summary Tool 🚀</h3>", unsafe_allow_html=True)

    # Tab selection for input method
    input_method = st.radio("Choose input method:", ["Upload Files", "PDF URLs"], horizontal=True)

    pdf_links = []  # This will store all PDF links (from uploads or URLs)

    if input_method == "Upload Files":
        uploaded_files = st.file_uploader("Choose PDF files to analyze:", type="pdf", accept_multiple_files=True)

        # Convert uploaded files to temporary links
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    # Create temporary file
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)

                    # Generate unique filename
                    timestamp = int(time.time() * 1000)
                    temp_filename = f"{timestamp}_{uploaded_file.name}"
                    temp_path = os.path.join(temp_dir, temp_filename)

                    # Save uploaded file to temporary location
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    pdf_links.append(temp_path)

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

    else:  # PDF URLs
        st.markdown("**Enter PDF URLs (one per line):**")
        url_input = st.text_area(
            "PDF URLs:",
            height=150,
            placeholder="https://arxiv.org/pdf/1706.03762\nhttps://aclanthology.org/D19-3019.pdf\n...",
            help="Enter each PDF URL on a new line"
        )

        if url_input.strip():
            urls = [url.strip() for url in url_input.strip().split('\n') if url.strip()]
            valid_urls = []
            for url in urls:
                valid_urls.append(url)
                pdf_links.append(url)

    if st.button("🧹 Clear Upload Cache", help="Removes temporarily saved uploaded files and resets the app state."):
        try:
            clear_upload_directory()
            st.success("Temporary upload directory cleared.")
            st.session_state.file_placeholders = {}
            st.session_state.results = {}
            st.session_state.component_content = {}
            st.session_state.tasks_running = 0
            st.session_state.start_time = None
            if st.session_state.timer_placeholder:
                st.session_state.timer_placeholder.empty()
            st.session_state.timer_placeholder = None
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing directory: {e}")
            logger.error(f"Error during cache clear: {e}", exc_info=True)

    # Store pdf_links in session state for use in main app
    st.session_state.pdf_links = pdf_links

    summarize_button = st.button("✨ Summarize All Files",
                                 type="primary",
                                 key="summarize_all",
                                 disabled=st.session_state.tasks_running > 0 or not pdf_links,
                                 use_container_width=True)

# Main content area
st.title("📄✨ Aya Insight - AI-Powered Scientific Summarization")

# Main app description - using more markdown features for better look
st.markdown(
    """
    Welcome to **Aya Insight**! Your intelligent assistant for dissecting and understanding PDF documents.
    """
)

if st.session_state.timer_placeholder is None:
    st.session_state.timer_placeholder = st.empty()


def get_and_summarize_component_task(comp, update_queue):
    component_key = comp['comp_name']
    stream = comp[component_key]
    filename = comp['filename']
    try:
        chunk_count = 0
        if component_key == 'resource_link':  # Special handling for non-LLM stream
            for event in stream:
                update_queue.put(('chunk', filename, component_key, str(event)))
                chunk_count += 1
        else:
            # Handle regular component streams (LLM responses)
            for event in stream:
                if event.choices[0].delta.content:
                    delta_text = event.choices[0].delta.content
                    update_queue.put(('chunk', filename, component_key, delta_text))
                    chunk_count += 1

        if chunk_count == 0:  # If the stream was empty or yielded no actual content
            update_queue.put(
                ('comp_done', filename, component_key, "*No specific content generated for this section.*"))
        else:
            update_queue.put(
                ('comp_done', filename, component_key, None))  # None indicates success, content already streamed

        logger.info(f"[{filename}-{component_key}] Finished processing stream. Chunks: {chunk_count}")

    except Exception as e:
        logger.error(f"Error in component task for {filename}-{component_key}: {e}", exc_info=True)
        error_msg = f"_Error processing {component_key.replace('_', ' ').title()}: {str(e)[:100]}..._"
        update_queue.put(('comp_error', filename, component_key, error_msg))


def process_file_task(doc_data, update_queue):
    filename = doc_data.get('filename', f'unknown_file_{uuid.uuid4()}')

    try:
        logger.info(f"[{filename}] Starting process_file_task.")
        update_queue.put(('status', filename, None, "🔄 Initializing analysis engine..."))
        summarizer = DocumentSummarizer()

        update_queue.put(('status', filename, None, "⚙️ Generating content components..."))

        summarizer_iterator = summarizer._process_component(comp_data=doc_data)

        get_and_summarize_component_task(summarizer_iterator, update_queue)
        logger.info(f"[{filename}] Summary tasks completed submission and processing.")
        update_queue.put(('file_done', filename))  # Signal that this file's processing (all components) is done

    except Exception as e:
        logger.error(f"[{filename}] Critical error in process_file_task: {e}", exc_info=True)
        update_queue.put(('file_error', filename, f"Critical processing error: {str(e)[:150]}..."))


# Get PDF links from session state (set by the sidebar)
pdf_links = st.session_state.get('pdf_links', [])

if pdf_links:
    if not summarize_button and not st.session_state.tasks_running:
        st.info("Click 'Summarize All Files' in the sidebar to process.")

    if summarize_button:
        st.markdown("---")
        st.markdown("## Processing Document Insights...")

        st.session_state.start_time = time.time()
        if st.session_state.timer_placeholder is None:
            st.session_state.timer_placeholder = st.empty()

        st.session_state.results = {}
        st.session_state.file_placeholders = {}
        st.session_state.component_content = {}
        st.session_state.tasks_running = len(pdf_links)

        update_queue = Queue()
        component_info = {
            'summary': "📚 Summary",
        }

        default_layout_order = [
            ['summary'],
        ]

        layout_order = []
        all_available_components = set(component_info.keys())
        used_components = set()

        for row_template in default_layout_order:
            current_row = [comp_key for comp_key in row_template if comp_key in all_available_components]
            if current_row:
                layout_order.append(current_row)
                for comp_key in current_row:
                    used_components.add(comp_key)

        # Add any remaining components that weren't in the default layout
        if all_available_components - used_components:
            remaining_components_sorted = sorted(list(all_available_components - used_components))
            if remaining_components_sorted:
                 layout_order.append(remaining_components_sorted)  # Add them as a single row, sorted alphabetically

        # Create expandable sections for each PDF
        for file_index, pdf_link in enumerate(pdf_links):
            # Create an expander for each file, styled by CSS
            with st.expander(f"📄 {pdf_link}", expanded=True):
                status_ph = st.empty()
                status_ph.info("Queued for processing...")

                component_phs_dict = {}
                component_content_dict = {}

                if not layout_order and component_info:
                    st.markdown("### Summary Sections")
                    for comp_key in component_info.keys():
                        comp_name = component_info.get(comp_key, comp_key.replace('_', ' ').title())
                        st.markdown(f"#### {comp_name}")
                        component_phs_dict[comp_key] = st.empty()
                        component_content_dict[comp_key] = ""
                elif not component_info:
                    st.markdown("### Processing Output")
                    component_phs_dict['summary'] = st.empty()  # Generic placeholder
                    component_content_dict['summary'] = ""
                else:
                    # Render based on the dynamic layout_order
                    for row in layout_order:
                        if not row: continue  # Skip empty rows
                        cols = st.columns(len(row))
                        for i, comp_key in enumerate(row):
                            with cols[i]:
                                with st.container(border=True):
                                    comp_name = component_info.get(comp_key, comp_key.replace('_', ' ').title())
                                    st.markdown(f"### {comp_name}")
                                    component_phs_dict[comp_key] = st.empty()
                                    component_content_dict[comp_key] = ""

                st.session_state.file_placeholders[pdf_link] = {
                    'status': status_ph,
                    'components': component_phs_dict
                }
                st.session_state.component_content[pdf_link] = component_content_dict

        try:
            # Process PDF links instead of uploaded files
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
            if filename and filename in st.session_state.file_placeholders:
                process_executor.submit(process_file_task, result_data, update_queue)
                submitted_files.add(filename)
                st.session_state.file_placeholders[filename]['status'].info(f"⏳ Processing: {filename}...")
            else:
                logger.warning(f"Filename {filename} from extraction_results not found in placeholders or is None.")
                st.session_state.tasks_running -= 1  # Decrement if a file can't be processed

        files_done_processing = set()
        # Store component status for each file to determine overall file status
        component_statuses = {fname: {} for fname in submitted_files}
        active_spinners_markers = {fname: {ckey: True for ckey in st.session_state.component_content[fname]} for fname in submitted_files}

        while st.session_state.tasks_running > 0:
            if st.session_state.start_time is not None and st.session_state.timer_placeholder:
                elapsed_time = time.time() - st.session_state.start_time
                st.session_state.timer_placeholder.markdown(
                    f"⏳ **Overall Processing Time:** `{elapsed_time:.2f} seconds`")

            try:
                msg = update_queue.get(timeout=0.1)  # Timeout to allow UI updates and time checks
                msg_type = msg[0]
                filename = msg[1]  # Filename is always the second element

                if filename not in st.session_state.file_placeholders:
                    logger.warning(f"Received message for unknown/already-cleared file: {filename}. Type: {msg_type}")
                    update_queue.task_done()
                    continue

                file_placeholders = st.session_state.file_placeholders[filename]
                file_component_content = st.session_state.component_content[filename]

                if msg_type == 'chunk':
                    _, _, comp_key, text_delta = msg
                    if comp_key in file_component_content:
                        file_component_content[comp_key] += text_delta
                        display_text = file_component_content[comp_key] + '<span class="blinking-cursor">▌</span>'
                        if comp_key in file_placeholders['components']:
                            file_placeholders['components'][comp_key].markdown(
                                display_text, unsafe_allow_html=True
                            )

                elif msg_type == 'comp_done':
                    _, _, comp_key, final_message = msg
                    component_statuses.setdefault(filename, {})[comp_key] = 'done'
                    active_spinners_markers.get(filename, {}).pop(comp_key, None)  # Remove marker

                    if comp_key in file_placeholders['components']:
                        final_content = file_component_content.get(comp_key, "")
                        if final_message:
                            if final_content and not final_content.endswith("\n\n"):
                                final_content += "\n\n"
                            final_content += f"*{final_message}*"
                        file_placeholders['components'][comp_key].markdown(final_content)

                elif msg_type == 'comp_error':
                    _, _, comp_key, error_msg = msg
                    component_statuses.setdefault(filename, {})[comp_key] = 'error'
                    active_spinners_markers.get(filename, {}).pop(comp_key, None)  # Remove marker

                    if comp_key in file_placeholders['components']:
                        file_placeholders['components'][comp_key].error(f"⚠️ {error_msg}")

                elif msg_type == 'status':  # General status update for the file
                    _, _, _, status_msg = msg  # comp_key is None for general file status
                    file_placeholders['status'].info(f"⏳ {status_msg}")  # Styled by CSS

                elif msg_type == 'file_done':
                    if filename not in files_done_processing:
                        files_done_processing.add(filename)
                        st.session_state.tasks_running -= 1

                        final_file_status = 'success'  # Assume success initially
                        file_comp_statuses = component_statuses.get(filename, {})

                        if not file_comp_statuses:
                            if any(v == 'error' for v in file_comp_statuses.values()):
                                final_file_status = 'error'
                            else:
                                final_file_status = 'warning_nodata'
                        elif any(v == 'error' for v in file_comp_statuses.values()):
                            final_file_status = 'error'
                        elif not any(file_component_content.get(k, "").strip() and file_component_content.get(k, "").strip() != "*No specific content generated for this section.*" for k in file_comp_statuses if file_comp_statuses.get(k) == 'done'):
                            final_file_status = 'warning_nodata'

                        status_ph = file_placeholders['status']
                        if final_file_status == 'success':
                            status_ph.success(f"✅ Summarization Complete for {filename}!")
                            st.session_state.results[filename] = True
                        elif final_file_status == 'warning_nodata':
                            status_ph.warning(f"⚠️ {filename} processed, but some sections have limited or no content.")
                            st.session_state.results[filename] = "warning"
                        else:  # 'error'
                            status_ph.error(f"❌ {filename} completed with errors in some sections.")
                            st.session_state.results[filename] = False

                        # Ensure all component spinners are removed for this file
                        for comp_key_iter, placeholder in file_placeholders['components'].items():
                            if active_spinners_markers.get(filename, {}).get(
                                    comp_key_iter):  # If spinner was still active
                                final_c_content = file_component_content.get(comp_key_iter, "")
                                placeholder.markdown(final_c_content)  # Update with final content

                elif msg_type == 'file_error':
                    _, critical_error_msg = msg
                    if filename not in files_done_processing:
                        files_done_processing.add(filename)
                        st.session_state.tasks_running -= 1
                        file_placeholders['status'].error(
                            f"❌ Critical Error processing {filename}: {critical_error_msg}")
                        st.session_state.results[filename] = False
                        for comp_key_iter, placeholder in file_placeholders['components'].items():
                            if active_spinners_markers.get(filename, {}).get(comp_key_iter):
                                placeholder.markdown("_Processing halted due to critical file error._")

                update_queue.task_done()

            except Empty:
                pass
            except Exception as loop_exc:
                logger.error(f"Error in main UI update loop: {loop_exc}", exc_info=True)
                st.error(f"A critical error occurred while updating the UI: {loop_exc}")
                st.session_state.tasks_running = 0

        if st.session_state.start_time is not None and st.session_state.timer_placeholder:
            final_elapsed_time = time.time() - st.session_state.start_time
            st.session_state.timer_placeholder.success(f"🎉 All processing finished in {final_elapsed_time:.2f} seconds!")

        st.session_state.start_time = None
        process_executor.shutdown(wait=False)

else:
    # Show message when no PDFs are available
    st.info("👈 Please upload PDF files or provide PDF URLs using the sidebar to get started.")