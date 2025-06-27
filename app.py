import os

import pandas as pd
import streamlit as st
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import uuid
from streaming_generator import clear_upload_directory, process_pdf_links
from app.summarization.summarizer import DocumentSummarizer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(threadName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Aya Insight - AI-Powered Scientific Summarization", page_icon="📄", layout="wide")

# --- CUSTOM CSS ---

# 2. Add the enhanced CSS to your existing custom_css
custom_css = """
<style>
    /* Your existing CSS... */

    /* Enhanced table styling for dataframes */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    .stDataFrame > div {
        border-radius: 8px;
    }

    /* Style for the dataframe headers */
    .stDataFrame thead th {
        background: linear-gradient(135deg, #1E88E5, #42A5F5) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 12px 16px !important;
    }

    /* Style for dataframe rows */
    .stDataFrame tbody td {
        border-bottom: 1px solid #f0f0f0 !important;
        padding: 10px 16px !important;
    }

    .stDataFrame tbody tr:hover {
        background-color: #f8f9fa !important;
    }

    .stDataFrame tbody tr:last-child td {
        border-bottom: none !important;
    }

    /* Remove the old nav-table CSS since we're using dataframes now */
</style>
"""

# Enhanced CSS for better table styling
enhanced_css = """
<style>
    /* Navigation table container styling */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    .stDataFrame > div {
        border-radius: 8px;
    }

    /* Style for the dataframe headers */
    .stDataFrame thead th {
        background: linear-gradient(135deg, #1E88E5, #42A5F5) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 12px 16px !important;
        text-align: center !important;
    }

    /* Style for dataframe rows */
    .stDataFrame tbody td {
        border-bottom: 1px solid #f0f0f0 !important;
        padding: 10px 16px !important;
        vertical-align: middle !important;
    }

    .stDataFrame tbody tr:hover {
        background-color: #f8f9fa !important;
        transition: background-color 0.2s ease;
    }

    .stDataFrame tbody tr:last-child td {
        border-bottom: none !important;
    }

    /* Status-specific styling */
    .stDataFrame tbody td:nth-child(2) {
        text-align: center !important;
        font-weight: 500;
    }

    .stDataFrame tbody td:nth-child(3) {
        text-align: center !important;
        font-style: italic;
        color: #666;
    }
</style>
"""


def get_status_display(filename):
    """Get status display information for a file"""
    if filename not in st.session_state.file_status_tracking:
        return "⏳ Pending"

    status = st.session_state.file_status_tracking[filename]
    if status == "processing":
        return "🔄 Processing"
    elif status == "success":
        return "✅ Complete"
    elif status == "error":
        return "❌ Error"
    elif status == "warning":
        return "⚠️ Warning"
    else:
        return "⏳ Pending"


def get_progress_info(filename):
    """Get progress information for a file"""
    if filename not in st.session_state.file_status_tracking:
        return "Waiting to start..."

    status = st.session_state.file_status_tracking[filename]
    if status == "processing":
        return "Processing components..."
    elif status == "success":
        return "✅ Completed successfully"
    elif status == "error":
        return "❌ Processing failed"
    elif status == "warning":
        return "⚠️ Completed with warnings"
    else:
        return "Waiting to start..."


def create_navigation_table_dataframe(pdf_links):
    """Create a navigation table using pandas DataFrame"""
    if not pdf_links:
        return None

    table_data = []
    for i, filename in enumerate(pdf_links):
        # Extract display name from filename/URL
        if "https" in filename:
            display_name = filename.split("/")[-1] if filename.split("/")[-1] else filename
        else:
            display_name = filename.split("/")[-1].split("_")[-1] if "_" in filename else filename.split("/")[-1]

        status_text = get_status_display(display_name)
        progress_text = get_progress_info(display_name)

        table_data.append({
            "📄 Document Name": display_name,
            "📊 Status": status_text,
            "🔄 Progress": progress_text
        })

    return pd.DataFrame(table_data)


def update_navigation_table(pdf_links):
    """Update the navigation table with current status"""
    if st.session_state.navigation_table_placeholder:
        df = create_navigation_table_dataframe(pdf_links)
        if df is not None and not df.empty:
            st.session_state.navigation_table_placeholder.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "📄 Document Name": st.column_config.TextColumn(
                        "📄 Document Name",
                        width="large",
                        help="Name of the PDF document"
                    ),
                    "📊 Status": st.column_config.TextColumn(
                        "📊 Status",
                        width="medium",
                        help="Current processing status"
                    ),
                    "🔄 Progress": st.column_config.TextColumn(
                        "🔄 Progress",
                        width="medium",
                        help="Detailed progress information"
                    ),
                }
            )
        else:
            print("[WARNING] df.shape is not found", df)


def generate_download_content(display_name):
    """Generates a single string with all component content for download."""
    content = [f"Summary for: {display_name}\n\n"]
    component_content = st.session_state.component_content.get(display_name, {})
    text = component_content.get('summary', "*No content generated for this section.*")
    content.append(f"--- 📚 Summary ---\n\n{text}\n\n")
    return "".join(content)


st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(enhanced_css, unsafe_allow_html=True)

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
    st.session_state.timer_placeholder = None
if 'file_status_tracking' not in st.session_state:
    st.session_state.file_status_tracking = {}
if 'navigation_table_placeholder' not in st.session_state:
    st.session_state.navigation_table_placeholder = None

with st.sidebar:
    st.markdown("<h2>Aya Multi-File Summary Tool</h2>", unsafe_allow_html=True)

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
        urls = [url.strip() for url in url_input.strip().split('\n') if url.strip()]
        valid_urls = []
        for url in urls:
            valid_urls.append(url)
            pdf_links.append(url)

    if st.button("🧹 Clear Upload Cache", help="Removes temporarily saved uploaded files and resets the app state."):
        try:
            clear_upload_directory()
            st.success("Temporary upload directory cleared.")
            # Reset all session state
            st.session_state.file_placeholders = {}
            st.session_state.results = {}
            st.session_state.component_content = {}
            st.session_state.tasks_running = 0
            st.session_state.start_time = None
            st.session_state.file_status_tracking = {}
            if st.session_state.timer_placeholder:
                st.session_state.timer_placeholder.empty()
            st.session_state.timer_placeholder = None
            if st.session_state.navigation_table_placeholder:
                st.session_state.navigation_table_placeholder.empty()
            st.session_state.navigation_table_placeholder = None
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

# Main app description
st.markdown(
    """
    Welcome to **Aya Insight**! Your intelligent assistant for dissecting and understanding PDF documents.
    """
)

# Create dedicated container for navigation table and timer
# This ensures they don't interfere with each other
navigation_container = st.container()
timer_container = st.container()

# Initialize placeholders in their respective containers
if st.session_state.navigation_table_placeholder is None:
    with navigation_container:
        st.session_state.navigation_table_placeholder = st.empty()

if st.session_state.timer_placeholder is None:
    with timer_container:
        st.session_state.timer_placeholder = st.empty()


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
    # Initialize file status tracking for new files
    for filename in pdf_links:
        display_name = filename.split("/")[-1].split("_")[-1] if "https" not in filename else filename
        if display_name not in st.session_state.file_status_tracking:
            st.session_state.file_status_tracking[display_name] = "pending"

    # Update navigation table
    update_navigation_table(pdf_links)

    if not summarize_button and not st.session_state.tasks_running:
        st.info("Click 'Summarize All Files' in the sidebar to process.")

    if summarize_button:

        st.session_state.start_time = time.time()

        st.session_state.results = {}
        st.session_state.file_placeholders = {}
        st.session_state.component_content = {}
        st.session_state.tasks_running = len(pdf_links)

        # Reset status tracking for processing
        for filename in pdf_links:
            display_name = filename.split("/")[-1].split("_")[-1] if "https" not in filename else filename
            st.session_state.file_status_tracking[display_name] = "processing"

        print("Processing files/..//")
        update_navigation_table(pdf_links)

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

        # Create expandable sections for each PDF
        for file_index, filename in enumerate(pdf_links):
            # Create an expander for each file, styled by CSS
            display_name = filename.split("/")[-1].split("_")[-1] if "https" not in filename else filename

            # Add unique ID for smooth scrolling
            expander_html = f'<div id="expander_{file_index}"></div>'
            st.markdown(expander_html, unsafe_allow_html=True)

            with st.expander(f"📄 {display_name}", expanded=True):
                status_ph = st.empty()
                status_ph.info("Queued for processing...")

                component_phs_dict = {}
                component_content_dict = {}
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

                st.session_state.file_placeholders[display_name] = {
                    'status': status_ph,
                    'components': component_phs_dict
                }
                st.session_state.component_content[display_name] = component_content_dict

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
            display_name = filename.split("/")[-1].split("_")[-1] if "https" not in filename else filename
            if display_name and display_name in st.session_state.file_placeholders:
                process_executor.submit(process_file_task, result_data, update_queue)
                submitted_files.add(display_name)
                st.session_state.file_placeholders[display_name]['status'].info(f"⏳ Processing: {display_name}...")
            else:
                logger.warning(f"Filename {display_name} from extraction_results not found in placeholders or is None.")
                st.session_state.tasks_running -= 1  # Decrement if a file can't be processed

        files_done_processing = set()
        # Store component status for each file to determine overall file status
        component_statuses = {fname: {} for fname in submitted_files}
        active_spinners_markers = {fname: {ckey: True for ckey in st.session_state.component_content[fname]} for fname
                                   in submitted_files}

        # Track last update times to reduce frequency
        last_timer_update = 0
        last_table_update = 0
        timer_update_interval = 1.0  # Update timer every 1 second
        table_update_interval = 2.0  # Update table every 2 seconds

        while st.session_state.tasks_running > 0:
            current_time = time.time()

            # Update timer less frequently
            if st.session_state.start_time is not None and st.session_state.timer_placeholder:
                if current_time - last_timer_update >= timer_update_interval:
                    elapsed_time = current_time - st.session_state.start_time
                    with timer_container:
                        st.session_state.timer_placeholder.markdown(
                            f"⏳ **Overall Processing Time:** `{elapsed_time:.2f} seconds`")
                    last_timer_update = current_time

            try:
                msg = update_queue.get(timeout=0.1)  # Timeout to allow UI updates and time checks
                msg_type = msg[0]
                filename = msg[1]
                display_name = filename.split("/")[-1].split("_")[-1] if "https" not in filename else filename

                if display_name not in st.session_state.file_placeholders:
                    logger.warning(
                        f"Received message for unknown/already-cleared file: {display_name}. Type: {msg_type}")
                    update_queue.task_done()
                    continue

                file_placeholders = st.session_state.file_placeholders[display_name]
                file_component_content = st.session_state.component_content[display_name]

                table_needs_update = False

                if msg_type == 'chunk':
                    _, _, comp_key, text_delta = msg
                    if comp_key in file_component_content:
                        file_component_content[comp_key] += text_delta
                        display_text = file_component_content[comp_key] + '<span class="blinking-cursor">▌</span>'
                        if comp_key in file_placeholders['components']:
                            file_placeholders['components'][comp_key].markdown(display_text, unsafe_allow_html=True)

                elif msg_type == 'comp_done':
                    _, _, comp_key, final_message = msg
                    component_statuses.setdefault(display_name, {})[comp_key] = 'done'
                    active_spinners_markers.get(display_name, {}).pop(comp_key, None)  # Remove marker

                    if comp_key in file_placeholders['components']:
                        final_content = file_component_content.get(comp_key, "")
                        if final_message:
                            if final_content and not final_content.endswith("\n\n"):
                                final_content += "\n\n"
                            final_content += f"*{final_message}*"
                        file_placeholders['components'][comp_key].markdown(final_content)

                elif msg_type == 'comp_error':
                    _, _, comp_key, error_msg = msg
                    component_statuses.setdefault(display_name, {})[comp_key] = 'error'
                    active_spinners_markers.get(display_name, {}).pop(comp_key, None)  # Remove marker

                    if comp_key in file_placeholders['components']:
                        file_placeholders['components'][comp_key].error(f"⚠️ {error_msg}")

                elif msg_type == 'status':  # General status update for the file
                    _, _, _, status_msg = msg  # comp_key is None for general file status
                    file_placeholders['status'].info(f"⏳ {status_msg}")  # Styled by CSS

                elif msg_type == 'file_done':
                    if display_name not in files_done_processing:
                        files_done_processing.add(display_name)
                        st.session_state.tasks_running -= 1

                        final_file_status = 'success'  # Assume success initially
                        file_comp_statuses = component_statuses.get(display_name, {})

                        if not file_comp_statuses:
                            if any(v == 'error' for v in file_comp_statuses.values()):
                                final_file_status = 'error'
                            else:
                                final_file_status = 'warning_nodata'
                        elif any(v == 'error' for v in file_comp_statuses.values()):
                            final_file_status = 'error'
                        elif not any(file_component_content.get(k, "").strip() and file_component_content.get(k,
                                                                                                              "").strip() != "*No specific content generated for this section.*"
                                     for k in file_comp_statuses if file_comp_statuses.get(k) == 'done'):
                            final_file_status = 'warning_nodata'

                        status_ph = file_placeholders['status']
                        if final_file_status == 'success':
                            status_ph.success(f"✅ Summarization Complete for {display_name}!")
                            st.session_state.results[display_name] = True
                            st.session_state.file_status_tracking[display_name] = "success"
                        elif final_file_status == 'warning_nodata':
                            status_ph.warning(
                                f"⚠️ {display_name} processed, but some sections have limited or no content.")
                            st.session_state.results[display_name] = "warning"
                            st.session_state.file_status_tracking[display_name] = "warning"
                        else:  # 'error'
                            status_ph.error(f"❌ {display_name} completed with errors in some sections.")
                            st.session_state.results[display_name] = False
                            st.session_state.file_status_tracking[display_name] = "error"

                        table_needs_update = True

                        # Ensure all component spinners are removed for this file
                        for comp_key_iter, placeholder in file_placeholders['components'].items():
                            if active_spinners_markers.get(display_name, {}).get(
                                    comp_key_iter):  # If spinner was still active
                                final_c_content = file_component_content.get(comp_key_iter, "")
                                placeholder.markdown(final_c_content)  # Update with final content

                elif msg_type == 'file_error':
                    _, critical_error_msg = msg
                    if display_name not in files_done_processing:
                        files_done_processing.add(display_name)
                        st.session_state.tasks_running -= 1
                        file_placeholders['status'].error(
                            f"❌ Critical Error processing {display_name}: {critical_error_msg}")
                        st.session_state.results[display_name] = False
                        st.session_state.file_status_tracking[display_name] = "error"

                        table_needs_update = True

                        for comp_key_iter, placeholder in file_placeholders['components'].items():
                            if active_spinners_markers.get(display_name, {}).get(comp_key_iter):
                                placeholder.markdown("_Processing halted due to critical file error._")

                # Update navigation table only when needed and not too frequently
                if table_needs_update and (current_time - last_table_update >= table_update_interval):
                    print("Updating navigation table due to status change...")
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

        # Final updates
        if st.session_state.start_time is not None and st.session_state.timer_placeholder:
            final_elapsed_time = time.time() - st.session_state.start_time
            with timer_container:
                st.session_state.timer_placeholder.success(
                    f"🎉 All processing finished in {final_elapsed_time:.2f} seconds!")

        # Final table update
        with navigation_container:
            update_navigation_table(pdf_links)

        st.session_state.start_time = None
        process_executor.shutdown(wait=False)

else:
    # Show message when no PDFs are available
    st.info("👈 Please upload PDF files or provide PDF URLs using the sidebar to get started.")