# app.py
import streamlit as st
import os
import sys
import logging

# Add the project root to the sys.path to allow importing modules like config, document_processing, etc.
# This assumes app.py is in the project root directory.
# Adjust the path if your app.py is in a subdirectory.

try:
    # Import the necessary functions from your main script    print("YES1")
    from main import process_uploaded_files, setup_retrieval_system, summarize_extracted_documents
    # Configure Streamlit's logging to match your application's settings
    logging.basicConfig(level='INFO', format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Streamlit app started and logging configured.")

    # Flag to check if modules were imported successfully
    modules_loaded = True

except ImportError as e:
    st.error(f"Could not import application modules. Please ensure your project structure is correct and dependencies are installed.")
    st.error(f"ImportError: {e}")
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import application modules: {e}", exc_info=True)
    modules_loaded = False # Set flag to False if imports fail


# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Aya Insight Document Summarizer",
    page_icon="📄",
    layout="wide"
)

# --- Session State Initialization ---
# Initialize session state variables if they don't exist
if 'api_key_entered' not in st.session_state:
    st.session_state.api_key_entered = False
if 'summary_results' not in st.session_state:
    st.session_state.summary_results = None
if 'selected_filename' not in st.session_state:
    st.session_state.selected_filename = None


# --- API Key Input Section ---
if not st.session_state.api_key_entered:
    st.title("🔒 Enter Your Cohere API Key to Unlock")
    api_key = st.text_input("Cohere API Key", type="password", help="Enter your Cohere API key to use the summarization service.")

    if st.button("Unlock"):
        if api_key:
            # Basic validation: Just check if it's not empty.
            # For a real application, you might want to validate by making a small API call.
            os.environ["COHERE_API_KEY"] = api_key # Set the environment variable
            st.session_state.api_key_entered = True
            st.success("API Key accepted. You can now upload documents.")
            st.rerun() # Rerun the app to show the main content
        else:
            st.warning("Please enter your Cohere API key.")

# --- Main Application Content (Unlocked) ---
if st.session_state.api_key_entered and modules_loaded:
    st.title("📄 Aya Insight Document Summarizer")
    st.markdown("""
    Upload one or more PDF or image files to get a structured summary for each document.
    """)

    # --- File Uploader ---
    uploaded_files = st.file_uploader(
        "Choose Document Files",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif"], # Added image types
        accept_multiple_files=True,
        help="You can upload multiple PDF or image documents here."
    )

    # --- Summarize Button and Logic ---
    if uploaded_files: # Only show button if files are uploaded
        st.info(f"You have uploaded {len(uploaded_files)} file(s).")

        if st.button("Generate Summaries", key="summarize_button"):
            st.session_state.selected_filename = None # Reset selected file on new summary generation
            if not uploaded_files:
                st.warning("Please upload at least one file before generating summaries.")
            else:
                st.subheader("Processing Documents...")
                all_summary_results = [] # To store results for display

                # Use a spinner to indicate processing
                with st.spinner("Processing documents and generating summaries... This may take a few minutes depending on file size and number."):
                    try:
                        # Step 1: Process uploaded files (Extraction)
                        logger.info(f"Calling process_uploaded_files with {len(uploaded_files)} files.")
                        extraction_results = process_uploaded_files(uploaded_files)
                        logger.info(f"Finished document extraction. {len(extraction_results)} results obtained.")

                        # Check if any files were successfully extracted
                        if not any(res.get('text') for res in extraction_results):
                            st.error("No text could be extracted from the uploaded files. Please check the file formats.")
                            logger.error("No text extracted from any uploaded file.")
                            st.session_state.summary_results = [] # Store empty results
                            # st.stop() # Don't stop, allow user to try again

                        # Step 2: Setup retrieval system (Vector Store and Embedding)
                        logger.info("Calling setup_retrieval_system.")
                        extraction_results_with_chunks, retriever = setup_retrieval_system(extraction_results)
                        logger.info("Retriever system setup complete.")

                        # Step 3: Summarize the extracted documents
                        logger.info("Calling summarize_extracted_documents.")
                        summary_results = summarize_extracted_documents(extraction_results_with_chunks, retriever)
                        logger.info(f"Finished summarization. {len(summary_results)} summary results obtained.")

                        st.session_state.summary_results = summary_results # Store results in session state

                    except FileNotFoundError as fnf_error:
                        st.error(f"Configuration Error: {fnf_error}. Please check your environment settings.")
                        logger.error(f"Configuration Error during Streamlit process: {fnf_error}", exc_info=True)
                        st.session_state.summary_results = [] # Store empty results on error
                    except Exception as e:
                        st.error(f"An unexpected error occurred during processing: {e}")
                        logger.error(f"An unexpected error occurred during Streamlit process: {e}", exc_info=True)
                        st.session_state.summary_results = [] # Store empty results on error


    # --- Display Document Tiles and Summaries ---
    if st.session_state.summary_results is not None:
        st.subheader("Summaries:")

        if not st.session_state.summary_results:
            st.info("No summaries were generated. Upload files and click 'Generate Summaries'.")
        else:
            # Display files as a grid of clickable tiles
            files_per_row = 3
            rows = len(st.session_state.summary_results) // files_per_row + (len(st.session_state.summary_results) % files_per_row > 0)

            # Create a list of filenames for easy access
            filenames = [res.get('filename', f'File {i+1}') for i, res in enumerate(st.session_state.summary_results)]

            for i in range(rows):
                cols = st.columns(files_per_row)
                for j in range(files_per_row):
                    file_index = i * files_per_row + j
                    if file_index < len(st.session_state.summary_results):
                        result = st.session_state.summary_results[file_index]
                        filename = result.get('filename', f'File {file_index+1}')
                        is_selected = st.session_state.selected_filename == filename

                        # Create a tile using a button or markdown link
                        # Using a button inside a column for simplicity
                        with cols[j]:
                            # Add a border or highlight if selected
                            tile_style = "border: 2px solid lightgrey; padding: 10px; margin: 5px; text-align: center; cursor: pointer;"
                            if is_selected:
                                tile_style = "border: 2px solid steelblue; padding: 10px; margin: 5px; text-align: center; cursor: pointer; background-color: #e6f3ff;" # Highlight color

                            # Use markdown with HTML to create the clickable tile appearance
                            # When clicked, set the selected filename in session state
                            st.markdown(
                                f"""
                                <div style="{tile_style}" onclick="document.getElementById('hidden_button_{file_index}').click()">
                                    📄<br>
                                    <strong>{filename}</strong>
                                </div>
                                <button id="hidden_button_{file_index}" style="display: none;" onclick="document.getElementById('hidden_button_{file_index}').click()"></button>
                                """,
                                unsafe_allow_html=True
                            )
                            # Streamlit buttons don't work directly with markdown clicks like this easily.
                            # A simpler approach is to use a standard button and handle the click.
                            # Let's use a standard button instead of complex markdown/JS.

                            # Alternative using a standard button:
                            if st.button(f"📄 {filename}", key=f"tile_button_{file_index}"):
                                st.session_state.selected_filename = filename
                                logger.info(f"Selected file: {filename}")
                                st.rerun() # Rerun to display the summary


            # Display summary of the selected file
            if st.session_state.selected_filename:
                st.markdown("---") # Separator
                st.subheader(f"Summary for: {st.session_state.selected_filename}")

                # Find the summary for the selected file
                selected_summary = None
                selected_result = None
                for result in st.session_state.summary_results:
                    if result.get('filename') == st.session_state.selected_filename:
                        selected_summary = result.get('summary')
                        selected_result = result
                        break

                if selected_summary:
                    if selected_result.get('success'):
                         st.markdown(selected_summary) # Render markdown summary
                    else:
                         st.error(f"Could not load summary for {st.session_state.selected_filename}: {selected_result.get('error', 'Unknown error')}")
                else:
                    st.info(f"Summary not available for {st.session_state.selected_filename}.")

            # Display overall processing status
            successful_count = sum(res.get('success', False) for res in st.session_state.summary_results)
            total_files = len(st.session_state.summary_results)
            st.markdown(f"---") # Final separator
            st.success(f"Processed {total_files} files. Successfully summarized {successful_count}.")
            if successful_count < total_files:
                st.warning("Some files could not be processed or summarized. See error messages above.")


# --- Message if API Key is not entered and modules loaded ---
if not st.session_state.api_key_entered:
    st.info("Enter your Cohere API Key above to unlock the application functionality.")

