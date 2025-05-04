import time
from typing import Dict, List
import cohere
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from ..config.settings import LLM_MODEL


class DocumentSummarizer:
    def __init__(self, retriever, batch_size=4):
        self.batch_size = batch_size
        self.retriever = retriever # Store the retriever here

        self.cohere_client = cohere.ClientV2()

        self.components = {
            'basic_info': "Basic Paper Information",
            'abstract': "Abstract Summary",
            'methods': "Methodology Summary",
            'results': "Key Results",
            'limitations': "Limitations & Future Work",
            'related_work': "Related Work",
            'applications': "Practical Applications",
            'technical': "Technical Details",
            'equations': "Key Equations",
        }

        self.prompts = self._initialize_prompts()

        # Validate prompts dictionary matches components
        # This helps catch missing prompts or components
        missing_prompts = [comp for comp in self.components if comp not in self.prompts]
        if missing_prompts:
             print(f"Warning: No prompts found for components: {missing_prompts}")
        missing_components = [prompt_key for prompt_key in self.prompts if prompt_key not in self.components]
        if missing_components:
             print(f"Warning: Prompts found for components not in self.components: {missing_components}")


    def _initialize_prompts(self):
        # It's better to explicitly import what you need
        from ..summarization.prompt2 import (
            basic_info_prompt, abstract_prompt,
            methods_prompt, results_prompt, visuals_prompt, limitations_prompt,
            contributions_prompt, related_work_prompt, applications_prompt,
            technical_prompt, quick_summary_prompt, reading_guide_prompt, # quick_summary & reading_guide prompts might be needed
            equations_prompt
        )
        return {
            'basic_info': basic_info_prompt,
            'abstract': abstract_prompt,
            'methods': methods_prompt,
            'results': results_prompt,
            'limitations': limitations_prompt,
            'related_work': related_work_prompt,
            'applications': applications_prompt,
            'technical': technical_prompt,
            'equations': equations_prompt,
        }

    def summarize_text(self, documents: List[Dict], prompt: str, language: str):
        """
        Summarizes the provided documents using the given prompt and language
        via the Cohere Chat API.
        """
        if not documents:
            print("Warning: No documents provided for summarization.")
            return None # Or an empty string, depending on desired behavior

        # Use the initialized client
        try:
            response = self.cohere_client.chat(
                model=LLM_MODEL,
                documents=documents, # Pass the list of dicts directly
                messages=[
                    {"role": "system", "content": f"You are an expert summarization AI. Please respond in {language}."},
                    {"role": "user", "content": f"{prompt}"}
                ],
            )
            if response and response.message and response.message.content and response.message.content[0] and response.message.content[0].text:
                 return response.message.content[0].text
            else:
                 print(f"Warning: Unexpected API response structure for prompt: {prompt[:50]}...")
                 return None

        except Exception as e:
            print(f"Error during Cohere API call: {e}")
            return None


    def extract_relevant_documents(self, component: str, filename: str, chunk_size: int):
        """
        Extracts relevant documents for a specific component from the retriever.
        """
        query = f"Analyze the {self.components.get(component, component)} section from the document titled '{filename}'."
        # Use the retriever stored in self.
        # Pass the chunk_size parameter correctly
        try:
            documents = self.retriever.get_relevant_docs(
                chromdb_query=query,
                rerank_query=query,
                filter={'filename': filename},
                chunk_size=chunk_size
            )
            return documents
        except Exception as e:
            print(f"Error during document retrieval for component {component}: {e}")
            return []


    def summerize_document(self, filename: str, language: str, chunk_size: int):
        """
        Summarizes a document by processing each component in parallel.
        """
        start_total = time.time()
        components = list(self.components.keys())
        results = {}
        errors = {} # Track errors

        def process_component(comp):
            comp_start = time.time()
            print(f"Starting processing for component: {comp}")
            try:
                document_chunks = self.extract_relevant_documents(comp, filename, chunk_size)

                if not document_chunks:
                    print(f"No documents found for component: {comp} for file {filename}")
                    return comp, None, f"No documents found"

                prompt = self.prompts.get(comp)
                if not prompt:
                    print(f"No prompt defined for component: {comp}")
                    return comp, None, f"No prompt defined"

                # Summarize the retrieved documents
                summary = self.summarize_text(document_chunks, prompt, language)

                comp_end = time.time()
                print(f"Finished processing for component: {comp}. Time taken: {comp_end - comp_start:.2f} seconds")
                return comp, summary, None # Return comp, result, error

            except Exception as e:
                comp_end = time.time()
                print(f"Error processing component {comp}: {e}. Time taken: {comp_end - comp_start:.2f} seconds")
                return comp, None, str(e) # Return comp, result, error

        # Use ThreadPoolExecutor for I/O-bound tasks (API calls)
        # max_workers=None uses a default appropriate for the system
        with ThreadPoolExecutor(max_workers=None) as executor:
            # Submit all component tasks
            future_to_component = {executor.submit(process_component, comp): comp for comp in components}

            # Process results as they complete
            for future in as_completed(future_to_component):
                comp = future_to_component[future]
                try:
                    comp_name, result, error = future.result()
                    if result is not None:
                        results[comp_name] = result
                    elif error:
                        errors[comp_name] = error

                except Exception as exc:
                    # This catches exceptions *within* the future's result retrieval, less common
                    print(f'{comp} generated an exception: {exc}')
                    errors[comp] = str(exc)


        end_total = time.time()
        print(f"\n--- Total summarization time for {filename}: {end_total - start_total:.2f} seconds ---\n")

        # You might want to return both results and errors
        # For simplicity, let's just compile the available results for now
        compiled = self.compile_summary(filename, results)
        # Optionally, add errors to the compiled output or return them separately
        if errors:
             print(f"Components that failed or returned no data: {list(errors.keys())}")
             # You could append error messages to the compiled summary
             # compiled += "\n\n## Processing Errors\n" + "\n".join([f"- {k}: {v}" for k, v in errors.items()])


        return compiled


    def compile_summary(self, filename: str, results: Dict[str, str]) -> str:
        """
        Compiles a summary for a document by concatenating the results of all requested components.
        Orders sections according to a predefined list.
        """
        # Include all components that might have results, maintaining desired order
        sections_order = [
            'basic_info', 'abstract',
            'methods', 'results', 'equations', 'technical',
            'related_work', 'applications', 'limitations'
        ]

        lines = [f"# Summary of {filename}", f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]
        for section in sections_order:
            # Only add a section if it was processed and returned a result
            if section in results and results[section]:
                # Use .get with a default in case a component was added to results
                # but not self.components (though validate init helps prevent this)
                title = self.components.get(section, section).title()
                lines.append(f"## {title}\n") # Use ## for subheadings
                lines.append(f"{results[section]}\n")

        return "\n".join(lines)