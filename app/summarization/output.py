"""
Output handling for document summaries in multiple formats.
"""
import os
from ..config.settings import SUMMARIES_OUTPUT_DIR


class SummaryOutputManager:
    """
    Manager for saving and retrieving document summaries in multiple formats.
    """

    def __init__(self, output_dir=SUMMARIES_OUTPUT_DIR):
        """
        Initialize output manager with output directory.

        Args:
            output_dir (str): Directory to save summaries
        """
        self.output_dir = output_dir
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def save_summary(self, filename, summary, formats=None):
        """
        Save a document summary to files in specified formats.

        Args:
            filename (str): Name of the original document
            summary (str): Summary text
            formats (list): List of formats to save. Defaults to ['markdown', 'html']

        Returns:
            dict: Paths to the saved summary files by format
        """
        if formats is None:
            formats = ['markdown', 'html']

        output_paths = {}

        # Generate and save in each requested format
        for fmt in formats:
            if fmt == 'markdown':
                output_paths['markdown'] = self._save_markdown(filename, summary)
            elif fmt == 'html':
                output_paths['html'] = self._save_html(filename, summary)
            else:
                print(f"Warning: Unsupported format '{fmt}' requested")

        return output_paths

    def _save_markdown(self, filename, summary):
        """
        Save a document summary to a markdown file.

        Args:
            filename (str): Name of the original document
            summary (str): Summary text

        Returns:
            str: Path to the saved markdown file
        """
        # Create markdown output
        markdown_content = f""
        markdown_content += summary
        markdown_content += "\n\n---\n"

        # Save to file
        output_path = os.path.join(self.output_dir, f"{filename}.md")
        with open(output_path, "w") as f:
            f.write(markdown_content)

        print(f"Saved markdown summary to: {output_path}")
        return output_path

    def _save_html(self, filename, summary):
        """
        Save a document summary to an HTML file.

        Args:
            filename (str): Name of the original document
            summary (str): Summary text

        Returns:
            str: Path to the saved HTML file
        """
        # Convert summary to HTML paragraphs
        paragraphs = summary.split('\n\n')
        html_paragraphs = ''.join([f"<p>{p}</p>" for p in paragraphs if p.strip()])

        # Create HTML output with basic styling
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Summary for {filename}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        p {{
            margin-bottom: 16px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <h1>Summary for {filename}</h1>
    <div class="content">
        {html_paragraphs}
    </div>
    <div class="footer">
        <p>Generated summary</p>
    </div>
</body>
</html>
"""
        # Save to file
        output_path = os.path.join(self.output_dir, f"{filename}.html")
        with open(output_path, "w") as f:
            f.write(html_content)

        print(f"Saved HTML summary to: {output_path}")
        return output_path

    def get_available_formats(self, filename):
        """
        Check which formats are available for a given file.

        Args:
            filename (str): Base filename to check

        Returns:
            list: Available formats for this file
        """
        available_formats = []
        base_name = os.path.splitext(filename)[0]

        if os.path.exists(os.path.join(self.output_dir, f"{base_name}.md")):
            available_formats.append('markdown')
        if os.path.exists(os.path.join(self.output_dir, f"{base_name}.html")):
            available_formats.append('html')

        return available_formats