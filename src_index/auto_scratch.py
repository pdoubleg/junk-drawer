import os
import pandas as pd
import markdown
from markdown_include.include import MarkdownInclude
import pdfkit

class MarkdownConverter:
    """
    A class used to convert markdown files to HTML and PDF.

    ...

    Attributes
    ----------
    css_file_name : str
        a string specifying the name of the CSS file
    directory : str
        a string specifying the directory where markdown files are stored

    Methods
    -------
    generate_md_files(df, title_col, *content_cols):
        Generates markdown files from a pandas DataFrame.
    generate_md_from_string(md_string, file_name):
        Generates a markdown file from a string.
    convert(markdown_file_name):
        Converts a markdown file to HTML and PDF.
    """

    def __init__(self, css_file_name, directory="markdown_files"):
        """
        Constructs all the necessary attributes for the MarkdownConverter object.

        Parameters
        ----------
            css_file_name : str
                a string specifying the name of the CSS file
            directory : str, optional
                a string specifying the directory where markdown files are stored (default is "markdown_files")
        """
        self.css_file_name = css_file_name
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def generate_md_files(self, df, title_col, *content_cols):
        """
        Generates markdown files from a pandas DataFrame.

        Parameters
        ----------
            df : pandas.DataFrame
                DataFrame containing the data
            title_col : str
                Column name to be used for the title of the markdown file
            *content_cols : str
                Column names to be used for the content of the markdown file
        """
        for _, row in df.iterrows():
            try:
                with open(os.path.join(self.directory, f"{row[title_col]}.md"), "w") as f:
                    f.write(f"# {row[title_col]}\n")
                    for col in content_cols:
                        f.write(f"{row[col]}\n")
            except Exception as e:
                print(f"Error writing file {row[title_col]}.md: {e}")

    def generate_md_from_string(self, md_string, file_name):
        """
        Generates a markdown file from a string.

        Parameters
        ----------
            md_string : str
                String containing the markdown content
            file_name : str
                Name of the file to be created
        """
        try:
            with open(os.path.join(self.directory, f"{file_name}.md"), "w") as f:
                f.write(md_string)
        except Exception as e:
            print(f"Error writing file {file_name}.md: {e}")

    def _html(self, markdown_file_name):
        with open(markdown_file_name, mode="r", encoding="utf-8") as markdown_file:
            with open(self.css_file_name, mode="r", encoding="utf-8") as css_file:
                markdown_input = markdown_file.read()
                css_input = css_file.read()

                markdown_path = os.path.dirname(markdown_file_name)
                markdown_include = MarkdownInclude(configs={"base_path": markdown_path})
                html = markdown.markdown(
                    markdown_input, extensions=["extra", markdown_include, "meta", "tables"]
                )

                return f"""
                <html>
                  <head>
                    <style>{css_input}</style>
                  </head>
                  <body>{html}</body>
                </html>
                """

    def convert(self, markdown_file_name):
        file_name = os.path.splitext(os.path.basename(markdown_file_name))[0]
        html_string = self._html(markdown_file_name)

        with open(
            os.path.join(self.directory, file_name + ".html"), "w", encoding="utf-8", errors="xmlcharrefreplace"
        ) as output_file:
            output_file.write(html_string)
        config = pdfkit.configuration(wkhtmltopdf='../../wkhtmltopdf/bin/wkhtmltopdf.exe')
        pdfkit.from_file(os.path.join(self.directory, file_name + ".html"), os.path.join(self.directory, file_name + ".pdf"), configuration=config, css=self.css_file_name)
