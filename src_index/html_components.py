import os
import tempfile
from html import escape

def create_clean_html_content(row):
    # Creating a clean HTML structure with modern styling
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{escape(row['title'])}</title>
        <link href="https://fonts.googleapis.com/css?family=Roboto&display=swap" rel="stylesheet"> 
        <style>
            * {{
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Roboto', sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }}
            .card {{
                background-color: #ffffff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                max-width: 800px;
                width: 90%;
                overflow: auto;
                height: 80vh;
            }}
            .highlight {{
                font-weight: bold;
                color: #ff9900;
            }}
            h1 {{
                color: #333333;
                text-align: left;
            }}
            .section {{
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 1px solid #eeeeee;
            }}
            p {{
                line-height: 1.6;
                margin: 0;
            }}
            a {{
                color: #0066cc;
                text-decoration: none;
            }}
            .state {{
                color: #333333;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>{escape(row['llm_title'])}</h1>
            <div class="section">
                <p><span class="highlight">Title:</span> {escape(row['title'])}</p>
            </div>
            <div class="section">
                <p><span class="highlight">Topic:</span> {escape(row['topic_title'])}</p>
            </div>
            <div class="section">
                <p><span class="highlight">Legal Question:</span> {escape(row['body'])}</p>
            </div>
            <div class="section">
                <p>Text Label: {escape(row['text_label'])}</p>
                <p>Timestamp: {row['timestamp']}</p>
                <p><a href="{row['full_link']}">Full Link</a></p>
            </div>
            <div class="section">
                <p class="state">State: {escape(row['State'])}</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


import os
import tempfile
from html import escape

def generate_html_files(df, row_limit=None, output_dir='data'):
    urls = []
    
    if row_limit is None:
        row_limit = len(df)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for index, row in df.iterrows():
        if index == row_limit:
            break

        html_content = create_clean_html_content(row)
        
        # Create a temporary file in the specified directory
        with tempfile.NamedTemporaryFile(dir=output_dir, delete=False, suffix='.html') as temp:
            temp.write(html_content.encode('utf-8'))
        
        file_url = 'file://' + os.path.abspath(temp.name)
        urls.append(file_url)
    
    df['html_url'] = urls
    
    return df


def delete_temp_files(df):
    """Delete the html files created by `generate_html_files`"""
    for url in df['html_url']:
        # Convert the URL back to a file path
        file_path = url.replace('file://', '')
        
        # Check if the file exists and delete it
        if os.path.isfile(file_path):
            os.remove(file_path)

