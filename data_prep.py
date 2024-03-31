import fitz  # PyMuPDF
import os
from concurrent.futures import ProcessPoolExecutor

def process_pdf(pdf_path):
    print(f"Processing: {pdf_path}")
    text = ''
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Failed to process {pdf_path}: {e}")
    return text

def extract_text_from_pdfs(folder_path, output_file):
    pdf_paths = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.pdf'):
                pdf_paths.append(os.path.join(root, filename))

    texts = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_pdf, path): path for path in pdf_paths}
        for future in futures:
            try:
                texts.append(future.result())
            except Exception as e:
                print(f"Error processing file {futures[future]}: {e}")

    with open(output_file, 'w') as output:
        for text in texts:
            output.write(text)

    print(f"All files processed. Output written to: {os.path.abspath(output_file)}")

# Specify the root folder containing the PDFs
folder_path = 'path/to/your/pdf/folder'
output_file = 'input.txt'

# Call the function with your specified folder and output file
extract_text_from_pdfs(folder_path, output_file)
