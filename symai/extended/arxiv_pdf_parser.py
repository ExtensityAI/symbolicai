import os
import re
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor
from .. import Expression, Symbol
from .file_merger import FileMerger


class ArxivPdfParser(Expression):
    def __init__(self, url_pattern: str = r'https://arxiv.org/(?:pdf|abs)/(.*?)(?:\.pdf)?'):
        super().__init__()
        self.url_pattern = url_pattern
        self.merger = FileMerger()

    def forward(self, data: Symbol) -> Symbol:
        # Extract all urls from the data
        urls = re.findall(self.url_pattern, str(data))

        # Convert all urls to pdf urls
        pdf_urls = [f"https://arxiv.org/pdf/{url.split('/')[-1]}.pdf" for url in urls]

        # Create temporary folder in the home directory
        home_dir = os.path.expanduser("~")
        output_path = os.path.join(home_dir, ".symai", "temp/downloads")
        os.makedirs(output_path, exist_ok=True)

        pdf_files = []
        print(pdf_urls)
        with ThreadPoolExecutor() as executor:
            # Download all pdfs in parallel
            pdf_files = list(executor.map(self.download_pdf, pdf_urls, [output_path]*len(pdf_urls)))

        # Merge all pdfs into one file
        merged_file = self.merger(output_path)

        # Return the merged file as a Symbol
        return_file = self._sym_return_type(merged_file)

        # Delete the temporary folder after merging the files
        shutil.rmtree(output_path)

        return return_file

    def download_pdf(self, url, output_path):
        # Download pdfs
        response = requests.get(url)
        file = os.path.join(output_path, f'{url.split("/")[-1]}.pdf')
        with open(file, 'wb') as f:
            f.write(response.content)
        return file