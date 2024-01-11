import os
import re
import shutil
import requests

from concurrent.futures import ThreadPoolExecutor, as_completed

from ..symbol import Expression, Symbol
from .file_merger import FileMerger


class ArxivPdfParser(Expression):
    def __init__(self, url_pattern: str = r'https://arxiv.org/(?:pdf|abs)/(\d+.\d+)(?:\.pdf)?', **kwargs):
        super().__init__(**kwargs)
        self.url_pattern = url_pattern
        self.merger = FileMerger()

    def forward(self, data: Symbol, **kwargs) -> Symbol:
        # Extract all urls from the data
        urls = re.findall(self.url_pattern, str(data))

        # Convert all urls to pdf urls
        pdf_urls = [f"https://arxiv.org/pdf/" + (f"{url.split('/')[-1]}.pdf" if 'pdf' not in url else {url.split('/')[-1]}) for url in urls]

        # Create temporary folder in the home directory
        home_dir = os.path.expanduser("~")
        output_path = os.path.join(home_dir, ".symai", "temp/downloads")
        os.makedirs(output_path, exist_ok=True)

        pdf_files = []
        with ThreadPoolExecutor() as executor:
            # Download all pdfs in parallel
            future_to_url = {executor.submit(self.download_pdf, url, output_path): url for url in pdf_urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    pdf_files.append(future.result())
                except Exception as exc:
                    print('%r generated an exception: %s' % (url, exc))

        if len(pdf_files) == 0:
            return None

        # Merge all pdfs into one file
        merged_file = self.merger(output_path, **kwargs)

        # Return the merged file as a Symbol
        return_file = self._to_symbol(merged_file)

        # Delete the temporary folder after merging the files
        shutil.rmtree(output_path)

        return return_file

    def download_pdf(self, url, output_path):
        # Download pdfs
        response = requests.get(url)
        file = os.path.join(output_path, f'{url.split("/")[-1]}')
        with open(file, 'wb') as f:
            f.write(response.content)
        return file
