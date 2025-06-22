import pandas as pd


def load_markdown_files(paths):
    data = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        data.append({'path': path, 'text': file_content})

    df = pd.DataFrame(data)
    return df
