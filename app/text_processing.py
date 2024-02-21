import pandas as pd

def process_text_file(file):
    df = pd.read_csv(file, header=None)
    texts = df.iloc[0, :].tolist()
    return texts
