from pathlib import Path
import numpy as np
import pandas as pd
import tarfile
import urllib.request
import requests as rq
import csv 


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))
housing_full = load_housing_data()



def download_csv(URL_='') -> pd.DataFrame:
    with rq.Session() as s:
        download = s.get(URL_)
        decoded_content = download.content.decode('utf-8')
        cv = csv.reader(decoded_content.splitlines(),delimiter=',')
        

    return pd.read_csv(cv)