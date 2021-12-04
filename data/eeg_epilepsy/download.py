import pathlib
import requests
import zipfile

URL = "https://web.archive.org/web/20200318000445/http://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv"


def download(from_path, to_path):

    if not to_path.exists():

        try:
            r = requests.get(url=from_path)

            with open(to_path, "wb") as file:
                file.write(r.content)

        except:
            print("error downloading {}".format(str(from_path)))


if __name__ == "__main__":

    print("downloading eeg epilepsy data")
    download(URL, to_path=pathlib.Path("dataset.csv"))
