import multiprocessing
import os
from argparse import ArgumentParser
from joblib import Parallel, delayed

import requests


def download_folder(a, r, file):
    download_directory = os.path.join(a.download_folder, file[:file.find(".txt")])

    if os.path.exists(download_directory) is False:
        os.mkdir(download_directory)

    with open(os.path.join(r, file), "r") as f:
        lines = f.readlines()
        for l in lines:
            l = l.split(" ")
            if l[1].find("tinypic") > -1:
                continue
            try:
                response = requests.get(l[1], timeout=5)
                img_data = response.content

                if "Content-Type" not in response.headers:
                    index = -1
                else:
                    index = response.headers["Content-Type"].find("image")
                if index > -1:
                    with open(
                            os.path.join(download_directory, l[0] + "." + response.headers["Content-Type"][6:]),
                            'wb') as handler:
                        handler.write(img_data)
            except (
                    requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout,
                    requests.exceptions.TooManyRedirects):
                pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_source", required=True)
    parser.add_argument("--download_folder", default="images")

    args = parser.parse_args()

    if os.path.exists(args.download_folder) is False:
        os.mkdir(args.download_folder)

    cpu_count = multiprocessing.cpu_count()

    with Parallel(n_jobs=cpu_count, prefer='threads') as parallel:
        for root, _, files in os.walk(args.image_source):
            parallel(delayed(download_folder)(args, root, file) for file in files)
