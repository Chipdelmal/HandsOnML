
import os
import tarfile
import urllib


(DOWNLOAD_ROOT, HOUSING_PATH, HOUSING_URL) = (
        "https://raw.githubusercontent.com/ageron/handson-ml2/master/",
        os.path.join("datasets", "housing"),
        "datasets/housing/housing.tgz"
    )


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
