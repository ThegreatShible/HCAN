import wget
from zipfile import ZipFile
import os


#Assign a value
     = ""

def setup() :
    glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
    wget.download(url, embedding_root_dir)
    zip_file = os.path.join(embedding_root_dir, "glove.6B.zip")
    with ZipFile(zip_file, 'r') as zipObj:
        zipObj.extractall()
    


if __name__ == "__main__":
    setup()