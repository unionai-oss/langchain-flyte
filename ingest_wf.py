import functools
import hashlib
from typing import List, Annotated

from flytekit import ImageSpec
from flytekit import task, workflow, HashMethod, Resources, map_task
from flytekit.tools import subprocess
from langchain import FAISS
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# BUF_SIZE = 65536  # let's read docs in 64kb chunks!

image = ImageSpec(registry="ghcr.io/unionai-oss",
                  name="langchain-flyte",
                  packages=["langchain", "sentence_transformers", "faiss-cpu", "beautifulsoup4"],
                  apt_packages=["wget"])


# def hash_flyte_file(f: FlyteFile) -> str:
#     """Hash a FlyteFile."""
#     md5 = hashlib.md5()
#     with f.open("rb") as fp:
#         while True:
#             data = f.read(BUF_SIZE)
#             if not data:
#                 break
#             md5.update(data)
#     return md5.hexdigest()


def hash_document(d: Document) -> str:
    """TODO: this can be made way more efficinet by using a hash as part of the pydantic base class."""
    md5 = hashlib.md5()
    md5.update(d.json())
    return md5.hexdigest()


# @task(cache_version="1", cache=True, container_image=image)
# def download_documents(docs_home: str) -> List[Annotated[FlyteFile, HashMethod[hash_flyte_file]]]:
#     """Load documents."""
#     subprocess.check_call(f"wget -r -A.html -P rtdocs {docs_home}")
#     return [FlyteFile(f"rtdocs/{f}") for f in os.listdir("rtdocs")]


# TODO This should not be cached, as the source has to be loaded multiple times, but for testing purposes, we are caching it.
@task(cache_version="1", cache=True, container_image=image)
def download_load_documents(docs_home: str) -> List[Annotated[Document, HashMethod[hash_document]]]:
    """
    We could download the documents to a special folder using
    f"wget -r -A.html -P rtdocs {docs_home}" and then load them from there.
    """
    subprocess.check_call(f"wget -r -A.html {docs_home}")
    return ReadTheDocsLoader("langchain.readthedocs.io/en/latest/").load()


@task(cache_version="1", cache=True, requests=Resources(cpu="1", mem="8Gi"), container_image=image)
def split_doc_create_embeddings(raw_document: Document, chunk_size: int, chunk_overlap: int) -> FAISS:
    """
    Split documents into chunks.
    # TODO: we can also perform the "Embeddings" lookup independently here, using a Flyte Agent, which can allow independent retries and scaling.
    # TODO: I am currently using pickle for FAISS, ideally we should use FlyteFile like interface so that we can lazily load them on the target.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = text_splitter.split_documents([raw_document])
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


@task(cache_version="1", cache=True, requests=Resources(cpu="1", mem="8Gi"), container_image=image)
def merge_embeddings(vectorstores: List[FAISS]) -> FAISS:
    """Merge embeddings.
    TODO: We should convert FAISS stores to be FlyteFiles, so that they can be lazily loaded.
    Rather
    change the function signature to
    def merge_embeddings(vectorstores: Iterator[FAISS]) -> FAISS:
      ...
    To support this we need to write a new typetransformer for Iterator.
    """
    # Ideally have a method like this
    aggregated_store = vectorstores[0]
    for i in range(1, len(vectorstores)):
        aggregated_store.merge_from(vectorstores[i])
    return aggregated_store


@workflow
def ingest(docs_home: str = "langchain.readthedocs.io/en/latest/") -> FAISS:
    documents = download_load_documents(docs_home=docs_home)
    splitter = functools.partial(split_doc_create_embeddings, chunk_size=1000, chunk_overlap=200)
    vectorstores = map_task(splitter)(raw_document=documents)
    return merge_embeddings(vectorstores=vectorstores)


if __name__ == '__main__':
    ingest()
