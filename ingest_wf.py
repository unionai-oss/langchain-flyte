import functools
import hashlib
import logging
import typing
from subprocess import CalledProcessError
from typing import List, Annotated

from flytekit import ImageSpec
from flytekit import task, workflow, HashMethod, Resources, map_task
from flytekit.tools import subprocess
from langchain import FAISS
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

new_flytekit = ["git+https://github.com/flyteorg/flytekit@be5b3fa3618fe3aaded817fbc4d7cc902c8fe031"]
new_flytekit_apt_requires = ["git"]
image = ImageSpec(registry="ghcr.io/unionai-oss",
                  name="langchain-flyte",
                  packages=["langchain", "sentence_transformers", "faiss-cpu", "beautifulsoup4"] + new_flytekit,
                  apt_packages=["wget"] + new_flytekit_apt_requires,
)


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
    prev_err = None
    try:
        subprocess.check_call(f"wget -r -A.html --content-on-error {docs_home}")
    except Exception as e:
        logging.warning("wget failed, but we will try to load the documents anyway.")
        prev_err = e
    docs = ReadTheDocsLoader(f"{docs_home}").load()
    if len(docs) == 0:
        if prev_err:
            raise ValueError("No documents were loaded.") from prev_err
        else:
            raise ValueError("wget succeeded, but no documents were loaded.")
    return docs


@task(cache_version="1", cache=True, requests=Resources(cpu="1", mem="8Gi"), container_image=image)
def split_doc_create_embeddings(raw_document: Document, chunk_size: int, chunk_overlap: int) -> typing.Optional[FAISS]:
    """
    Split documents into chunks.
    # TODO: we can also perform the "Embeddings" lookup independently here, using a Flyte Agent, which can allow independent retries and scaling.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = text_splitter.split_documents([raw_document])
    if documents is None or len(documents) == 0:
        return None
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


@task(cache_version="1", cache=True, requests=Resources(cpu="1", mem="8Gi"), container_image=image)
def merge_embeddings(vectorstores: List[typing.Optional[FAISS]]) -> typing.Optional[FAISS]:
    """Merge embeddings.
    TODO: We should convert FAISS stores to be FlyteFiles, so that they can be lazily loaded.
    Rather
    change the function signature to
    def merge_embeddings(vectorstores: Iterator[FAISS]) -> FAISS:
      ...
    To support this we need to write a new typetransformer for Iterator.
    """
    # Ideally have a method like this
    if len(vectorstores) == 0:
        return None
    aggregated_store = vectorstores[0]
    for i in range(1, len(vectorstores)):
        if vectorstores[i] is not None:
            if aggregated_store is None:
                aggregated_store = vectorstores[i]
            else:
                aggregated_store.merge_from(vectorstores[i])
    return aggregated_store


@workflow
def ingest(docs_home: str = "langchain.readthedocs.io/en/latest/") -> typing.Optional[FAISS]:
    documents = download_load_documents(docs_home=docs_home)
    splitter = functools.partial(split_doc_create_embeddings, chunk_size=1000, chunk_overlap=200)
    vectorstores = map_task(splitter)(raw_document=documents)
    return merge_embeddings(vectorstores=vectorstores)


if __name__ == '__main__':
    ingest()
