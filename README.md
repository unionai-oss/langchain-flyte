# langchain-flyte
Repo that demonstrates hows to build langchain data loaders in Flyte to help build production grade langchain apps

# How to run it?

- Local execution!

```bash
pyflyte run ingest_wf.py ingest
```

- Remote Execution
```bash
pyflyte run --remote ingest_wf.py ingest
```


# TODOs
1. Test remote
2. Imagespec to manage dependencies
3. Docment TypeTransformer (or pydantic) so that UI can visualize this better
4. Better vector store management, so that during the reduction step, vector stores can be loaded lazily so that memory requirements can be lower.
5. Visualization through FlyteDecks

# Serving TODOS
1. Upadate the langchain serving example to load the FAISS vector store from flyte registry
2. add this to UnionML as a default way of writing prod ready Langchain apps
