import os
from dotenv import load_dotenv

load_dotenv()
import s3fs

from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)


# load documents
documents = SimpleDirectoryReader(
    "../../../examples/paul_graham_essay/data/"
).load_data()
print(len(documents))
index = VectorStoreIndex.from_documents(documents)


# set up s3fs
AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]
R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]

assert AWS_KEY is not None and AWS_KEY != ""

s3 = s3fs.S3FileSystem(
    key=AWS_KEY,
    secret=AWS_SECRET,
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    s3_additional_kwargs={"ACL": "public-read"},
)

# save index to remote blob storage
index.set_index_id("vector_index")
# this is {bucket_name}/{index_name}
index.storage_context.persist("llama-index/storage_demo", fs=s3)

# load index from s3
sc = StorageContext.from_defaults(persist_dir="llama-index/storage_demo", fs=s3)
index2 = load_index_from_storage(sc, "vector_index")
