# DVC Setup for Datasets

Follow these steps once to configure DVC with MinIO (S3 compatible).

1. Initialize DVC (run from repo root):

```bash
dvc init
```

2. Configure remote pointing at MinIO (edit credentials if needed):

```bash
dvc remote add -d s3remote s3://datasets
dvc remote modify s3remote endpointurl http://minio:9000
dvc remote modify s3remote access_key_id minio
dvc remote modify s3remote secret_access_key minio123
```

3. Package dataset shards (tar/TFRecord) for training and push:

```bash
# example
dvc add data/shards/train.tar
git add data/shards/train.tar.dvc .gitignore
dvc push
```

4. Reproduce datasets with `dvc repro` as you add pipelines.

Maintaining datasets via DVC keeps training runs reproducible and tied to code revisions.

