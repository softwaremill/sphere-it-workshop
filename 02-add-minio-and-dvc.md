## Install dvc

```bash
pip install 'dvc[s3]'
```

```
cd data
mkdir imdb
mkdir splits
mkdir misc

mv imdb_*.csv imdb/
mv stopwords misc/
```

```
dvc init
dvc remote add -d minio s3://dvc-bucket
dvc remote modify minio endpointurl http://localhost:9010/minio/
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key miniopassword
```
