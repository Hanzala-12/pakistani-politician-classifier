# MLflow, DVC and EC2 Quick Reference

This document explains how to configure MLflow, DVC remote storage, and perform a minimal EC2 deployment for the project.

MLflow
------

- To run the MLflow server locally (docker-compose includes a service):

```bash
docker-compose -f docker/docker-compose.yml up -d mlflow
```

- To configure tracking URI instead of local `mlruns/` set `MLFLOW_TRACKING_URI` environment variable.

DVC
---

- `dvc.yaml` is present and describes the pipeline stages. A placeholder `dvc.lock` and `.dvc/config` are included.
- Configure remote storage with `dvc remote add -f <name> <url>` or use the helper:

```bash
bash scripts/dvc_setup.sh default s3://my-bucket/dvc-cache
```

- After configuring credentials, use `dvc add`, `dvc push`, and `dvc repro` as needed.

EC2
---

- Use `deploy/ec2_deploy.sh` for a minimal manual deploy (SSH + Docker required):

```bash
bash deploy/ec2_deploy.sh ubuntu@1.2.3.4 /path/to/key.pem mydockeruser/politician-classifier:latest
```

- The GitHub Actions pipeline contains a deploy job that can be used as a reference for automated deploys.
