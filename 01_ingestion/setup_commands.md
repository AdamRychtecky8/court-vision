# GCP Environment Setup Commands
## court-vision | PSTAT 135 Final Project

*This file is the reproducibility record for the cloud environment.
Update it whenever a setting or version changes.*

---

## 1. First-time Authentication & Project Config

Run once after installing gcloud. Sets your default project and region so
you never have to specify them on every command:

```powershell
gcloud init
gcloud auth login
gcloud config set project <YOUR_GCP_PROJECT_ID>
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
gcloud config set dataproc/region us-central1
```

Verify it took:
```powershell
gcloud config list
gcloud auth list
```

---

## 2. Enable Required APIs

Run once per project. Safe to re-run at any time — it skips already-enabled ones:

```powershell
gcloud services enable dataproc.googleapis.com compute.googleapis.com storage.googleapis.com cloudresourcemanager.googleapis.com
```

---

## 3. IAM Fix (if cluster creation fails with a permissions error)

If you see "missing permissions: storage.buckets.get" when creating a cluster,
run this and wait ~60 seconds before retrying the create:

```powershell
gcloud projects add-iam-policy-binding <YOUR_GCP_PROJECT_ID> --member="serviceAccount:<YOUR_PROJECT_NUMBER>-compute@developer.gserviceaccount.com" --role="roles/dataproc.worker"
```

---

## 4. Storage Bucket

| Field       | Value                          |
|-------------|-------------------------------|
| Bucket name | `<YOUR_GCS_BUCKET>`            |
| Location    | US (multi-region)              |
| Project     | `<YOUR_GCP_PROJECT_ID>`         |

**GCS folder structure:**
```
gs://<YOUR_GCS_BUCKET>/
├── raw/
│   ├── shots/       ← raw shot CSV (Phase 1)
│   └── pbp/         ← raw play-by-play files (Phase 1 part 2)
└── processed/       ← Parquet outputs (Phase 2 onward)
```

Check bucket exists:
```powershell
gcloud storage buckets list --format="table(name,location)"
```

---

## 5. Cluster Lifecycle

**Cluster config:**
| Field        | Value          |
|--------------|----------------|
| Name         | mycluster      |
| Type         | Single Node    |
| Machine      | e2-highmem-4   |
| Region       | us-central1    |
| Components   | Jupyter + Component Gateway |

**Create** (run this if the cluster has been deleted):
```powershell
gcloud dataproc clusters create mycluster --region=us-central1 --single-node --master-machine-type=e2-highmem-4 --optional-components=JUPYTER --enable-component-gateway
```
*Takes ~90 seconds. Cluster starts in RUNNING state immediately after.*

**Start** (after a stop — faster than recreating):
```powershell
gcloud dataproc clusters start mycluster
```

**Stop — run this at the end of every session:**
```powershell
gcloud dataproc clusters stop mycluster
```

**Delete** (if you want zero idle cost; requires recreate next session):
```powershell
gcloud dataproc clusters delete mycluster
```

**Check status:**
```powershell
gcloud dataproc clusters list
```

---

## 6. Submit a PySpark Job

General pattern — run from the `court-vision/` project root in VSCode terminal:
```powershell
gcloud dataproc jobs submit pyspark SCRIPT_NAME.py --cluster=mycluster
```

Passing GCS input/output paths as arguments to your script:
```powershell
gcloud dataproc jobs submit pyspark SCRIPT_NAME.py --cluster=mycluster -- gs://<YOUR_GCS_BUCKET>/raw/shots/ gs://<YOUR_GCS_BUCKET>/processed/
```

---

## 7. Upload Local Files to GCS

Single file:
```powershell
gcloud storage cp data\raw\filename.csv gs://<YOUR_GCS_BUCKET>/raw/shots/
```

Entire local folder:
```powershell
gcloud storage cp -r data\raw\pbp\ gs://<YOUR_GCS_BUCKET>/raw/pbp/
```

Verify the upload landed:
```powershell
gcloud storage ls gs://<YOUR_GCS_BUCKET>/raw/shots/
```

---

## 8. Smoke Test

Confirms end-to-end PySpark execution. Run from `court-vision/` root:

```powershell
gcloud dataproc clusters start mycluster
gcloud dataproc jobs submit pyspark smoke_test.py --cluster=mycluster
gcloud dataproc clusters stop mycluster
```

Expected: a line containing `5` appears in the job output.
Status at bottom should read: `state: DONE` and `Agent reported job success`.

---

## 9. Cost Reference

| State   | Approx. cost          |
|---------|-----------------------|
| RUNNING | ~$0.22/hr (e2-highmem-4 single-node) |
| STOPPED | Minimal disk cost only |
| DELETED | $0                    |

**Rule: stop the cluster at the end of every session, no exceptions.**
Cluster create time is ~90 seconds, so deleting between sessions is fine if
you want zero idle cost — paste the create command above to spin it back up.

---

## 10. Environment Versions

| Tool       | Version              |
|------------|----------------------|
| Python     | 3.13.5               |
| Git        | 2.48.1.windows.1     |
| gcloud SDK | 570.0.0              |
| Cluster    | Dataproc 2.x default |

*Update Dataproc image version once confirmed from cluster details page.*

---

## 11. Troubleshooting Quick Reference

| Symptom | Fix |
|---------|-----|
| `NOT_FOUND: Cluster mycluster` | Cluster was deleted — run the create command in section 5 |
| `missing permissions: storage.buckets.get` | Run the IAM fix in section 3, wait 60s, retry |
| Command not recognized after install | Close and reopen the VSCode terminal; PATH needs refresh |
| Download blocked on UCSB network | Use Google Cloud Shell (browser terminal at console.cloud.google.com) |
