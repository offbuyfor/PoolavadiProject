# Deploy API Endpoint + Batch Prediction (Cost-Effective)

This guide uses one Docker image for both:
- Cloud Run service (real-time API)
- Cloud Run Job (batch predictions)

## 1) Prerequisites

- Install and login to Google Cloud CLI:
  - `gcloud auth login`
  - `gcloud auth application-default login`
- Set your project:
  - `gcloud config set project YOUR_PROJECT_ID`
- Enable APIs:
  - `gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com`

## 2) Build and push image

```powershell
$REGION="us-central1"
$REPO="ml-inference"
$IMAGE="model-serving"
$TAG="v1"

gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION
gcloud auth configure-docker "$REGION-docker.pkg.dev"

gcloud builds submit --tag "$REGION-docker.pkg.dev/$env:CLOUDSDK_CORE_PROJECT/$REPO/$IMAGE`:$TAG"
```

## 3) Deploy API endpoint (Cloud Run)

```powershell
$SERVICE_NAME="model-api"
$MODEL_PATH="models/best_model.joblib"
$LABEL_ENCODER_PATH="models/label_encoder.joblib"

gcloud run deploy $SERVICE_NAME `
  --image "$REGION-docker.pkg.dev/$env:CLOUDSDK_CORE_PROJECT/$REPO/$IMAGE`:$TAG" `
  --region $REGION `
  --allow-unauthenticated `
  --min-instances 0 `
  --max-instances 2 `
  --memory 512Mi `
  --cpu 1 `
  --set-env-vars "MODEL_PATH=$MODEL_PATH,LABEL_ENCODER_PATH=$LABEL_ENCODER_PATH"
```

Cost tips:
- Keep `--min-instances 0` (scale to zero)
- Keep `--max-instances` low until traffic grows
- Start with `512Mi`, then increase only if needed

## 4) Test API

```powershell
$URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)")
Invoke-RestMethod -Method Get -Uri "$URL/health"
```

Example prediction call:

```powershell
$body = @{
  rows = @(
    @{ feature1 = 1.2; feature2 = 3.4 }
  )
  return_proba = $true
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri "$URL/predict" -ContentType "application/json" -Body $body
```

## 5) Create bucket for batch I/O

```powershell
$BUCKET="gs://YOUR_BATCH_BUCKET"
gsutil mb -l $REGION $BUCKET
```

Upload input:

```powershell
gsutil cp .\sample_predictions.csv "$BUCKET/incoming/input.csv"
```

## 6) Deploy Cloud Run Job for batch predictions

```powershell
$JOB_NAME="model-batch-job"

gcloud run jobs create $JOB_NAME `
  --image "$REGION-docker.pkg.dev/$env:CLOUDSDK_CORE_PROJECT/$REPO/$IMAGE`:$TAG" `
  --region $REGION `
  --memory 1Gi `
  --cpu 1 `
  --max-retries 1 `
  --set-env-vars "MODEL_PATH=$MODEL_PATH,LABEL_ENCODER_PATH=$LABEL_ENCODER_PATH" `
  --command python `
  --args batch_predict.py,--input,gs://YOUR_BATCH_BUCKET/incoming/input.csv,--output,gs://YOUR_BATCH_BUCKET/predictions/output.csv,--return-proba
```

Run batch job:

```powershell
gcloud run jobs execute $JOB_NAME --region $REGION --wait
```

Fetch output:

```powershell
gsutil cp gs://YOUR_BATCH_BUCKET/predictions/output.csv .\batch_output.csv
```

## 7) Update batch inputs per run (without recreating job)

```powershell
gcloud run jobs update $JOB_NAME `
  --region $REGION `
  --args batch_predict.py,--input,gs://YOUR_BATCH_BUCKET/incoming/new_file.csv,--output,gs://YOUR_BATCH_BUCKET/predictions/new_output.csv,--return-proba

gcloud run jobs execute $JOB_NAME --region $REGION --wait
```

## 8) Local run (same code path)

API:
```powershell
uvicorn serve:app --host 0.0.0.0 --port 8000
```

Batch:
```powershell
python batch_predict.py --input sample_predictions.csv --output output_local.csv --return-proba
```

## 9) Recommended low-cost defaults

- API:
  - `min-instances=0`
  - `max-instances=2`
  - `memory=512Mi` to start
- Batch:
  - Run only on demand or scheduled
  - Keep retries low (`1`) unless input source is flaky
  - Use chunking (`--chunksize`) for large input files
