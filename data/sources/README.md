# Data sources catalog

## What it does

Documents **which** datasets this platform is allowed to touch, including license, publisher, size, and whether automatic download is permitted.

## Why it exists

Acquisition code must fail closed. A URL in Python is not provenance. The catalog is the policy file that a data owner would review.

## Who owns it in an enterprise

Data governance / data platform, with MLOps as a consumer. An individual MLOps engineer does not silently "find a CSV on the internet."

## Inputs

- Publisher landing pages
- License text
- Manual review recorded in `dataset_catalog.yaml`

## Outputs

- Approved dataset ids consumed by `data_acquisition/csv_source.py`
- Human documentation in `docs/data_sources.md`

## Failure scenarios

- Dataset id not in catalog
- `approved_for_automatic_download: false`
- Missing license or download URL

## Security considerations

The catalog contains only public URLs. No credentials.

## Reproducibility

Pin the landing page, publisher, and expected row counts. Optional SHA-256 can be added after the first trusted download.

## How to run it

This folder is documentation plus policy, not an executable.

```bash
# Inspect the catalog
cat data/sources/dataset_catalog.yaml
```

## What replaces it in production

A data catalog (DataHub, Glue, Unity Catalog) plus legal review. The YAML file is the local stand-in.

## Classification used here

| Class | Meaning |
| --- | --- |
| Public sample | Fictional or tutorial data (IBM Telco) |
| Public research | Anonymized research dump with a license (UCI Iranian Churn) |
| Synthetic | Generated in this repo, labeled as fake |
| Local simulation | SQLite / mock API standing in for production systems |
