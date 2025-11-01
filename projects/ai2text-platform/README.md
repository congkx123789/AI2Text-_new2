# AI2Text Platform

**Infrastructure as Code** for AI2Text microservices.

This repository contains:
- Helm charts for all services
- Environment overlays (dev/stage/prod)
- Database migrations
- NATS streams configuration
- Observability dashboards and alerts

## Structure

```
platform/
├── helm/
│   ├── charts/          # Service Helm charts
│   └── values/          # Environment overlays
│       ├── dev/
│       ├── stage/
│       └── prod/
├── migrations/           # Database migrations
│   └── metadata-db/
├── nats/                 # NATS streams config
│   └── streams.yaml
└── observability/        # Dashboards & alerts
    ├── dashboards/
    └── alerts/
```

## Deployment

### Development

```bash
helm upgrade --install ai2text helm/charts/ai2text \
  -f helm/values/dev/values.yaml \
  --namespace ai2text-dev
```

### Staging

```bash
helm upgrade --install ai2text helm/charts/ai2text \
  -f helm/values/stage/values.yaml \
  --namespace ai2text-stage
```

### Production

```bash
helm upgrade --install ai2text helm/charts/ai2text \
  -f helm/values/prod/values.yaml \
  --namespace ai2text-prod
```

## Database Migrations

```bash
# Run migrations
kubectl run migrations --image=postgres:15 \
  --rm -i --restart=Never \
  --env="PGPASSWORD=$DB_PASSWORD" \
  -- psql -h postgres-host -U postgres -d asrmeta \
  -f migrations/metadata-db/V1__init.sql
```

## NATS Streams

```bash
# Apply stream configuration
nats stream add RECORDINGS --subjects "recording.*" \
  --storage file --max-age 7d
```


