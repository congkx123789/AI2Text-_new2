# 🚀 AI2Text Production Roadmap (Nov-Dec 2025)

## 🎯 Objective
Ship a stable, contract-first platform with p95 latency/SLOs met for Gateway, ASR streaming, and Search; enable staged rollouts and cost-aware autoscaling.

---

## 📅 Milestones & Timeline

### M0 — Kickoff & Readiness (Mon, Nov 3, 2025)
- ✅ Contracts repo at v1.1.0 (OpenAPI + AsyncAPI)
- ✅ ai2text-common v0.1.x published
- ✅ CI templates available for all service repos
- **Go/No-Go** to enter Sprint 1

### M1 — Contract-First + Platform Baseline (Fri, Nov 14, 2025)
- Contract tests run in CI across gateway, asr, search
- Platform repo delivers Helm charts + env overlays
- NATS streams + DLQs defined
- Basic Grafana dashboards live (latency, error rate, DLQ backlog)

### M2 — Streaming & Search Performance (Fri, Nov 28, 2025)
- ASR streaming p95 < 500ms E2E on 50 concurrent streams (dev)
- Search p95 < 50ms (HNSW tuned, cache on hot queries)
- Embeddings pipeline idempotent; retry/poison-pill handling

### M3 — Feature Freeze / RC (Fri, Dec 12, 2025)
- Freeze branch `release/2025.12`
- RC1 images promoted to stage
- Security review + perf report signed
- GA checklist complete

### M4 — GA & Hypercare (Mon, Dec 15 → Fri, Dec 19, 2025)
- Blue/green for Gateway; rolling for stateless services
- SLOs monitored; error budget burn ≤ 20%
- Post-launch review & backlog triage

---

## 📊 Weekly Sprint Plan

### Week of Nov 3 (Sprint 1): Contracts & Common

**Work:**
- Extract OpenAPI/AsyncAPI specs
- Add versioning rules and codegen Makefile
- Publish ai2text-common (logging/tracing/events)
- Add contract-test job to CI templates

**Exit Criteria:**
- ✅ contracts@v1.1.0 tagged
- ✅ common@0.1.x on registry
- PRs blocked if contract tests fail
- Demo: generated Python/TS clients used by one service

**Owners:**
- Contracts: Lead A
- Common: Lead B
- CI: DevOps

---

### Week of Nov 10 (Sprint 2): Platform & Gateway Hardening

**Work:**
- Helm charts per service + env overlays (dev/stage/prod)
- NATS JetStream: subjects, retention, DLQs; alert rules
- Gateway: RS256 JWT, rate limit, structured errors

**Exit Criteria:**
- Dev + Stage clusters deploy cleanly via Helm
- DLQs visible in Grafana; alert fire-drill demo
- Gateway p95 < 150ms on 1k rps synthetic

**Owners:**
- Platform: DevOps
- Gateway: Backend

---

### Week of Nov 17 (Sprint 3): ASR Streaming & Batch Split

**Work:**
- Separate streaming vs batch pipeline (distinct deployments & HPA)
- Back-pressure and queue-drain logic
- Canary strategy for streaming
- Latency instrumentation E2E (WebSocket → event → response)

**Exit Criteria:**
- Streaming p95 < 500ms @ 50 streams
- Batch throughput target met
- Canary recipe documented; rollback tested

**Owners:**
- ASR: ML Eng + Backend

---

### Week of Nov 24 (Sprint 4): Embeddings + Search

**Work:**
- Idempotent embedding jobs; dedupe on content hash
- Qdrant HNSW tuning (M, efConstruction)
- Hot-query cache
- Re-ranker hook (optional, behind a flag)

**Exit Criteria:**
- Search p95 < 50ms on 10M vectors (stage)
- Zero duplicate vectors on re-ingest

**Owners:**
- Embeddings/Search: Backend + Data

---

### Week of Dec 1 (Sprint 5): Ingestion + Metadata

**Work:**
- Ingestion S3 lifecycle & checksum verification
- Metadata write path: idempotency keys
- Migration V2 indexes
- API quotas and 429 policy
- Client-side retry guidance

**Exit Criteria:**
- Metadata write p95 < 40ms
- Ingestion retry success > 99%
- DB migration dry-run + rollback plan

**Owners:**
- Ingestion/Metadata: Backend + DBA

---

### Week of Dec 8 (Sprint 6): Freeze, RC, and Drills

**Work:**
- Branch `release/2025.12`; tag RC1/RC2 as needed
- Security review (secrets, SBOM, container scans)
- DR tabletop + restore test
- Load/perf report finalized

**Exit Criteria:**
- RC promoted to Stage; all gates green
- GA checklist signed by Leads (Eng, QA, Sec)

**Owners:**
- All teams
- Release captain: DevOps Lead

---

## 🎯 KPIs & SLO Gates (Must Be Green to Ship)

| Service | Metric | Target | Window |
|---------|--------|--------|--------|
| **Gateway** | p95 latency | < 150ms | 5min |
| **Gateway** | Availability | 99.9% | - |
| **Gateway** | 5xx rate | < 1% | 5min |
| **ASR Streaming** | E2E p95 | < 500ms | - |
| **ASR Streaming** | Drop rate | < 0.1% | - |
| **Search** | p95 latency | < 50ms | - |
| **Search** | p99 latency | < 120ms | - |
| **Search** | Recall@10 | ≥ baseline | - |
| **Embeddings** | Job success | ≥ 99.5% | - |
| **Embeddings** | Duplicate rate | ≈ 0% | - |
| **Metadata** | Write p95 | < 40ms | - |
| **Metadata** | Error rate | < 0.5% | - |
| **Cost** | GPU utilization | > 50% avg | - |
| **Cost** | Infra cost | Within budget | - |

---

## 🔄 Release Management

### Branching Strategy
- `main` → `release/2025.12`
- Hotfixes via `hotfix/*`

### Versioning
- Services use **SemVer**
- Contracts bump **major** for breaking changes

### Promotion Flow
```
dev (auto) → stage (manual, signed) → prod (manual with change ticket)
```

### Deployment Strategy
- **Gateway**: Blue/green
- **Others**: Rolling updates
- **Workers**: Queue-drain prior to updates

---

## ⚠️ Risk Register & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| ASR quality on Vietnamese diacritics | High | Keep nlp-post as separate stage; add rule-based fallback; collect error samples |
| NATS backlog growth | High | Per-consumer quotas + DLQ alerts; autoscale workers; dead-letter drill weekly |
| Qdrant RAM pressure | Medium | Apply sharding/compaction; ef tuning; cold-tier vectors to disk |
| Schema drift | Medium | Contract-tests mandatory + DB migration checks in CI |
| GPU cost spikes | High | Scheduled scaling windows; batch jobs off-peak; usage dashboards |

---

## 👥 RACI Matrix

| Role | Responsibility |
|------|----------------|
| **Release Captain** | DevOps Lead (A/R) |
| **Contracts** | Architect (A), Service Leads (R), QA (C) |
| **Platform** | DevOps (A/R), Backend (C) |
| **ASR** | ML Eng (A/R), Backend (R), QA (C) |
| **Search/Embeddings** | Backend (A/R), Data Eng (R), QA (C) |

**Legend:** A=Accountable, R=Responsible, C=Consulted, I=Informed

---

## ✅ GA Deliverables Checklist

### Code & Contracts
- [ ] contracts@v1.1.0 tagged and published
- [ ] ai2text-common@0.1.x published to registry
- [ ] All services have contract tests in CI
- [ ] Generated clients/SDKs available

### Infrastructure
- [ ] Helm charts for all services (dev/stage/prod)
- [ ] NATS streams + DLQs configured
- [ ] Database migrations with rollback plans
- [ ] Secrets management configured

### Observability
- [ ] Grafana dashboards (latency, errors, DLQ backlog)
- [ ] Prometheus alerts configured
- [ ] Distributed tracing enabled
- [ ] SLO tracking dashboards

### Security
- [ ] Security review completed
- [ ] SBOM generated for all images
- [ ] Container vulnerability scans passing
- [ ] Secrets rotated and secured

### Performance
- [ ] All SLOs met in stage environment
- [ ] Load testing completed and documented
- [ ] Performance report signed off
- [ ] Cost/perf telemetry validated

### Operations
- [ ] Runbooks for all services
- [ ] DR/restore tested
- [ ] Rollback procedures documented
- [ ] On-call rotation established

### Documentation
- [ ] API documentation published
- [ ] Client integration guides
- [ ] Operations playbooks
- [ ] Architecture decision records (ADRs)

---

## 🤔 Decision Points

### 1. Re-ranker On/Off at GA
**Based on:** perf/quality tradeoff
**Deadline:** Sprint 4 (Nov 24)
**Owner:** Backend Lead

### 2. Monorepo vs Multi-repo
**Based on:** Sprint 3 ops feedback
**Deadline:** Sprint 3 (Nov 17)
**Owner:** DevOps Lead

### 3. Cloud Region (SG vs Local)
**Based on:** Latency tests
**Deadline:** Sprint 2 (Nov 10)
**Owner:** Infrastructure Lead

---

## 📢 Communication & Demos

### Weekly Cadence
- **Friday Demo**: Publish perf snapshot and risk deltas
- **Daily Standups**: 15min sync
- **Incidents**: Shared channel with runbook links

### GA Announcement
- Internal how-to guide for clients/SDKs
- External announcement (if applicable)
- Post-mortem and lessons learned

---

## 📈 Success Metrics

### Technical
- All SLOs green for 7 consecutive days in prod
- Zero P0/P1 incidents in first 2 weeks
- Error budget burn < 20%

### Operational
- MTTR < 30 minutes
- Deployment frequency: Multiple per day
- Change failure rate < 5%

### Business
- Cost per transaction within budget
- GPU utilization > 50%
- Infrastructure cost optimized

---

## 🎯 Next Steps (Immediate Actions)

1. **This Week (Nov 3-7)**
   - [ ] Finalize contracts@v1.1.0
   - [ ] Set up contract test framework
   - [ ] Publish common library
   - [ ] Hold Go/No-Go meeting

2. **Next Week (Nov 10-14)**
   - [ ] Deploy to dev environment
   - [ ] Validate Helm charts
   - [ ] Configure NATS streams
   - [ ] Run first performance baseline

3. **Following Week (Nov 17-21)**
   - [ ] Implement ASR streaming optimizations
   - [ ] Add latency instrumentation
   - [ ] Test canary deployments
   - [ ] Document rollback procedures

---

**Status**: 🟢 On Track | 🟡 At Risk | 🔴 Blocked

**Last Updated**: November 1, 2025

**Questions?** Contact Release Captain or check `#ai2text-platform` channel

