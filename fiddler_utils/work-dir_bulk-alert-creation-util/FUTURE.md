# Future Improvements & Deferred Features

This document captures features explicitly deferred from V0, with rationale and unblocking criteria.

---

## Deferred: REST API Direct Access for Alert CRUD

**Description:** Bypass the Fiddler Python SDK and invoke the REST API directly (`/v3/alert-rules`) for operations not exposed by the SDK (e.g., bulk delete by filter, batch create, listing with richer filters).

**Why deferred:** The SDK covers create, update, delete, list, and notification config — sufficient for V0. Direct API access adds complexity (auth token management, pagination handling, error parsing) without clear incremental value yet.

**What would unblock it:**
- Discovery of an operation that cannot be performed via the SDK
- Performance needs (batch create vs. sequential create)
- Need to read alert properties not exposed by the SDK entity

**Where it matters:** The `AlertManager._create_asset()` method currently creates alerts one at a time via `fdl.AlertRule().create()`. A REST API batch endpoint would improve throughput for large environments (500+ alerts).

---

## Deferred: Two-Way Sync / Desired-State Reconciliation

**Description:** A `sync` mode that ensures the alert state in Fiddler matches the profile exactly: create missing alerts, update drifted alerts, and *delete* alerts that exist in Fiddler but are not in the profile.

**Why deferred:** Deletion of alerts the user may have manually created is dangerous without a clear "ownership" model. The customer is expected to tune alerts after bulk creation — deleting those tuned alerts on a sync run would destroy their work.

**What would unblock it:**
- An ownership tagging mechanism (e.g., a `category` field on alerts set to `'bulk_managed'` to distinguish tool-created from user-created)
- Customer validation that sync-with-delete is desired
- State file tracking which alerts were created by the tool

**Current alternative:** Use `mode='update'` to keep existing alerts in sync with profile for mutable fields. Manually delete unwanted alerts.

---

## Deferred: Alert-to-Profile Reverse Mapping

**Description:** Given a set of existing alerts in Fiddler, reconstruct an `AlertProfile` object that represents them. Enables "export current alerts as a profile" workflow.

**Why deferred:** The existing `AlertManager.export_assets()` already extracts alert data. Converting that to `AlertSpec` objects requires mapping metric IDs back to categories and inferring `columns_source` from column lists, which is lossy.

**What would unblock it:** Customer request for "save my current alert config as a reusable profile."

---

## Deferred: State Tracking Between Runs

**Description:** Persist a record of what the tool created, which profile was used, and when it ran. Enables: audit trail, selective undo, "what changed since last run" diffing.

**Why deferred:** Adds file I/O or database dependency. Current approach (list existing alerts, compare by name) is stateless and sufficient for V0.

**What would unblock it:**
- Need for audit compliance
- Multi-user environments where runs need to be attributed
- Undo/rollback feature request

---

## Deferred: Alert Count Guardrails

**Description:** Cap the number of per-column alerts created per model. A model with 200 input columns generates 600+ per-column alerts (drift + null + range), which can overwhelm notification channels.

**Why deferred:** Current design logs a warning at 100+ columns. The user can control this via `columns_source` (explicit list instead of `'inputs'`). Hard caps require policy decisions (which columns to prioritize?).

**What would unblock it:**
- Customer hitting notification fatigue from too many alerts
- Feature to auto-select "top N most important columns" (e.g., by feature impact)

---

## Deferred: Model Rename Handling

**Description:** If a model is renamed after alerts are created, the name-based idempotency breaks — re-runs create duplicate alerts because the name template generates different names.

**Why deferred:** Model renames are rare. The tool's name template uses `{model_name}` which is resolved at creation time. Model renames don't retroactively change existing alert names.

**What would unblock it:**
- Using alert IDs instead of names for idempotency (requires state tracking)
- Using `(model_id, metric_id, columns)` tuple as the identity key instead of name

---

## Deferred: Fetching Alert Rules via REST API (Bypassing SDK)

**Description:** Use the `/v3/alert-rules` REST endpoint directly to list alert rules with richer filtering than the SDK provides (e.g., filter by `threshold_type`, `category`, or `created_by`).

**Why deferred:** The SDK's `AlertRule.list(model_id=...)` is sufficient for V0. It returns full alert objects with all properties.

**What would unblock it:**
- Need to filter alerts by fields not exposed in SDK list parameters
- Need to list alerts across models without iterating (cross-model query)

---

## Deferred: Segment-Aware Alert Profiles

**Description:** Create alerts scoped to specific data segments (e.g., "US traffic only", "high-value customers"). The `AlertSpec` has a placeholder for `segment_id` but profiles don't currently resolve segments.

**Why deferred:** Segments are model-specific — you can't define a generic "segment" in a profile because segment IDs differ per model. Would need a segment name → ID resolution step.

**What would unblock it:** Customer request for segment-scoped alerts + a segment naming convention.
