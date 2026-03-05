Here's the concentrated requirements doc, alert-discussion-only:

---

## Thumbtack Bulk Alert Creation Utility — V0 Requirements

**Source:** Thumbtack–Fiddler Office Hours, Feb 19, 2026
**Stakeholder:** Oleksandr Pryimak (Thumbtack)
**Owner:** Saif Raja (Fiddler CS)

---

### The Core Pain Point (Oleksandr's words, verbatim)

> *"We have zero alerts. We want a lot of alerts. Right now they onboard a model, say everything's great — we have zero business value. We are not actually monitoring anything and no one goes to Fiddler AI. It sucks. We want it not to suck."*

The problem has been discussed for months with no resolution. Zero alerts = zero passive monitoring = Fiddler is invisible in Thumbtack's day-to-day.

---

### Root Cause of Zero Alerts

- Creating alerts is entirely manual, model by model
- Each alert requires involvement from the data scientist who owns that model
- DS owners are time-constrained; alert setup is deprioritised to zero
- No defaults exist at model onboarding time — customers start blind

---

### Oleksandr's Stated Ideal State

- Alerts created **automatically** with reasonable defaults the moment a model is onboarded
- Defaults based on **historical data** — 2σ or 3σ thresholds
- Let them be **noisy initially** — customers can adjust from there
- Current state: zero → target state: everything has *something*

---

### What Was Committed

| Item | Owner | Deadline |
|---|---|---|
| V0 Python script for bulk alert creation | Saif | ~Mar 4, 2026 (2 weeks from Feb 19) |
| Script handover to Thumbtack team | Saif | Same |
| Thumbtack runs the script on their account | Oleksandr / John / James / Kevin | After handover |

---

### Technical Constraints

- **Tooling:** Fiddler Python SDK + REST API (both confirmed to expose alert creation endpoints)
- **Execution boundary:** Saif writes the script; Thumbtack runs it. Saif does NOT execute on the customer account — explicitly agreed.
- **Format:** Standalone Python script — pulls all models, creates alerts for all of them
- **Threshold mechanism:** Fiddler's recently launched **standard deviation-based alert thresholds** (new capability, key enabler)

---

### Strategy: Overshoot First, Tune Later

Both parties converged on this approach:

- **Don't** try to scope the ideal alert set upfront — requires DS owner availability that doesn't exist
- **Do** overshoot volume, then use noise/complaint signal to tune down in V1+
- Oleksandr explicitly endorsed this: *"Let's overshoot once"*
- Oleksandr offered to connect Saif with model owners post-V0 for tuning — but with a caveat: DS owners will likely say everything is fine, so their input needs to be filtered through judgment

---

### Hard Constraint Acknowledged by Both

> Fiddler cannot auto-determine the optimal alert. There is no free lunch. A sigma-based default on historical data is the best approximation — model owners need to tune from there.

---

### Open Questions (Not Resolved on the Call)

- Which alert **types** to create per model in V0? (drift, volume, data quality, custom?) — needs an opinionated list
- Default sigma level: **2σ or 3σ?**
- What's Thumbtack's preferred **notification channel**? (email / Slack / webhook)
- Any models to **exclude** from bulk creation?
- Script re-run safety — idempotency or clear documentation needed

---

### Out of Scope for V0

- LLM-specific alerts (toxic inputs, jailbreak, guardrails) — deferred
- Evals / agentic monitoring — no live data at Thumbtack yet
- Auto-tuning or ML-based threshold selection
- Fiddler Utilities capability — reserved for next meeting