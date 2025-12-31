# Event Publishing Examples

Examples for publishing inference events to Fiddler using the REST API.

## Available Examples

### Python (`python/`)
- **validate-rest-api-examples.ipynb** - Comprehensive Python examples covering:
  - Batch publishing (file upload + publish)
  - Streaming publishing (direct events)
  - Event updates (PATCH operations)
  - Production-ready publisher with exponential backoff retry logic

### TypeScript (`typescript/`)
- **validate-typescript-rest-api-example.nnb** - TypeScript examples demonstrating:
  - File upload and batch publishing with Job tracking
  - Streaming events with queue submission
  - Event updates with immutability constraints
  - Production-ready publisher class with retry logic

## Key Concepts

### Automatic Processing Mode Selection

Fiddler automatically selects processing mode based on data source:

- **Files (CSV, Parquet)** → Batch processing with Job tracking
  - Returns `job_id` for monitoring via Jobs API
  - Asynchronous background processing
  - Best for large datasets

- **JSON Arrays / Python Lists** → Streaming with queue submission
  - Returns `event_ids` confirming queue submission
  - Lower latency, queue-based processing
  - Best for real-time or small batches

### Environment Constraints

**CRITICAL**: Only PRODUCTION events can be updated. Non-production events (PRE_PRODUCTION) are immutable and cannot be modified via batch OR streaming processes once published.

## Documentation References

These examples validate code from:
- [Publishing via REST API (Quick Start)](https://docs.fiddler.ai/docs/python-client-guides/publishing-production-data/publishing-via-rest-api)
- [Advanced REST API Publishing](https://docs.fiddler.ai/docs/python-client-guides/publishing-production-data/publishing-via-rest-api-advanced)

## Running the Examples

1. Set environment variables (API key, endpoint, model ID)
2. Open the notebook for your preferred language
3. Run all cells to execute the validation tests

Each notebook is self-contained and demonstrates all publishing workflows end-to-end.
