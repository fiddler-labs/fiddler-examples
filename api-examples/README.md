# Fiddler REST API Examples

This directory contains examples and validation notebooks for working with the Fiddler REST API.

## Directory Structure

- **publishing/** - Event publishing examples (batch, streaming, updates)
  - `python/` - Python REST API examples using `requests` library
  - `typescript/` - TypeScript REST API examples using native `fetch`

## About These Examples

These notebooks serve dual purposes:

1. **Validation** - Test and validate code examples from the official Fiddler documentation
2. **Reference** - Provide production-ready example code for REST API integration

Unlike the quick start guides in `/quickstart/`, these examples focus on:
- REST API usage patterns and best practices
- Language-agnostic integration (not requiring the Python SDK)
- Production-ready error handling and retry logic
- Comprehensive coverage of API operations

## Prerequisites

### Python Examples
```bash
pip install requests ipykernel
```

### TypeScript Examples
For TypeScript notebooks, install the tslab kernel:
```bash
npm install -g tslab
tslab install
```

## Getting Started

1. Set your environment variables:
   ```bash
   export FIDDLER_API_KEY="your-api-key"
   export FIDDLER_ENDPOINT="https://your-instance.fiddler.ai"
   export MODEL_ID="your-model-uuid"
   ```

2. Choose your language and open the corresponding notebook
3. Run all cells to validate the examples

## Documentation

These examples validate code from the official Fiddler documentation:
- [Publishing via REST API](https://docs.fiddler.ai) - Quick-start guide
- [Advanced REST API Publishing](https://docs.fiddler.ai) - Production patterns

## Contributing

When adding new REST API examples:
1. Create language-specific subdirectories under the relevant topic (e.g., `publishing/python/`)
2. Include validation tests for documented code examples
3. Provide production-ready implementations with error handling
4. Update this README with links to the new examples
