# WarpFactory Validation & Testing

This directory contains all validation and integration test scripts for verifying the WarpFactory Python implementation.

## Integration Test Scripts

### Workflow Tests
- **[test_basic.py](test_basic.py)** - Basic functionality tests (5-6 tests)
  - Module imports, constants, Minkowski metric, tensor operations, 3+1 decomposition
  - Run with: `python validation/test_basic.py`

- **[test_comprehensive.py](test_comprehensive.py)** - Comprehensive workflow tests
  - All metrics creation, solver module, complete physics pipeline
  - Run with: `python validation/test_comprehensive.py`

### Unit Tests (Separate Location)
The main unit test suite (190 tests) is in **`warpfactory/tests/`**:
- Run with: `pytest warpfactory/tests/`
- 100% passing (190/190 tests)

## Paper Validation

### Validation Scripts
- **[validate_paper_results.py](validate_paper_results.py)** - Reproduces results from arXiv:2404.03095
  - Validates Alcubierre, Van Den Broeck, Modified Time metrics
  - Compares against published paper data
  - Run with: `python validation/validate_paper_results.py`

### Validation Results
- **[PAPER_VALIDATION.md](PAPER_VALIDATION.md)** - Detailed validation report
- **[VALIDATION_SUMMARY.txt](VALIDATION_SUMMARY.txt)** - Quick summary
- **[VALIDATION_OUTPUT.txt](VALIDATION_OUTPUT.txt)** - Full terminal output
- **[TEST_RESULTS_SUMMARY.txt](TEST_RESULTS_SUMMARY.txt)** - Complete test results

## Running Validation

To run the paper validation script:

```bash
cd validation/
python validate_paper_results.py
```

This will execute all validation tests and generate detailed output comparing the Python implementation's results against expected values from the published research.

## Validation Status

The validation suite confirms that the Python implementation accurately reproduces the results from the original MATLAB code and published papers, ensuring scientific accuracy and reliability.

For additional testing, see the [tests/](../tests/) directory for unit and integration tests.
