# WarpFactory Validation

This directory contains validation scripts and results for verifying the accuracy and correctness of the WarpFactory Python implementation against published research papers and the original MATLAB implementation.

## Validation Scripts

### Main Validation Script
- **[validate_paper_results.py](validate_paper_results.py)** - Primary validation script that reproduces results from the CQG paper and compares them against expected values

## Validation Results

### Paper Validation
- **[PAPER_VALIDATION.md](PAPER_VALIDATION.md)** - Detailed report on validation against published paper results

### Test Results
- **[VALIDATION_SUMMARY.txt](VALIDATION_SUMMARY.txt)** - Summary of validation test results
- **[VALIDATION_OUTPUT.txt](VALIDATION_OUTPUT.txt)** - Raw output from validation runs
- **[validation_output.txt](validation_output.txt)** - Additional validation output data
- **[TEST_RESULTS_SUMMARY.txt](TEST_RESULTS_SUMMARY.txt)** - Summary of comprehensive test suite results

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
