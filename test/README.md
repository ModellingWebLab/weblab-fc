# Function Curation tests

This folder contains the unit/integration tests for the ``fc`` module.

The directory structure is organised as follows:

- `test` - all `test_x.py` files go here
- `test/models` - simple models created specificially for unit testing
- `test/models/real` - real-world models, used by some unit tests (and will eventually be scanned periodically to test if there's any real-world edge-cases we've missed out).
- `test/protocols` - protocols created specifically for unit testing
- `test/protocols/real` - real-world protocols
- `test/output` - reference data to compare test output against
- `test/input` - data used as input to unit tests
  
