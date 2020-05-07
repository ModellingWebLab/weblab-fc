# Function Curation tests

This folder contains the unit/integration tests for the ``fc`` module.

The directory structure is organised as follows:

- `test` - all `test_x.py` files go here
- `test/models` - simple models created specificially for unit testing
- `test/models/real` - real-world models
- `test/protocols` - protocols created specifically for unit testing
- `test/protocols/real` - real-world protocols
- `test/output` - reference data to compare test output against
- `test/input` - data used as input to unit tests

The `/real` subfolders can be used to periodically test the weblab against a large set of real-world test cases.
  
