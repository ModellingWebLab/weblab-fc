# Contributing to Web Lab FC

*Under construction*

## Workflow

## Installation

See [README.md](./README.md)

## Coding style guidelines

We use the [PEP8 coding style recommendations](https://www.python.org/dev/peps/pep-0008/) and check them with [flake8](http://flake8.pycqa.org/en/latest/).

Exceptions are listed in [setup.cfg](./setup.cfg) under the heading `[flake8]`.

The order of ``import`` statements is checked by [isort](https://pypi.org/project/isort/) according to [its own rules](https://github.com/timothycrosley/isort#how-does-isort-work).

### Naming

### Spelling

### Python versions

Python 3.6+ only.

## Testing

Currently migrating to [pytest](https://docs.pytest.org/en/latest/), see [README.md](./README.md)

## Profiling performance

Performance tests can be run using [pytest-profiling](https://pypi.org/project/pytest-profiling/).
For example
```sh
pytest --profile-svg test/test_algebraic_models.py
```
will show profiling output for the algebraic models test, and create an SVG file in the `prof` directory.

## Documentation

[Docstrings](https://www.python.org/dev/peps/pep-0257/) are written in [reStructuredText](http://docutils.sourceforge.net/docs/user/rst/quickref.html), a markup language designed specifically for writing [technical documentation](https://en.wikipedia.org/wiki/ReStructuredText).

For arguments, return types, and exceptions raised, we use the [basic Sphinx autodoc syntax](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#python-signatures)

