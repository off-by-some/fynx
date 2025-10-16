FROM python:3.10-slim

# Install fynx from TestPyPI
RUN pip install --index-url https://test.pypi.org/simple/ \
                --extra-index-url https://pypi.org/simple/ \
                fynx

# Basic verification that package installed
RUN python -c "import fynx; print('Fynx', fynx.__version__, 'installed successfully')"
