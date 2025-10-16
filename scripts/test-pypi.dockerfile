FROM python:3.10-slim

# Install fynx from TestPyPI
RUN pip install --index-url https://test.pypi.org/simple/ \
                --extra-index-url https://pypi.org/simple/ \
                fynx

# Test basic functionality
RUN python -c "import fynx; print('Fynx installed successfully')"
RUN python -c "import fynx; print('Version:', fynx.__version__)"
RUN python -c "import fynx; print('Available classes:', [x for x in dir(fynx) if not x.startswith('_')][:8])"
RUN python -c "from fynx import Observable, Store, computed, observable; print('Core imports work')"
RUN python -c "from fynx import observable; obs = observable('test'); print('Observable creation works')"
RUN python -c "from fynx import Store; class TestStore(Store): name = observable('test'); store = TestStore(); print('Store creation works')"
RUN python -c "print('All tests passed.')"