
**Decision:** Use Python 3.13.5 during development.

**Reason:** It is already installed and working. The project uses a virtual environment and `requirements.txt`, making it straightforward to recreate the environment with Python 3.12 later if a library compatibility issue arises.

**Consequence:** We avoid unnecessary setup work now while keeping a clear recovery path if compatibility problems appear.
