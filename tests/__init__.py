import sys
from pathlib import Path

# Add the project root directory to sys.path to allow absolute imports from 'app' and 'tests'
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
