
__all__ = ["TARGET_DIR", "DATA_DIR"]


import os
from pathlib import Path

TARGET_DIR = Path(os.environ["TARGET_DIR"])
DATA_DIR   = Path(os.environ["DATA_DIR"])

from . import utils
__all__.extend( utils.__all__ )
from .utils import *

from . import validation
__all__.extend( validation.__all__ )
from .validation import *

from . import tasks
__all__.extend( tasks.__all__ )
from .tasks import *

from . import pipelines
__all__.extend( pipelines.__all__ )
from .pipelines import *
