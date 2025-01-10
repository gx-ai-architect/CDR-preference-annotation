# we make scripts modules callable, so that it can be used in Jupyter Notebook.

from . import process_preference_data
from .process_preference_data import load_and_format_dpo
