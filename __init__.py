from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .core import register_wan_experiments_models

# Import mad science nodes if available
try:
    from .nodes_madscience import NODE_CLASS_MAPPINGS as MADSCI_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MADSCI_DISPLAY_MAPPINGS
    NODE_CLASS_MAPPINGS.update(MADSCI_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(MADSCI_DISPLAY_MAPPINGS)
except ImportError:
    pass  # Mad science nodes not present

from .nodes_s2v_bernini import NODE_CLASS_MAPPINGS as S2V_BERNINI_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as S2V_BERNINI_DISPLAY_MAPPINGS
NODE_CLASS_MAPPINGS.update(S2V_BERNINI_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(S2V_BERNINI_DISPLAY_MAPPINGS)

# Register model enhancements at import time
register_wan_experiments_models()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]