__version__ = "2.5.6"

from flash_sigmoid.flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    # We do not support varlen and kvcache variants.
)
