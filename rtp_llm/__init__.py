import time

st = time.time()
try:
    import triton
except ImportError:
    pass

# load th_transformer.so
# Import internal models to register them
from rtp_llm.utils.import_util import has_internal_source
from rtp_llm.utils.torch_patch import *
from rtp_llm.utils.triton_compile_patch import enable_compile_monitor

from .ops import *

# check triton version
# if triton.__version__ < "3.4":
#     enable_compile_monitor()


# enable_compile_monitor()


if has_internal_source():
    # The frontend-only (slim) image ships a partial internal_source tree
    # (tokenizers / openai_renderers) but not internal_source.rtp_llm.models_py,
    # so has_internal_source() can be True while this submodule is absent.
    # models_py only holds model-execution / backend kernel registrations the
    # frontend never runs, so a missing module must degrade gracefully instead
    # of crashing the process at import time.
    try:
        import internal_source.rtp_llm.models_py
    except ImportError as e:
        import logging

        logging.warning(
            "internal_source.rtp_llm.models_py unavailable, skipping "
            "(expected on frontend slim image): %s",
            e,
        )


consume_s = time.time() - st
print(f"import in __init__ took {consume_s:.2f}s")
