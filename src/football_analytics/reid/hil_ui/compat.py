"""Streamlit API compatibility helpers for pinned football-hil-ui (1.37.1)."""

from __future__ import annotations

import inspect
from typing import Any


def streamlit_image(st_module: Any, image: Any, *, caption: str | None = None, **kwargs: Any) -> Any:
    """Call ``st.image`` with the width kwarg supported by the installed Streamlit.

    Streamlit 1.37.x accepts ``use_column_width`` and rejects ``use_container_width``.
    Newer Streamlit versions flipped the preference. This wrapper never upgrades packages.
    """
    sig = inspect.signature(st_module.image)
    params = sig.parameters
    call_kwargs = dict(kwargs)
    if caption is not None:
        call_kwargs["caption"] = caption

    wants_full = bool(call_kwargs.pop("use_container_width", False) or call_kwargs.pop("use_column_width", False))
    if wants_full:
        if "use_column_width" in params:
            call_kwargs["use_column_width"] = True
        elif "use_container_width" in params:
            call_kwargs["use_container_width"] = True
    # Drop unknown kwargs that would crash older APIs
    filtered = {k: v for k, v in call_kwargs.items() if k in params or k in {"caption", "image"}}
    # ``image`` is positional
    filtered.pop("image", None)
    return st_module.image(image, **filtered)


def streamlit_image_api_report(st_module: Any) -> dict[str, Any]:
    sig = inspect.signature(st_module.image)
    params = list(sig.parameters)
    return {
        "streamlit_version": getattr(st_module, "__version__", "unknown"),
        "image_parameters": params,
        "supports_use_column_width": "use_column_width" in sig.parameters,
        "supports_use_container_width": "use_container_width" in sig.parameters,
        "preferred_full_width_kwarg": (
            "use_column_width"
            if "use_column_width" in sig.parameters
            else ("use_container_width" if "use_container_width" in sig.parameters else None)
        ),
        "root_cause_mehil_r1": (
            "ImageMixin.image() got unexpected keyword argument use_container_width "
            "because Streamlit 1.37.1 only declares use_column_width"
        ),
    }
