# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# License for AMD contributions = MIT. See LICENSE for more information

"""PyTorch opaque-object registration across the torch 2.14 rename.

torch 2.14 renamed ``register_opaque_type`` to ``register_custom_class``,
``is_opaque_value_type`` to ``is_opaque_constant_type`` and ``typ="value"`` to
``typ="constant"``, with unchanged semantics. TE uses the new names; on an older
torch (ROCm wheels ship 2.12) they are mapped onto the old ones, so the
torch.compile custom-op path stays available instead of being switched off.
"""

from torch._library import opaque_object as _opaque_object

if hasattr(_opaque_object, "register_custom_class"):
    register_custom_class = _opaque_object.register_custom_class
    is_opaque_constant_type = _opaque_object.is_opaque_constant_type
else:

    def register_custom_class(cls, *, typ: str, **kwargs) -> None:
        """``register_custom_class`` spelled with the pre-2.14 API."""
        _opaque_object.register_opaque_type(
            cls, typ="value" if typ == "constant" else typ, **kwargs
        )

    is_opaque_constant_type = _opaque_object.is_opaque_value_type
