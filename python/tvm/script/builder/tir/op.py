# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""TVM Script TIR Op"""

from tvm.tir.expr import Broadcast, Ramp, Select, Shuffle
from tvm.tir.generic import cast
from tvm.tir.op import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    call_extern,
    call_packed,
    ceil,
    clz,
    comm_reducer,
    copysign,
    cos,
    cosh,
    erf,
    exp,
    exp2,
    exp10,
    floor,
    floordiv,
    floormod,
    fmod,
    hypot,
    if_then_else,
    infinity,
    isfinite,
    isinf,
    isnan,
    ldexp,
    log,
    log1p,
    log2,
    log10,
    max_value,
    min_value,
    nearbyint,
    nextafter,
    popcount,
    power,
    reinterpret,
    round,
    rsqrt,
    sigmoid,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trunc,
    truncdiv,
    truncmod,
)

from . import _ffi_api


def int8(expr=None):
    return _ffi_api.Int8(expr)


def int16(expr=None):
    return _ffi_api.Int16(expr)


def int32(expr=None):
    return _ffi_api.Int32(expr)


def int64(expr=None):
    return _ffi_api.Int64(expr)


def uint8(expr=None):
    return _ffi_api.UInt8(expr)


def uint16(expr=None):
    return _ffi_api.UInt16(expr)


def uint32(expr=None):
    return _ffi_api.UInt32(expr)


def uint64(expr=None):
    return _ffi_api.UInt64(expr)


def float8(expr=None):
    return _ffi_api.Float8(expr)


def float16(expr=None):
    return _ffi_api.Float16(expr)


def float32(expr=None):
    return _ffi_api.Float32(expr)


def float64(expr=None):
    return _ffi_api.Float64(expr)


def boolean(expr=None):
    return _ffi_api.Boolean(expr)


def handle():
    return _ffi_api.Handle()


def min(a, b):
    """Compute the minimum value of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api.min(a, b)  # type: ignore


def max(a, b):
    """Compute the maximum value of two expressions.

    Parameters
    ----------
    a : PrimExpr
        The left hand operand

    b : PrimExpr
        The right hand operand

    Returns
    -------
    res : PrimExpr
        The result expression.

    Note
    ----
    This is the default integer division behavior in C.
    """
    return _ffi_api.max(a, b)  # type: ignore
