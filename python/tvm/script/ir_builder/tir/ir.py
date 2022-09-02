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
# pylint: disable=missing-docstring
"""IRBuilder for TIR"""
import functools
import inspect
from numbers import Integral
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
from tvm.ir import Range, Type
from tvm.runtime import convert, ndarray
from tvm.tir import Broadcast as broadcast
from tvm.tir import (
    Buffer,
    BufferLoad,
    BufferRegion,
    Cast,
    CommReducer,
    IntImm,
    IterVar,
    Let,
    PrimExpr,
)
from tvm.tir import Ramp as ramp
from tvm.tir import Select, Shuffle, StringImm, Var
from tvm.tir.generic import cast
from tvm.tir import op as _tir_op
from tvm.tir import type_annotation

from . import _ffi_api, frame


def buffer_decl(
    shape: Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral],
    dtype: str = "float32",
    data: Var = None,
    strides: List[PrimExpr] = None,
    elem_offset: PrimExpr = None,
    scope: str = "",
    align: int = 0,
    offset_factor: int = 0,
    buffer_type: str = "",
    axis_separators: List[int] = None,
) -> Buffer:
    """The buffer declaration function.

    Parameters
    ----------
    shape : Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral]
        The type of the buffer prior to flattening.

    dtype : str
        The data type in the content of the buffer.

    data : Var
        The pointer to the head of the data.

    strides : List[PrimExpr]
        The strides of each dimension.

    elem_offset : PrimExpr
        The offset in terms of number of dtype elements (including lanes).

    scope : str
        The optional storage scope of buffer data pointer.

    align : int
        The alignment requirement of data pointer in bytes.

    offset_factor : int
        The factor of elem_offset field.

    buffer_type : str
        The buffer type.

    axis_separators : List[int]
        The separators between input axes when generating flattened output axes.

    Returns
    -------
    res : Buffer
        The declared buffer.
    """
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    return _ffi_api.BufferDecl(  # pylint: disable=no-member # type: ignore
        shape,
        dtype,
        "",
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def ptr(dtype, storage_scope="global"):
    """The pointer declaration function.

    Parameters
    ----------
    dtype : str
        The data type of the pointer.

    storage_scope : str
        The storage scope of the pointer.

    Returns
    -------
    res : Var
        The pointer.
    """
    return _ffi_api.Ptr(dtype, storage_scope)  # pylint: disable=no-member # type: ignore


def block(name: str = "", no_realize: bool = False) -> frame.BlockFrame:
    """The block declaration statement.

    Parameters
    ----------
    name : str
        The name of the block.

    no_realize : bool
        The flag whether to construct BlockRealize or Block.

    Returns
    -------
    res : frame.BlockFrame
        The BlockFrame.
    """
    return _ffi_api.Block(name, no_realize)  # pylint: disable=no-member # type: ignore


def init() -> frame.BlockInitFrame:
    """The block initialization statement.

    Returns
    -------
    res : frame.BlockInitFrame
        The BlockInitFrame.
    """
    return _ffi_api.Init()  # pylint: disable=no-member # type: ignore


def where(predicate: Union[PrimExpr, int]) -> None:
    """The block predicate statement.

    Parameters
    ----------
    predicate : Union[PrimExpr, Literal[0, 1]]
        The predicate condition.
    """
    if isinstance(predicate, bool):
        predicate = IntImm("bool", predicate)
    if isinstance(predicate, int):
        if predicate in [0, 1]:
            predicate = IntImm("bool", predicate)
        else:
            raise ValueError(f"Invalid value for predicate: {predicate}")
    _ffi_api.Where(predicate)  # pylint: disable=no-member # type: ignore


def reads(*buffer_slices: List[Union[BufferRegion, BufferLoad]]) -> None:
    """The block buffer region reading statement.

    Parameters
    ----------
    buffer_slices : List[Union[BufferRegion, BufferLoad]]
        The array of buffer regions to read.
    """
    if len(buffer_slices) == 1:
        if isinstance(buffer_slices[0], tuple):
            buffer_slices = list(buffer_slices[0])
        elif isinstance(buffer_slices[0], list):
            buffer_slices = buffer_slices[0]  # type: ignore
        else:
            buffer_slices = [buffer_slices[0]]  # type: ignore
    else:
        buffer_slices = list(buffer_slices)  # type: ignore
    _ffi_api.Reads(buffer_slices)  # pylint: disable=no-member # type: ignore


def writes(*buffer_slices: List[Union[BufferRegion, BufferLoad]]) -> None:
    """The block buffer region writing statement.

    Parameters
    ----------
    buffer_slices : List[Union[BufferRegion, BufferLoad]]
        The array of buffer regions to write.
    """
    if len(buffer_slices) == 1:
        if isinstance(buffer_slices[0], tuple):
            buffer_slices = list(buffer_slices[0])
        elif isinstance(buffer_slices[0], list):
            buffer_slices = buffer_slices[0]  # type: ignore
        else:
            buffer_slices = [buffer_slices[0]]
    else:
        buffer_slices = list(buffer_slices)  # type: ignore
    _ffi_api.Writes(buffer_slices)  # pylint: disable=no-member # type: ignore


def block_attr(attrs: Dict[str, Any]) -> None:
    """The block annotation statement.

    Parameters
    ----------
    attrs : Dict[str, Any]
        The annotation of the block.
    """
    return _ffi_api.BlockAttrs(attrs)  # pylint: disable=no-member # type: ignore


def alloc_buffer(
    shape: Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral],
    dtype: str = "float32",
    data: Var = None,
    strides: List[PrimExpr] = None,
    elem_offset: PrimExpr = None,
    scope: str = "",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
    axis_separators: List[int] = None,
) -> Buffer:
    """The buffer alllocation function.

    Parameters
    ----------
    shape : Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral]
        The type of the buffer prior to flattening.

    dtype : str
        The data type in the content of the buffer.

    data : Var
        The pointer to the head of the data.

    strides : List[PrimExpr]
        The strides of each dimension.

    elem_offset : PrimExpr
        The offset in terms of number of dtype elements (including lanes).

    scope : str
        The optional storage scope of buffer data pointer.

    align : int
        The alignment requirement of data pointer in bytes.

    offset_factor : int
        The factor of elem_offset field.

    buffer_type : str
        The buffer type.

    axis_separators : List[int]
        The separators between input axes when generating flattened output axes.

    Returns
    -------
    res : Buffer
        The allocated buffer.
    """
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is None:
        strides = []
    return _ffi_api.AllocBuffer(  # pylint: disable=no-member # type: ignore
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def _as_range(dom: Union[Range, List[PrimExpr]]) -> Range:
    """The range constructor.

    Parameters
    ----------
    dom : Union[Range, List[PrimExpr]]
        The domain.

    Returns
    -------
    res : Range
        The Range.
    """
    if isinstance(dom, Range):
        return dom
    if isinstance(dom, (list, tuple)):
        return Range(dom[0], dom[1])
    return Range(0, dom)


class axis:  # pylint: disable=invalid-name
    @staticmethod
    def spatial(
        dom: Union[Range, List[PrimExpr], Tuple[PrimExpr]], binding: PrimExpr, dtype: str = "int32"
    ) -> Var:
        """The spatial block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[PrimExpr], Tuple[PrimExpr]]
            The domain of the iteration variable.

        binding : PrimExpr
            The binding value of the iteration variable.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisSpatial(  # pylint: disable=no-member # type: ignore
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def reduce(
        dom: Union[Range, List[PrimExpr], Tuple[PrimExpr]], binding: PrimExpr, dtype: str = "int32"
    ) -> Var:
        """The reduced block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[PrimExpr], Tuple[PrimExpr]]
            The domain of the iteration variable.

        binding : PrimExpr
            The binding value of the iteration variable.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisReduce(  # pylint: disable=no-member # type: ignore
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def scan(
        dom: Union[Range, List[PrimExpr], Tuple[PrimExpr]], binding: PrimExpr, dtype: str = "int32"
    ) -> Var:
        """The scanning block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[PrimExpr], Tuple[PrimExpr]]
            The domain of the iteration variable.

        binding : PrimExpr
            The binding value of the iteration variable.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisScan(  # pylint: disable=no-member # type: ignore
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def opaque(
        dom: Union[Range, List[PrimExpr], Tuple[PrimExpr]], binding: PrimExpr, dtype: str = "int32"
    ) -> Var:
        """The opaque block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[PrimExpr], Tuple[PrimExpr]]
            The domain of the iteration variable.

        binding : PrimExpr
            The binding value of the iteration variable.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisOpaque(  # pylint: disable=no-member # type: ignore
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def remap(kinds: str, bindings: List[PrimExpr], dtype: str = "int32") -> Union[List[Var], Var]:
        """The block axis remapping function.

        Parameters
        ----------
        kinds : str
            The types of the iteration variables.

        bindings : List[PrimExpr]
            The binding values of the iteration variables.

        dtype : str
            The data types of the iteration variables.

        Returns
        -------
        res : Var
            The iteration variables.
        """
        iter_vars = _ffi_api.AxisRemap(  # pylint: disable=no-member # type: ignore
            kinds, bindings, dtype
        )
        return iter_vars[0] if len(iter_vars) == 1 else iter_vars

    S = spatial  # pylint: disable=invalid-name
    R = reduce  # pylint: disable=invalid-name


def serial(
    start: PrimExpr, stop: PrimExpr = None, *, annotations: Dict[str, Any] = None
) -> frame.ForFrame:
    """The serial For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Serial(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def parallel(
    start: PrimExpr, stop: PrimExpr = None, *, annotations: Dict[str, Any] = None
) -> frame.ForFrame:
    """The parallel For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Parallel(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def vectorized(
    start: PrimExpr, stop: PrimExpr = None, *, annotations: Dict[str, Any] = None
) -> frame.ForFrame:
    """The vectorized For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Vectorized(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def unroll(
    start: PrimExpr, stop: PrimExpr = None, *, annotations: Dict[str, Any] = None
) -> frame.ForFrame:
    """The unrolled For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if stop is None:
        stop = start
        start = 0
    return _ffi_api.Unroll(start, stop, annotations)  # pylint: disable=no-member # type: ignore


def thread_binding(
    start: PrimExpr,
    stop: PrimExpr = None,
    thread: str = None,
    *,
    annotations: Dict[str, Any] = None,
) -> frame.ForFrame:
    """The thread-binding For statement.

    Parameters
    ----------
    start : PrimExpr
        The minimum value of iteration.

    stop : PrimExpr
        The maximum value of iteration.

    thread : str
        The thread for loop variable to bind.

    annotations : Dict[str, Any]
        The optional annotations of the For statement.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    if thread is None:
        if not isinstance(stop, str):
            raise ValueError("Thread cannot be None for thread_binding")
        thread = stop
        stop = start
        start = 0
    elif stop is None:
        stop = start
        start = 0
    return _ffi_api.ThreadBinding(  # pylint: disable=no-member # type: ignore
        start, stop, thread, annotations
    )


def grid(*extents: PrimExpr) -> frame.ForFrame:
    """The grid For statement.

    Parameters
    ----------
    extents : PrimExpr
        The extents of the iteration.

    Returns
    -------
    res : frame.ForFrame
        The ForFrame.
    """
    return _ffi_api.Grid(extents)  # pylint: disable=no-member # type: ignore


def prim_func() -> frame.PrimFuncFrame:
    """The primitive function statement.

    Returns
    -------
    res : frame.PrimFuncFrame
        The PrimFuncFrame.
    """
    return _ffi_api.PrimFunc()  # pylint: disable=no-member # type: ignore


def arg(name: str, obj: Union[Var, Buffer]) -> Union[Var, Buffer]:
    """The PrimFunc arguments adding function.

    Parameters
    ----------
    name : str
        The name of the argument.

    var : Union[Var, Buffer]
        The argument of Var or Buffer.

    Returns
    -------
    res : Union[Var, Buffer]
        The argument.
    """
    return _ffi_api.Arg(name, obj)  # pylint: disable=no-member # type: ignore


def func_name(name: str) -> None:
    """The PrimFunc naming statement.

    Parameters
    ----------
    name : str
        The name of the PrimFunc.
    """
    _ffi_api.FuncName(name)  # pylint: disable=no-member # type: ignore


def func_attr(attrs: Dict[str, Any]) -> None:
    """The PrimFunc annotation statement.

    Parameters
    ----------
    attrs : Dict[str, Any]
        The annotations of the PrimFunc.
    """
    _ffi_api.FuncAttrs(attrs)  # pylint: disable=no-member # type: ignore


def func_ret(ret_type: Type) -> Type:
    """The PrimFunc return type statement.

    Parameters
    ----------
    ret_type : Type
        The return type of the PrimFunc.

    Returns
    -------
    res : Type
        The return type.
    """
    return _ffi_api.FuncRet(ret_type)  # pylint: disable=no-member # type: ignore


def match_buffer(
    param: Union[Var, BufferLoad, BufferRegion],
    shape: Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral],
    dtype: str = "float32",
    data: Var = None,
    strides: List[PrimExpr] = None,
    elem_offset: PrimExpr = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
    axis_separators: List[int] = None,
) -> Buffer:
    """The buffer match function.

    Note
    ----
    This function will perform different behavior, depending on the type of param.
    If the param is a var in function parameter, it will create a buffer from DLTensor.
    Else if the param is a subregion of other buffers, then create a subregion match inside a block.

    Example
    -------
    Match buffer from function parameter
    .. code-block:: python
        A = T.match_buffer(a, (128, 128), dtype="float32")

    Match buffer from Buffer subregion
    .. code-block:: python
        A = T.match_buffer(B[0:128, i * 128 : i * 128 + 128], (128, 128), dtype="float32")

    Parameters
    ----------
    param : Union[Var, BufferLoad, BufferRegion]
        The parameter of the PrimFunc to match.

    shape : Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral]
        The type of the buffer prior to flattening.

    dtype : str
        The data type in the content of the buffer.

    data : Var
        The pointer to the head of the data.

    strides : List[PrimExpr]
        The strides of each dimension.

    elem_offset : PrimExpr
        The offset in terms of number of dtype elements (including lanes).

    scope : str
        The optional storage scope of buffer data pointer.

    align : int
        The alignment requirement of data pointer in bytes.

    offset_factor : int
        The factor of elem_offset field.

    buffer_type : str
        The buffer type.

    axis_separators : List[int]
        The separators between input axes when generating flattened output axes.

    Returns
    -------
    res : Buffer
        The matched buffer.
    """
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is None:
        strides = []
    return _ffi_api.MatchBuffer(  # pylint: disable=no-member # type: ignore
        param,
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def preflattened_buffer(
    postflattened: Buffer,
    shape: Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral],
    dtype: str = "float32",
    data: Var = None,
    strides: List[PrimExpr] = None,
    elem_offset: PrimExpr = None,
    scope: str = "global",
    align: int = -1,
    offset_factor: int = 0,
    buffer_type: str = "default",
    axis_separators: List[int] = None,
) -> None:
    """The pre-flattened buffer statement.

    Parameters
    ----------
    postflattened : Buffer
        The original buffer to be flattened.

    shape : Union[List[PrimExpr], Tuple[PrimExpr], PrimExpr, Integral]
        The type of the buffer prior to flattening.

    dtype : str
        The data type in the content of the buffer.

    data : Var
        The pointer to the head of the data.

    strides : List[PrimExpr]
        The strides of each dimension.

    elem_offset : PrimExpr
        The offset in terms of number of dtype elements (including lanes).

    scope : str
        The optional storage scope of buffer data pointer.

    align : int
        The alignment requirement of data pointer in bytes.

    offset_factor : int
        The factor of elem_offset field.

    buffer_type : str
        The buffer type.

    axis_separators : List[int]
        The separators between input axes when generating flattened output axes.
    """
    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    if strides is None:
        strides = []
    _ffi_api.PreflattenedBuffer(  # pylint: disable=no-member # type: ignore
        postflattened,
        shape,
        dtype,
        data,
        strides,
        elem_offset,
        scope,
        align,
        offset_factor,
        buffer_type,
        axis_separators,
    )


def evaluate(value: PrimExpr) -> None:
    """Evaluate the input expression.

    Parameters
    ----------
    value: PrimExpr
        The input expression to evaluate.
    """
    if isinstance(value, str):
        value = StringImm(value)
    return _ffi_api.Evaluate(value)  # pylint: disable=no-member # type: ignore


def int8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int8(expr)  # pylint: disable=no-member # type: ignore


def int16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int16(expr)  # pylint: disable=no-member # type: ignore


def int32(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int32(expr)  # pylint: disable=no-member # type: ignore


def int64(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int64(expr)  # pylint: disable=no-member # type: ignore


def uint8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.UInt8(expr)  # pylint: disable=no-member # type: ignore


def uint16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.UInt16(expr)  # pylint: disable=no-member # type: ignore


def uint32(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.UInt32(expr)  # pylint: disable=no-member # type: ignore


def uint64(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.UInt64(expr)  # pylint: disable=no-member # type: ignore


def float8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    if not isinstance(expr, PrimExpr):
        expr = convert(expr)
    return _ffi_api.Float8(expr)  # pylint: disable=no-member # type: ignore


def float16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    if not isinstance(expr, PrimExpr):
        expr = convert(expr)
    return _ffi_api.Float16(expr)  # pylint: disable=no-member # type: ignore


def float32(expr: Optional[PrimExpr] = None) -> PrimExpr:
    if not isinstance(expr, PrimExpr):
        expr = convert(expr)
    return _ffi_api.Float32(expr)  # pylint: disable=no-member # type: ignore


def float64(expr: Optional[PrimExpr] = None) -> PrimExpr:
    if not isinstance(expr, PrimExpr):
        expr = convert(expr)
    return _ffi_api.Float64(expr)  # pylint: disable=no-member # type: ignore


def int32x4(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int32x4(expr)  # pylint: disable=no-member # type: ignore


def int32x8(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int32x8(expr)  # pylint: disable=no-member # type: ignore


def int32x16(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Int32x16(expr)  # pylint: disable=no-member # type: ignore


def boolean(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Boolean(expr)  # pylint: disable=no-member # type: ignore


def handle(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Handle(expr)  # pylint: disable=no-member # type: ignore


def void(expr: Optional[PrimExpr] = None) -> PrimExpr:
    return _ffi_api.Void(expr)  # pylint: disable=no-member # type: ignore


def var(dtype, name="") -> Var:
    return Var(name, dtype)  # pylint: disable=no-member # type: ignore


# pylint: enable=invalid-name


__all__ = [
    "Cast",
    "Let",
    "Select",
    "Shuffle",
    "alloc_buffer",
    "arg",
    "axis",
    "block",
    "block_attr",
    "boolean",
    "broadcast",
    "buffer_decl",
    "cast",
    "evaluate",
    "float16",
    "float32",
    "float64",
    "float8",
    "func_attr",
    "func_name",
    "func_ret",
    "grid",
    "handle",
    "init",
    "int16",
    "int32",
    "int32x16",
    "int32x4",
    "int32x8",
    "int64",
    "int8",
    "match_buffer",
    "parallel",
    "preflattened_buffer",
    "prim_func",
    "ptr",
    "ramp",
    "reads",
    "serial",
    "thread_binding",
    "type_annotation",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "unroll",
    "var",
    "vectorized",
    "void",
    "where",
    "writes",
]
