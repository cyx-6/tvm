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
# pylint: disable=invalid-name, unused-variable, unused-argument
"""Transposed 3D convolution operators (sometimes called Deconvolution)."""
from tvm import te

from ..utils import simplify
from .dilate import dilate
from .pad import pad
from .utils import get_pad_tuple3d


def conv3d_transpose_ncdhw(Input, Filter, strides, padding, out_dtype, output_padding):
    """Transposed 3D convolution ncdhw forward operator.

    Parameters
    ----------
    Input : tvm.te.Tensor
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    Filter : tvm.te.Tensor
        5-D with shape [in_channel, num_filter, filter_depth, filter_height, filter_width]

    strides : int or a list/tuple of three ints
        The spatial stride along depth,height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype : str
        The output data type. This is used for mixed precision.

    output_padding : tuple of ints
        Used to get the right output shape for gradients

    Returns
    -------
    Output : tvm.te.Tensor
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    return declaration_conv3d_transpose_impl(
        Input, Filter, strides, padding, out_dtype, output_padding
    )


def conv3d_transpose_ncdhw_preprocess(data, kernel, strides, padding, out_dtype, output_padding):
    """Preprocess data and kernel to make the compute pattern
    of conv3d_transpose the same as conv3d"""
    batch, in_c, in_d, in_h, in_w = data.shape
    _, out_c, filter_d, filter_h, filter_w = kernel.shape
    stride_d, stride_h, stride_w = strides
    opad_d, opad_h, opad_w = output_padding
    assert opad_d < stride_d and opad_h < stride_h and opad_w < stride_w
    # dilate data
    data_dilate = dilate(data, [1, 1, stride_d, stride_h, stride_w], name="data_dilate")
    # pad data
    fpad_front, fpad_top, fpad_left, fpad_back, fpad_bottom, fpad_right = get_pad_tuple3d(
        padding, (filter_d, filter_h, filter_w)
    )
    bpad_front = filter_d - 1 - fpad_front
    bpad_back = filter_d - 1 - fpad_back + opad_d
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom + opad_h
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right + opad_w
    data_pad = pad(
        data_dilate,
        [0, 0, bpad_front, bpad_top, bpad_left],
        [0, 0, bpad_back, bpad_bottom, bpad_right],
        name="data_pad",
    )
    # transform kernel layout from IODHW to OIDHW, and rotate kernel by 180 degrees
    kernel_transform = te.compute(
        (out_c, in_c, filter_d, filter_h, filter_w),
        lambda o, i, d, h, w: kernel[i][o][filter_d - 1 - d][filter_h - 1 - h][filter_w - 1 - w],
        name="kernel_transform",
    )
    return data_pad, kernel_transform


def declaration_conv3d_transpose_impl(data, kernel, strides, padding, out_dtype, output_padding):
    """Implementation of conv3d transpose"""
    data_pad, kernel_transform = conv3d_transpose_ncdhw_preprocess(
        data, kernel, strides, padding, out_dtype, output_padding
    )
    batch, in_c, in_d, in_h, in_w = data_pad.shape
    out_c, _, filter_d, filter_h, filter_w = kernel_transform.shape
    stride_d, stride_h, stride_w = strides

    # convolution stage
    out_c = simplify(out_c)
    out_d = simplify(in_d - filter_d + 1)
    out_h = simplify(in_h - filter_h + 1)
    out_w = simplify(in_w - filter_w + 1)
    dc = te.reduce_axis((0, in_c), name="dc")
    dd = te.reduce_axis((0, filter_d), name="dd")
    dh = te.reduce_axis((0, filter_h), name="dh")
    dw = te.reduce_axis((0, filter_w), name="dw")

    Output = te.compute(
        (batch, out_c, out_d, out_h, out_w),
        lambda b, c, d, h, w: te.sum(
            data_pad[b, dc, d + dd, h + dh, w + dw].astype(out_dtype)
            * kernel_transform[c, dc, dd, dh, dw].astype(out_dtype),
            axis=[dc, dd, dh, dw],
        ),
        tag="conv3d_transpose_ncdhw",
    )

    return Output


def group_conv3d_transpose_ncdhw(data, kernel, strides, padding, out_dtype, output_padding, groups):
    """Transposed group 3D convolution ncdhw forward operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    kernel : tvm.te.Tensor
        5-D with shape [in_channel, num_filter, filter_depth, filter_height, filter_width]

    strides : int or a list/tuple of three ints
        The spatial stride along depth,height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    out_dtype : str
        The output data type. This is used for mixed precision.

    output_padding : tuple of ints
        Used to get the right output shape for gradients

    groups : int
        number of groups

    Returns
    -------
    Output : tvm.te.Tensor
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    if not isinstance(strides, (tuple, list)):
        strides = (strides, strides, strides)

    if groups == 1:
        return conv3d_transpose_ncdhw(data, kernel, strides, padding, out_dtype, output_padding)

    data_pad, kernel_transform = conv3d_transpose_ncdhw_preprocess(
        data, kernel, strides, padding, out_dtype, output_padding
    )
    batch, in_c, in_d, in_h, in_w = data_pad.shape
    out_c, _, filter_d, filter_h, filter_w = kernel_transform.shape
    assert in_c % groups == 0, f"input channels {in_c} must divide group size {groups}"

    # convolution stage
    out_c = simplify(out_c * groups)
    out_d = simplify(in_d - filter_d + 1)
    out_h = simplify(in_h - filter_h + 1)
    out_w = simplify(in_w - filter_w + 1)
    dc = te.reduce_axis((0, in_c // groups), name="dc")
    dd = te.reduce_axis((0, filter_d), name="dd")
    dh = te.reduce_axis((0, filter_h), name="dh")
    dw = te.reduce_axis((0, filter_w), name="dw")

    # data: batch, in_channels, out_d, out_h, out_w
    # weight: out_channels // G, in_channels, out_d, out_h, out_w
    return te.compute(
        (batch, out_c, out_d, out_h, out_w),
        lambda b, c, d, h, w: te.sum(
            data_pad[
                b, c // (out_c // groups) * (in_c // groups) + dc, d + dd, h + dh, w + dw
            ].astype(out_dtype)
            * kernel_transform[
                c % (out_c // groups),
                c // (out_c // groups) * (in_c // groups) + dc,
                dd,
                dh,
                dw,
            ].astype(out_dtype),
            axis=[dc, dd, dh, dw],
        ),
        tag="group_conv3d_transpose_ncdhw",
    )
