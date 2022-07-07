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
from contextlib import contextmanager
import inspect
from typing import Callable


def deferred(f: Callable[[], None]):
    @contextmanager
    def context():
        try:
            yield
        finally:
            f()

    return context()


def extra_vars(func: Callable, module_prefix: str):
    closure_vars = inspect.getclosurevars(func)
    vars = {}
    for k, v in closure_vars.globals.items():
        if inspect.ismodule(v) and v.__name__.startswith(module_prefix):
            vars[k] = v
        elif (inspect.isfunction(v) or inspect.isclass(v)) and v.__module__.startswith(
            module_prefix
        ):
            vars[k] = v
        else:
            raise ValueError(f"Method {k} is not from {module_prefix}")
    return vars
