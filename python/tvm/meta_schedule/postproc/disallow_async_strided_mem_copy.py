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
"""A postprocessor that checks if the IRModule has any strided memory copies"""

from tvm._ffi.registry import register_object
from .. import _ffi_api
from .postproc import Postproc


@register_object("meta_schedule.DisallowAsyncStridedMemCopy")
class DisallowAsyncStridedMemCopy(Postproc):
    """A postprocessor that disallows schedules that use async strided mem copies.

    Parameters
    ----------
    merge_async_commit_queue_scope : bool
       Whether or not to merge the async commit queue scope.
    """

    def __init__(self, merge_async_commit_queue_scope=True) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.PostprocDisallowAsyncStridedMemCopy,  # type: ignore # pylint: disable=no-member
            merge_async_commit_queue_scope,
        )
