# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for DslContextManager definition.

In pipeline DSL, there are advanced semantics using "with" block. Context
managers (DslContextManager; e.g. Cond) are used to define a scope of a context.

    with Cond(predicate):  # Cond is a DslContextManager
      node = Node(...)     # The node is associated with ConditionalContext.

Nodes defined within the context manager are associated with the DslContext
(e.g. ConditionalContext). If with blocks are nested, multiple DslContexts are
associated (order matters). DslContext is not directly exposed to user, but
instead DslContextManager can define arbitrary handle that is captured in
with-as block.
"""

import abc
import types
from typing import TypeVar, Generic, Optional, Type

from tfx.dsl.auto_collect import dsl_context
from tfx.dsl.auto_collect import pipeline_registry

_Handle = TypeVar('_Handle')


def _generate_unique_id(
    context: dsl_context.DslContext,
    registry: pipeline_registry.PipelineRegistry) -> str:
  context_type = str(context.__class__.__name__)
  unique_number = len(registry.all_contexts) + 1
  return f'{context_type}:{unique_number}'


class DslContextManager(Generic[_Handle], abc.ABC):
  """Base class for all context managers for pipeline DSL."""

  @abc.abstractmethod
  def create_context(self) -> dsl_context.DslContext:
    """Creates an underlying DslContext object.

    All nodes defined within this DslContextManager would be associated with
    the created DslContext.

    Since DslContextManager can __enter__ multiple times and each represents
    a different context, the return value should be newly created (not reused).

    Returns:
      Newly created DslContext object.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def enter(self, context: dsl_context.DslContext) -> _Handle:  # pylint: disable=unused-argument
    """Subclass hook method for __enter__.

    Returned value is captured at "with..as" clause. It can be any helper object
    to augment DSL syntax.

    Args:
      context: Newly created DslContext that DslContextManager has created for
          the __enter__().
    """

  def __enter__(self) -> _Handle:
    registry = pipeline_registry.get()
    context = self.create_context()
    context.id = _generate_unique_id(context, registry)
    context.parent = registry.peek_context()
    context.validate()
    registry.push_context(context)
    return self.enter(context)

  def __exit__(self, exc_type: Optional[Type[BaseException]],
               exc_val: Optional[BaseException],
               exc_tb: Optional[types.TracebackType]):
    pipeline_registry.get().pop_context()
