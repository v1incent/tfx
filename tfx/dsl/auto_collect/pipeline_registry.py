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
"""Module for PipelineRegistry."""

import collections
import threading
from typing import Any, List, Optional

from tfx.dsl.auto_collect import dsl_context

# Use Any to avoid cyclic import.
_BaseNode = Any


class PipelineRegistry:
  """Registry for DslContexts and associated BaseNodes of a pipeline DSL.

  PipelineRegistry manages the active DslContexts, their orders, BaseNodes,
  and association between DslContext and BaseNodes during the pipeline DSL
  definition. DslContext and BaseNode always belong to exactly one
  PipelineRegistry (1:n relationship).
  """

  def __init__(self):
    super().__init__()
    # Frame of currently active DSL context. Ordered by parent -> child.
    self._active_contexts: List[dsl_context.DslContext] = []
    # All DSL contexts that have ever been defined so far.
    self._all_contexts: List[dsl_context.DslContext] = []
    self._all_nodes: List[_BaseNode] = []
    # Mapping from Context ID to a list of nodes that belong to each context.
    # Each list of node is sorted chronologically.
    self._nodes_by_context_ids = collections.defaultdict(list)
    self._finalized = False

  def __copy__(self):
    result = PipelineRegistry()
    result._active_contexts = list(self._active_contexts)
    result._all_contexts = list(self._all_contexts)
    result._all_nodes = list(self._all_nodes)
    result._nodes_by_context_ids.update({
        k: list(v) for k, v in self._nodes_by_context_ids.items()
    })
    result._finalized = False
    return result

  @property
  def all_contexts(self) -> List[dsl_context.DslContext]:
    """All contexts defined during the lifespan of the registry."""
    return list(self._all_contexts)

  @property
  def active_contexts(self) -> List[dsl_context.DslContext]:
    """All active context frame in parent -> child order."""
    return list(self._active_contexts)

  @property
  def all_nodes(self):
    return self._all_nodes

  def finalize(self):
    """Finalize and make the instance immutable."""
    self._finalized = True

  def _check_mutable(self):
    if self._finalized:
      raise AssertionError('Cannot mutate PipelineRegistry after finalized.')

  def push_context(self, context: dsl_context.DslContext):
    """Pushes the context to the top of active context frames."""
    self._check_mutable()
    self._active_contexts.append(context)
    self._all_contexts.append(context)

  def pop_context(self) -> dsl_context.DslContext:
    """Removes the top context from the active context frame."""
    self._check_mutable()
    assert self._active_contexts, (
        'Internal assertion error; no active contexts to remove.')
    return self._active_contexts.pop()

  def peek_context(self) -> Optional[dsl_context.DslContext]:
    """Returns the top context of the active context frame."""
    return self._active_contexts[-1] if self._active_contexts else None

  def put_node(self, node: _BaseNode) -> None:
    """Associates the node to all active contexts."""
    self._check_mutable()
    self._all_nodes.append(node)
    for context in self._active_contexts:
      context.will_add_node(node)
    for context in self._active_contexts:
      self._nodes_by_context_ids[context.id].append(node)

  def replace_node(self, node_from: _BaseNode, node_to: _BaseNode) -> None:
    """Replaces one node instance to another in a registry."""
    self._check_mutable()
    if node_from not in self._all_nodes:
      raise AssertionError(f'{node_from} does not exist in pipeline registry.')
    self._all_nodes[self._all_nodes.index(node_from)] = node_to
    for context in self._all_contexts:
      nodes = self._nodes_by_context_ids[context.id]
      if node_from in nodes:
        nodes[nodes.index(node_from)] = node_to
        context.replace_node(node_from, node_to)

  def get_nodes(self, context: dsl_context.DslContext) -> List[_BaseNode]:
    """Gets all BaseNodes that belongs to the context."""
    return list(self._nodes_by_context_ids[context.id])

  def get_contexts(self, node: _BaseNode) -> List[dsl_context.DslContext]:
    """Gets all dsl_context.DslContexts that the node belongs to."""
    # This is O(N^2), but not performance critical.
    result = []
    for context in self._all_contexts:
      if node in self._nodes_by_context_ids[context.id]:
        result.append(context)
    return result


_registry_holder = threading.local()
_registry_holder.current = None


def _ensure_registry():
  if _registry_holder.current is None:
    _registry_holder.current = PipelineRegistry()


def get() -> PipelineRegistry:
  """Gets the current active registry that observes DSL definitions."""
  _ensure_registry()
  return _registry_holder.current


def pop() -> PipelineRegistry:
  """Retires the current global registry from observing additional DSL definitions."""
  _ensure_registry()
  result = _registry_holder.current
  _registry_holder.current = None
  return result
