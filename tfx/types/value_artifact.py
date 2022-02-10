# Copyright 2019 Google LLC. All Rights Reserved.
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
"""TFX artifact type definition."""

import abc
from typing import Any, Type, Optional

from tfx.dsl.io import fileio
from tfx.types.artifact import Artifact
from tfx.types.system_artifacts import SystemArtifact
from tfx.utils import doc_controls

_IS_NULL_KEY = '__is_null__'


class ValueArtifact(Artifact):
  """Artifacts of small scalar-values that can be easily loaded into memory."""

  def __init__(self, *args, **kwargs):
    """Initializes ValueArtifact."""
    self._has_value = False
    self._modified = False
    self._value = None
    super().__init__(*args, **kwargs)

  @doc_controls.do_not_doc_inheritable
  def read(self):
    if not self._has_value:
      file_path = self.uri
      # Assert there is a file exists.
      if not fileio.exists(file_path):
        raise RuntimeError(
            'Given path does not exist or is not a valid file: %s' % file_path)

      self._has_value = True
      if not self.get_int_custom_property(_IS_NULL_KEY):
        serialized_value = fileio.open(file_path, 'rb').read()
        self._value = self.decode(serialized_value)
    return self._value

  @doc_controls.do_not_doc_inheritable
  def write(self, value):
    if value is None:
      self.set_int_custom_property(_IS_NULL_KEY, 1)
      serialized_value = b''
    else:
      self.set_int_custom_property(_IS_NULL_KEY, 0)
      serialized_value = self.encode(value)
    with fileio.open(self.uri, 'wb') as f:
      f.write(serialized_value)

  @property
  def value(self):
    """Value stored in the artifact."""
    if not self._has_value:
      raise ValueError('The artifact value has not yet been read from storage.')
    return self._value

  @value.setter
  def value(self, value):
    self._modified = True
    self._value = value
    self.write(value)

  # Note: behavior of decode() method should not be changed to provide
  # backward/forward compatibility.
  @doc_controls.do_not_doc_inheritable
  @abc.abstractmethod
  def decode(self, serialized_value) -> bytes:
    """Method decoding the file content. Implemented by subclasses."""
    pass

  # Note: behavior of encode() method should not be changed to provide
  # backward/forward compatibility.
  @doc_controls.do_not_doc_inheritable
  @abc.abstractmethod
  def encode(self, value) -> Any:
    """Method encoding the file content. Implemented by subclasses."""
    pass

  @classmethod
  def annotate_as(cls, type_annotation: Optional[Type[SystemArtifact]] = None):
    """Annotate the value artifact type with a system artifact class.

    Example usage:

    from tfx.types.system_artifacts import Model
    ...
    tfx.Binary(
      name=component_name,
      mpm_or_target=...,
      flags=...,
      outputs={
          'experiment_id': standard_artifacts.String.annotate_as(Model)
      })

    Args:
      type_annotation: the system artifact class used to annotate the value
        artifact type. It is a subclass of SystemArtifact. The subclasses are
        defined in third_party/py/tfx/types/system_artifacts.py.

    Returns:
      A subclass of the method caller class (e.g., standard_artifacts.String,
      standard_artifacts.Float) with TYPE_ANNOTATION attribute set to be
      `type_annotation`; returns the class itself when`type_annotation` is None.
    """
    if not type_annotation:
      return cls
    if not issubclass(type_annotation, SystemArtifact):
      raise ValueError(
          'type_annotation %s is not a subclass of SystemArtifact.' %
          type_annotation)
    type_annotation_str = str(type_annotation.__name__)
    return type(
        str(cls.__name__) + '_' + type_annotation_str,
        (cls,),
        dict(
            TYPE_NAME=str(cls.TYPE_NAME) + '_' + type_annotation_str,
            TYPE_ANNOTATION=type_annotation,
        ),
    )
