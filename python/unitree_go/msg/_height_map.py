# generated from rosidl_generator_py/resource/_idl.py.em
# with input from unitree_go:msg/HeightMap.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'data'
import array  # noqa: E402, I100

# Member 'origin'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_HeightMap(type):
    """Metaclass of message 'HeightMap'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('unitree_go')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'unitree_go.msg.HeightMap')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__height_map
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__height_map
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__height_map
            cls._TYPE_SUPPORT = module.type_support_msg__msg__height_map
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__height_map

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class HeightMap(metaclass=Metaclass_HeightMap):
    """Message class 'HeightMap'."""

    __slots__ = [
        '_stamp',
        '_frame_id',
        '_resolution',
        '_width',
        '_height',
        '_origin',
        '_data',
    ]

    _fields_and_field_types = {
        'stamp': 'double',
        'frame_id': 'string',
        'resolution': 'float',
        'width': 'uint32',
        'height': 'uint32',
        'origin': 'float[2]',
        'data': 'sequence<float>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 2),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.stamp = kwargs.get('stamp', float())
        self.frame_id = kwargs.get('frame_id', str())
        self.resolution = kwargs.get('resolution', float())
        self.width = kwargs.get('width', int())
        self.height = kwargs.get('height', int())
        if 'origin' not in kwargs:
            self.origin = numpy.zeros(2, dtype=numpy.float32)
        else:
            self.origin = numpy.array(kwargs.get('origin'), dtype=numpy.float32)
            assert self.origin.shape == (2, )
        self.data = array.array('f', kwargs.get('data', []))

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.stamp != other.stamp:
            return False
        if self.frame_id != other.frame_id:
            return False
        if self.resolution != other.resolution:
            return False
        if self.width != other.width:
            return False
        if self.height != other.height:
            return False
        if all(self.origin != other.origin):
            return False
        if self.data != other.data:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def stamp(self):
        """Message field 'stamp'."""
        return self._stamp

    @stamp.setter
    def stamp(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'stamp' field must be of type 'float'"
        self._stamp = value

    @property
    def frame_id(self):
        """Message field 'frame_id'."""
        return self._frame_id

    @frame_id.setter
    def frame_id(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'frame_id' field must be of type 'str'"
        self._frame_id = value

    @property
    def resolution(self):
        """Message field 'resolution'."""
        return self._resolution

    @resolution.setter
    def resolution(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'resolution' field must be of type 'float'"
        self._resolution = value

    @property
    def width(self):
        """Message field 'width'."""
        return self._width

    @width.setter
    def width(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'width' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'width' field must be an unsigned integer in [0, 4294967295]"
        self._width = value

    @property
    def height(self):
        """Message field 'height'."""
        return self._height

    @height.setter
    def height(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'height' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'height' field must be an unsigned integer in [0, 4294967295]"
        self._height = value

    @property
    def origin(self):
        """Message field 'origin'."""
        return self._origin

    @origin.setter
    def origin(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'origin' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 2, \
                "The 'origin' numpy.ndarray() must have a size of 2"
            self._origin = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 2 and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'origin' field must be a set or sequence with length 2 and each value of type 'float'"
        self._origin = numpy.array(value, dtype=numpy.float32)

    @property
    def data(self):
        """Message field 'data'."""
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'data' array.array() must have the type code of 'f'"
            self._data = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'data' field must be a set or sequence and each value of type 'float'"
        self._data = array.array('f', value)
