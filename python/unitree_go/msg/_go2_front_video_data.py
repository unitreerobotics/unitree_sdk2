# generated from rosidl_generator_py/resource/_idl.py.em
# with input from unitree_go:msg/Go2FrontVideoData.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'video720p'
# Member 'video360p'
# Member 'video180p'
import array  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Go2FrontVideoData(type):
    """Metaclass of message 'Go2FrontVideoData'."""

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
                'unitree_go.msg.Go2FrontVideoData')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__go2_front_video_data
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__go2_front_video_data
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__go2_front_video_data
            cls._TYPE_SUPPORT = module.type_support_msg__msg__go2_front_video_data
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__go2_front_video_data

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Go2FrontVideoData(metaclass=Metaclass_Go2FrontVideoData):
    """Message class 'Go2FrontVideoData'."""

    __slots__ = [
        '_time_frame',
        '_video720p',
        '_video360p',
        '_video180p',
    ]

    _fields_and_field_types = {
        'time_frame': 'uint64',
        'video720p': 'sequence<uint8>',
        'video360p': 'sequence<uint8>',
        'video180p': 'sequence<uint8>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint64'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('uint8')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('uint8')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('uint8')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.time_frame = kwargs.get('time_frame', int())
        self.video720p = array.array('B', kwargs.get('video720p', []))
        self.video360p = array.array('B', kwargs.get('video360p', []))
        self.video180p = array.array('B', kwargs.get('video180p', []))

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
        if self.time_frame != other.time_frame:
            return False
        if self.video720p != other.video720p:
            return False
        if self.video360p != other.video360p:
            return False
        if self.video180p != other.video180p:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def time_frame(self):
        """Message field 'time_frame'."""
        return self._time_frame

    @time_frame.setter
    def time_frame(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'time_frame' field must be of type 'int'"
            assert value >= 0 and value < 18446744073709551616, \
                "The 'time_frame' field must be an unsigned integer in [0, 18446744073709551615]"
        self._time_frame = value

    @property
    def video720p(self):
        """Message field 'video720p'."""
        return self._video720p

    @video720p.setter
    def video720p(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'B', \
                "The 'video720p' array.array() must have the type code of 'B'"
            self._video720p = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'video720p' field must be a set or sequence and each value of type 'int' and each unsigned integer in [0, 255]"
        self._video720p = array.array('B', value)

    @property
    def video360p(self):
        """Message field 'video360p'."""
        return self._video360p

    @video360p.setter
    def video360p(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'B', \
                "The 'video360p' array.array() must have the type code of 'B'"
            self._video360p = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'video360p' field must be a set or sequence and each value of type 'int' and each unsigned integer in [0, 255]"
        self._video360p = array.array('B', value)

    @property
    def video180p(self):
        """Message field 'video180p'."""
        return self._video180p

    @video180p.setter
    def video180p(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'B', \
                "The 'video180p' array.array() must have the type code of 'B'"
            self._video180p = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'video180p' field must be a set or sequence and each value of type 'int' and each unsigned integer in [0, 255]"
        self._video180p = array.array('B', value)
