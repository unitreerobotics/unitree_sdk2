# generated from rosidl_generator_py/resource/_idl.py.em
# with input from unitree_go:msg/SportModeCmd.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'position'
# Member 'euler'
# Member 'velocity'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SportModeCmd(type):
    """Metaclass of message 'SportModeCmd'."""

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
                'unitree_go.msg.SportModeCmd')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__sport_mode_cmd
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__sport_mode_cmd
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__sport_mode_cmd
            cls._TYPE_SUPPORT = module.type_support_msg__msg__sport_mode_cmd
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__sport_mode_cmd

            from unitree_go.msg import BmsCmd
            if BmsCmd.__class__._TYPE_SUPPORT is None:
                BmsCmd.__class__.__import_type_support__()

            from unitree_go.msg import PathPoint
            if PathPoint.__class__._TYPE_SUPPORT is None:
                PathPoint.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SportModeCmd(metaclass=Metaclass_SportModeCmd):
    """Message class 'SportModeCmd'."""

    __slots__ = [
        '_mode',
        '_gait_type',
        '_speed_level',
        '_foot_raise_height',
        '_body_height',
        '_position',
        '_euler',
        '_velocity',
        '_yaw_speed',
        '_bms_cmd',
        '_path_point',
    ]

    _fields_and_field_types = {
        'mode': 'uint8',
        'gait_type': 'uint8',
        'speed_level': 'uint8',
        'foot_raise_height': 'float',
        'body_height': 'float',
        'position': 'float[2]',
        'euler': 'float[3]',
        'velocity': 'float[2]',
        'yaw_speed': 'float',
        'bms_cmd': 'unitree_go/BmsCmd',
        'path_point': 'unitree_go/PathPoint[30]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 3),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 2),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['unitree_go', 'msg'], 'BmsCmd'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.NamespacedType(['unitree_go', 'msg'], 'PathPoint'), 30),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.mode = kwargs.get('mode', int())
        self.gait_type = kwargs.get('gait_type', int())
        self.speed_level = kwargs.get('speed_level', int())
        self.foot_raise_height = kwargs.get('foot_raise_height', float())
        self.body_height = kwargs.get('body_height', float())
        if 'position' not in kwargs:
            self.position = numpy.zeros(2, dtype=numpy.float32)
        else:
            self.position = numpy.array(kwargs.get('position'), dtype=numpy.float32)
            assert self.position.shape == (2, )
        if 'euler' not in kwargs:
            self.euler = numpy.zeros(3, dtype=numpy.float32)
        else:
            self.euler = numpy.array(kwargs.get('euler'), dtype=numpy.float32)
            assert self.euler.shape == (3, )
        if 'velocity' not in kwargs:
            self.velocity = numpy.zeros(2, dtype=numpy.float32)
        else:
            self.velocity = numpy.array(kwargs.get('velocity'), dtype=numpy.float32)
            assert self.velocity.shape == (2, )
        self.yaw_speed = kwargs.get('yaw_speed', float())
        from unitree_go.msg import BmsCmd
        self.bms_cmd = kwargs.get('bms_cmd', BmsCmd())
        from unitree_go.msg import PathPoint
        self.path_point = kwargs.get(
            'path_point',
            [PathPoint() for x in range(30)]
        )

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
        if self.mode != other.mode:
            return False
        if self.gait_type != other.gait_type:
            return False
        if self.speed_level != other.speed_level:
            return False
        if self.foot_raise_height != other.foot_raise_height:
            return False
        if self.body_height != other.body_height:
            return False
        if all(self.position != other.position):
            return False
        if all(self.euler != other.euler):
            return False
        if all(self.velocity != other.velocity):
            return False
        if self.yaw_speed != other.yaw_speed:
            return False
        if self.bms_cmd != other.bms_cmd:
            return False
        if self.path_point != other.path_point:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def mode(self):
        """Message field 'mode'."""
        return self._mode

    @mode.setter
    def mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'mode' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'mode' field must be an unsigned integer in [0, 255]"
        self._mode = value

    @property
    def gait_type(self):
        """Message field 'gait_type'."""
        return self._gait_type

    @gait_type.setter
    def gait_type(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'gait_type' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'gait_type' field must be an unsigned integer in [0, 255]"
        self._gait_type = value

    @property
    def speed_level(self):
        """Message field 'speed_level'."""
        return self._speed_level

    @speed_level.setter
    def speed_level(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'speed_level' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'speed_level' field must be an unsigned integer in [0, 255]"
        self._speed_level = value

    @property
    def foot_raise_height(self):
        """Message field 'foot_raise_height'."""
        return self._foot_raise_height

    @foot_raise_height.setter
    def foot_raise_height(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'foot_raise_height' field must be of type 'float'"
        self._foot_raise_height = value

    @property
    def body_height(self):
        """Message field 'body_height'."""
        return self._body_height

    @body_height.setter
    def body_height(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'body_height' field must be of type 'float'"
        self._body_height = value

    @property
    def position(self):
        """Message field 'position'."""
        return self._position

    @position.setter
    def position(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'position' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 2, \
                "The 'position' numpy.ndarray() must have a size of 2"
            self._position = value
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
                "The 'position' field must be a set or sequence with length 2 and each value of type 'float'"
        self._position = numpy.array(value, dtype=numpy.float32)

    @property
    def euler(self):
        """Message field 'euler'."""
        return self._euler

    @euler.setter
    def euler(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'euler' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 3, \
                "The 'euler' numpy.ndarray() must have a size of 3"
            self._euler = value
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
                 len(value) == 3 and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'euler' field must be a set or sequence with length 3 and each value of type 'float'"
        self._euler = numpy.array(value, dtype=numpy.float32)

    @property
    def velocity(self):
        """Message field 'velocity'."""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'velocity' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 2, \
                "The 'velocity' numpy.ndarray() must have a size of 2"
            self._velocity = value
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
                "The 'velocity' field must be a set or sequence with length 2 and each value of type 'float'"
        self._velocity = numpy.array(value, dtype=numpy.float32)

    @property
    def yaw_speed(self):
        """Message field 'yaw_speed'."""
        return self._yaw_speed

    @yaw_speed.setter
    def yaw_speed(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'yaw_speed' field must be of type 'float'"
        self._yaw_speed = value

    @property
    def bms_cmd(self):
        """Message field 'bms_cmd'."""
        return self._bms_cmd

    @bms_cmd.setter
    def bms_cmd(self, value):
        if __debug__:
            from unitree_go.msg import BmsCmd
            assert \
                isinstance(value, BmsCmd), \
                "The 'bms_cmd' field must be a sub message of type 'BmsCmd'"
        self._bms_cmd = value

    @property
    def path_point(self):
        """Message field 'path_point'."""
        return self._path_point

    @path_point.setter
    def path_point(self, value):
        if __debug__:
            from unitree_go.msg import PathPoint
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
                 len(value) == 30 and
                 all(isinstance(v, PathPoint) for v in value) and
                 True), \
                "The 'path_point' field must be a set or sequence with length 30 and each value of type 'PathPoint'"
        self._path_point = value
