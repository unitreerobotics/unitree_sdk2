# generated from rosidl_generator_py/resource/_idl.py.em
# with input from unitree_go:msg/LowCmd.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'head'
# Member 'sn'
# Member 'version'
# Member 'wireless_remote'
# Member 'led'
# Member 'fan'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_LowCmd(type):
    """Metaclass of message 'LowCmd'."""

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
                'unitree_go.msg.LowCmd')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__low_cmd
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__low_cmd
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__low_cmd
            cls._TYPE_SUPPORT = module.type_support_msg__msg__low_cmd
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__low_cmd

            from unitree_go.msg import BmsCmd
            if BmsCmd.__class__._TYPE_SUPPORT is None:
                BmsCmd.__class__.__import_type_support__()

            from unitree_go.msg import MotorCmd
            if MotorCmd.__class__._TYPE_SUPPORT is None:
                MotorCmd.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class LowCmd(metaclass=Metaclass_LowCmd):
    """Message class 'LowCmd'."""

    __slots__ = [
        '_head',
        '_level_flag',
        '_frame_reserve',
        '_sn',
        '_version',
        '_bandwidth',
        '_motor_cmd',
        '_bms_cmd',
        '_wireless_remote',
        '_led',
        '_fan',
        '_gpio',
        '_reserve',
        '_crc',
    ]

    _fields_and_field_types = {
        'head': 'uint8[2]',
        'level_flag': 'uint8',
        'frame_reserve': 'uint8',
        'sn': 'uint32[2]',
        'version': 'uint32[2]',
        'bandwidth': 'uint16',
        'motor_cmd': 'unitree_go/MotorCmd[20]',
        'bms_cmd': 'unitree_go/BmsCmd',
        'wireless_remote': 'uint8[40]',
        'led': 'uint8[12]',
        'fan': 'uint8[2]',
        'gpio': 'uint8',
        'reserve': 'uint32',
        'crc': 'uint32',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 2),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint32'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint32'), 2),  # noqa: E501
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.NamespacedType(['unitree_go', 'msg'], 'MotorCmd'), 20),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['unitree_go', 'msg'], 'BmsCmd'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 40),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 2),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        if 'head' not in kwargs:
            self.head = numpy.zeros(2, dtype=numpy.uint8)
        else:
            self.head = numpy.array(kwargs.get('head'), dtype=numpy.uint8)
            assert self.head.shape == (2, )
        self.level_flag = kwargs.get('level_flag', int())
        self.frame_reserve = kwargs.get('frame_reserve', int())
        if 'sn' not in kwargs:
            self.sn = numpy.zeros(2, dtype=numpy.uint32)
        else:
            self.sn = numpy.array(kwargs.get('sn'), dtype=numpy.uint32)
            assert self.sn.shape == (2, )
        if 'version' not in kwargs:
            self.version = numpy.zeros(2, dtype=numpy.uint32)
        else:
            self.version = numpy.array(kwargs.get('version'), dtype=numpy.uint32)
            assert self.version.shape == (2, )
        self.bandwidth = kwargs.get('bandwidth', int())
        from unitree_go.msg import MotorCmd
        self.motor_cmd = kwargs.get(
            'motor_cmd',
            [MotorCmd() for x in range(20)]
        )
        from unitree_go.msg import BmsCmd
        self.bms_cmd = kwargs.get('bms_cmd', BmsCmd())
        if 'wireless_remote' not in kwargs:
            self.wireless_remote = numpy.zeros(40, dtype=numpy.uint8)
        else:
            self.wireless_remote = numpy.array(kwargs.get('wireless_remote'), dtype=numpy.uint8)
            assert self.wireless_remote.shape == (40, )
        if 'led' not in kwargs:
            self.led = numpy.zeros(12, dtype=numpy.uint8)
        else:
            self.led = numpy.array(kwargs.get('led'), dtype=numpy.uint8)
            assert self.led.shape == (12, )
        if 'fan' not in kwargs:
            self.fan = numpy.zeros(2, dtype=numpy.uint8)
        else:
            self.fan = numpy.array(kwargs.get('fan'), dtype=numpy.uint8)
            assert self.fan.shape == (2, )
        self.gpio = kwargs.get('gpio', int())
        self.reserve = kwargs.get('reserve', int())
        self.crc = kwargs.get('crc', int())

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
        if all(self.head != other.head):
            return False
        if self.level_flag != other.level_flag:
            return False
        if self.frame_reserve != other.frame_reserve:
            return False
        if all(self.sn != other.sn):
            return False
        if all(self.version != other.version):
            return False
        if self.bandwidth != other.bandwidth:
            return False
        if self.motor_cmd != other.motor_cmd:
            return False
        if self.bms_cmd != other.bms_cmd:
            return False
        if all(self.wireless_remote != other.wireless_remote):
            return False
        if all(self.led != other.led):
            return False
        if all(self.fan != other.fan):
            return False
        if self.gpio != other.gpio:
            return False
        if self.reserve != other.reserve:
            return False
        if self.crc != other.crc:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def head(self):
        """Message field 'head'."""
        return self._head

    @head.setter
    def head(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint8, \
                "The 'head' numpy.ndarray() must have the dtype of 'numpy.uint8'"
            assert value.size == 2, \
                "The 'head' numpy.ndarray() must have a size of 2"
            self._head = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'head' field must be a set or sequence with length 2 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._head = numpy.array(value, dtype=numpy.uint8)

    @property
    def level_flag(self):
        """Message field 'level_flag'."""
        return self._level_flag

    @level_flag.setter
    def level_flag(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'level_flag' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'level_flag' field must be an unsigned integer in [0, 255]"
        self._level_flag = value

    @property
    def frame_reserve(self):
        """Message field 'frame_reserve'."""
        return self._frame_reserve

    @frame_reserve.setter
    def frame_reserve(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'frame_reserve' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'frame_reserve' field must be an unsigned integer in [0, 255]"
        self._frame_reserve = value

    @property
    def sn(self):
        """Message field 'sn'."""
        return self._sn

    @sn.setter
    def sn(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint32, \
                "The 'sn' numpy.ndarray() must have the dtype of 'numpy.uint32'"
            assert value.size == 2, \
                "The 'sn' numpy.ndarray() must have a size of 2"
            self._sn = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 4294967296 for val in value)), \
                "The 'sn' field must be a set or sequence with length 2 and each value of type 'int' and each unsigned integer in [0, 4294967295]"
        self._sn = numpy.array(value, dtype=numpy.uint32)

    @property
    def version(self):
        """Message field 'version'."""
        return self._version

    @version.setter
    def version(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint32, \
                "The 'version' numpy.ndarray() must have the dtype of 'numpy.uint32'"
            assert value.size == 2, \
                "The 'version' numpy.ndarray() must have a size of 2"
            self._version = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 4294967296 for val in value)), \
                "The 'version' field must be a set or sequence with length 2 and each value of type 'int' and each unsigned integer in [0, 4294967295]"
        self._version = numpy.array(value, dtype=numpy.uint32)

    @property
    def bandwidth(self):
        """Message field 'bandwidth'."""
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'bandwidth' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'bandwidth' field must be an unsigned integer in [0, 65535]"
        self._bandwidth = value

    @property
    def motor_cmd(self):
        """Message field 'motor_cmd'."""
        return self._motor_cmd

    @motor_cmd.setter
    def motor_cmd(self, value):
        if __debug__:
            from unitree_go.msg import MotorCmd
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
                 len(value) == 20 and
                 all(isinstance(v, MotorCmd) for v in value) and
                 True), \
                "The 'motor_cmd' field must be a set or sequence with length 20 and each value of type 'MotorCmd'"
        self._motor_cmd = value

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
    def wireless_remote(self):
        """Message field 'wireless_remote'."""
        return self._wireless_remote

    @wireless_remote.setter
    def wireless_remote(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint8, \
                "The 'wireless_remote' numpy.ndarray() must have the dtype of 'numpy.uint8'"
            assert value.size == 40, \
                "The 'wireless_remote' numpy.ndarray() must have a size of 40"
            self._wireless_remote = value
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
                 len(value) == 40 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'wireless_remote' field must be a set or sequence with length 40 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._wireless_remote = numpy.array(value, dtype=numpy.uint8)

    @property
    def led(self):
        """Message field 'led'."""
        return self._led

    @led.setter
    def led(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint8, \
                "The 'led' numpy.ndarray() must have the dtype of 'numpy.uint8'"
            assert value.size == 12, \
                "The 'led' numpy.ndarray() must have a size of 12"
            self._led = value
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
                 len(value) == 12 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'led' field must be a set or sequence with length 12 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._led = numpy.array(value, dtype=numpy.uint8)

    @property
    def fan(self):
        """Message field 'fan'."""
        return self._fan

    @fan.setter
    def fan(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint8, \
                "The 'fan' numpy.ndarray() must have the dtype of 'numpy.uint8'"
            assert value.size == 2, \
                "The 'fan' numpy.ndarray() must have a size of 2"
            self._fan = value
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
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'fan' field must be a set or sequence with length 2 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._fan = numpy.array(value, dtype=numpy.uint8)

    @property
    def gpio(self):
        """Message field 'gpio'."""
        return self._gpio

    @gpio.setter
    def gpio(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'gpio' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'gpio' field must be an unsigned integer in [0, 255]"
        self._gpio = value

    @property
    def reserve(self):
        """Message field 'reserve'."""
        return self._reserve

    @reserve.setter
    def reserve(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'reserve' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'reserve' field must be an unsigned integer in [0, 4294967295]"
        self._reserve = value

    @property
    def crc(self):
        """Message field 'crc'."""
        return self._crc

    @crc.setter
    def crc(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'crc' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'crc' field must be an unsigned integer in [0, 4294967295]"
        self._crc = value
