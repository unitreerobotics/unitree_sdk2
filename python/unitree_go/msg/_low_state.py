# generated from rosidl_generator_py/resource/_idl.py.em
# with input from unitree_go:msg/LowState.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'head'
# Member 'sn'
# Member 'version'
# Member 'foot_force'
# Member 'foot_force_est'
# Member 'wireless_remote'
# Member 'fan_frequency'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_LowState(type):
    """Metaclass of message 'LowState'."""

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
                'unitree_go.msg.LowState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__low_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__low_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__low_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__low_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__low_state

            from unitree_go.msg import BmsState
            if BmsState.__class__._TYPE_SUPPORT is None:
                BmsState.__class__.__import_type_support__()

            from unitree_go.msg import IMUState
            if IMUState.__class__._TYPE_SUPPORT is None:
                IMUState.__class__.__import_type_support__()

            from unitree_go.msg import MotorState
            if MotorState.__class__._TYPE_SUPPORT is None:
                MotorState.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class LowState(metaclass=Metaclass_LowState):
    """Message class 'LowState'."""

    __slots__ = [
        '_head',
        '_level_flag',
        '_frame_reserve',
        '_sn',
        '_version',
        '_bandwidth',
        '_imu_state',
        '_motor_state',
        '_bms_state',
        '_foot_force',
        '_foot_force_est',
        '_tick',
        '_wireless_remote',
        '_bit_flag',
        '_adc_reel',
        '_temperature_ntc1',
        '_temperature_ntc2',
        '_power_v',
        '_power_a',
        '_fan_frequency',
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
        'imu_state': 'unitree_go/IMUState',
        'motor_state': 'unitree_go/MotorState[20]',
        'bms_state': 'unitree_go/BmsState',
        'foot_force': 'int16[4]',
        'foot_force_est': 'int16[4]',
        'tick': 'uint32',
        'wireless_remote': 'uint8[40]',
        'bit_flag': 'uint8',
        'adc_reel': 'float',
        'temperature_ntc1': 'int8',
        'temperature_ntc2': 'int8',
        'power_v': 'float',
        'power_a': 'float',
        'fan_frequency': 'uint16[4]',
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
        rosidl_parser.definition.NamespacedType(['unitree_go', 'msg'], 'IMUState'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.NamespacedType(['unitree_go', 'msg'], 'MotorState'), 20),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['unitree_go', 'msg'], 'BmsState'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int16'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int16'), 4),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 40),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint16'), 4),  # noqa: E501
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
        from unitree_go.msg import IMUState
        self.imu_state = kwargs.get('imu_state', IMUState())
        from unitree_go.msg import MotorState
        self.motor_state = kwargs.get(
            'motor_state',
            [MotorState() for x in range(20)]
        )
        from unitree_go.msg import BmsState
        self.bms_state = kwargs.get('bms_state', BmsState())
        if 'foot_force' not in kwargs:
            self.foot_force = numpy.zeros(4, dtype=numpy.int16)
        else:
            self.foot_force = numpy.array(kwargs.get('foot_force'), dtype=numpy.int16)
            assert self.foot_force.shape == (4, )
        if 'foot_force_est' not in kwargs:
            self.foot_force_est = numpy.zeros(4, dtype=numpy.int16)
        else:
            self.foot_force_est = numpy.array(kwargs.get('foot_force_est'), dtype=numpy.int16)
            assert self.foot_force_est.shape == (4, )
        self.tick = kwargs.get('tick', int())
        if 'wireless_remote' not in kwargs:
            self.wireless_remote = numpy.zeros(40, dtype=numpy.uint8)
        else:
            self.wireless_remote = numpy.array(kwargs.get('wireless_remote'), dtype=numpy.uint8)
            assert self.wireless_remote.shape == (40, )
        self.bit_flag = kwargs.get('bit_flag', int())
        self.adc_reel = kwargs.get('adc_reel', float())
        self.temperature_ntc1 = kwargs.get('temperature_ntc1', int())
        self.temperature_ntc2 = kwargs.get('temperature_ntc2', int())
        self.power_v = kwargs.get('power_v', float())
        self.power_a = kwargs.get('power_a', float())
        if 'fan_frequency' not in kwargs:
            self.fan_frequency = numpy.zeros(4, dtype=numpy.uint16)
        else:
            self.fan_frequency = numpy.array(kwargs.get('fan_frequency'), dtype=numpy.uint16)
            assert self.fan_frequency.shape == (4, )
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
        if self.imu_state != other.imu_state:
            return False
        if self.motor_state != other.motor_state:
            return False
        if self.bms_state != other.bms_state:
            return False
        if all(self.foot_force != other.foot_force):
            return False
        if all(self.foot_force_est != other.foot_force_est):
            return False
        if self.tick != other.tick:
            return False
        if all(self.wireless_remote != other.wireless_remote):
            return False
        if self.bit_flag != other.bit_flag:
            return False
        if self.adc_reel != other.adc_reel:
            return False
        if self.temperature_ntc1 != other.temperature_ntc1:
            return False
        if self.temperature_ntc2 != other.temperature_ntc2:
            return False
        if self.power_v != other.power_v:
            return False
        if self.power_a != other.power_a:
            return False
        if all(self.fan_frequency != other.fan_frequency):
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
    def imu_state(self):
        """Message field 'imu_state'."""
        return self._imu_state

    @imu_state.setter
    def imu_state(self, value):
        if __debug__:
            from unitree_go.msg import IMUState
            assert \
                isinstance(value, IMUState), \
                "The 'imu_state' field must be a sub message of type 'IMUState'"
        self._imu_state = value

    @property
    def motor_state(self):
        """Message field 'motor_state'."""
        return self._motor_state

    @motor_state.setter
    def motor_state(self, value):
        if __debug__:
            from unitree_go.msg import MotorState
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
                 all(isinstance(v, MotorState) for v in value) and
                 True), \
                "The 'motor_state' field must be a set or sequence with length 20 and each value of type 'MotorState'"
        self._motor_state = value

    @property
    def bms_state(self):
        """Message field 'bms_state'."""
        return self._bms_state

    @bms_state.setter
    def bms_state(self, value):
        if __debug__:
            from unitree_go.msg import BmsState
            assert \
                isinstance(value, BmsState), \
                "The 'bms_state' field must be a sub message of type 'BmsState'"
        self._bms_state = value

    @property
    def foot_force(self):
        """Message field 'foot_force'."""
        return self._foot_force

    @foot_force.setter
    def foot_force(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.int16, \
                "The 'foot_force' numpy.ndarray() must have the dtype of 'numpy.int16'"
            assert value.size == 4, \
                "The 'foot_force' numpy.ndarray() must have a size of 4"
            self._foot_force = value
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
                 len(value) == 4 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= -32768 and val < 32768 for val in value)), \
                "The 'foot_force' field must be a set or sequence with length 4 and each value of type 'int' and each integer in [-32768, 32767]"
        self._foot_force = numpy.array(value, dtype=numpy.int16)

    @property
    def foot_force_est(self):
        """Message field 'foot_force_est'."""
        return self._foot_force_est

    @foot_force_est.setter
    def foot_force_est(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.int16, \
                "The 'foot_force_est' numpy.ndarray() must have the dtype of 'numpy.int16'"
            assert value.size == 4, \
                "The 'foot_force_est' numpy.ndarray() must have a size of 4"
            self._foot_force_est = value
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
                 len(value) == 4 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= -32768 and val < 32768 for val in value)), \
                "The 'foot_force_est' field must be a set or sequence with length 4 and each value of type 'int' and each integer in [-32768, 32767]"
        self._foot_force_est = numpy.array(value, dtype=numpy.int16)

    @property
    def tick(self):
        """Message field 'tick'."""
        return self._tick

    @tick.setter
    def tick(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'tick' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'tick' field must be an unsigned integer in [0, 4294967295]"
        self._tick = value

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
    def bit_flag(self):
        """Message field 'bit_flag'."""
        return self._bit_flag

    @bit_flag.setter
    def bit_flag(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'bit_flag' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'bit_flag' field must be an unsigned integer in [0, 255]"
        self._bit_flag = value

    @property
    def adc_reel(self):
        """Message field 'adc_reel'."""
        return self._adc_reel

    @adc_reel.setter
    def adc_reel(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'adc_reel' field must be of type 'float'"
        self._adc_reel = value

    @property
    def temperature_ntc1(self):
        """Message field 'temperature_ntc1'."""
        return self._temperature_ntc1

    @temperature_ntc1.setter
    def temperature_ntc1(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'temperature_ntc1' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'temperature_ntc1' field must be an integer in [-128, 127]"
        self._temperature_ntc1 = value

    @property
    def temperature_ntc2(self):
        """Message field 'temperature_ntc2'."""
        return self._temperature_ntc2

    @temperature_ntc2.setter
    def temperature_ntc2(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'temperature_ntc2' field must be of type 'int'"
            assert value >= -128 and value < 128, \
                "The 'temperature_ntc2' field must be an integer in [-128, 127]"
        self._temperature_ntc2 = value

    @property
    def power_v(self):
        """Message field 'power_v'."""
        return self._power_v

    @power_v.setter
    def power_v(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'power_v' field must be of type 'float'"
        self._power_v = value

    @property
    def power_a(self):
        """Message field 'power_a'."""
        return self._power_a

    @power_a.setter
    def power_a(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'power_a' field must be of type 'float'"
        self._power_a = value

    @property
    def fan_frequency(self):
        """Message field 'fan_frequency'."""
        return self._fan_frequency

    @fan_frequency.setter
    def fan_frequency(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint16, \
                "The 'fan_frequency' numpy.ndarray() must have the dtype of 'numpy.uint16'"
            assert value.size == 4, \
                "The 'fan_frequency' numpy.ndarray() must have a size of 4"
            self._fan_frequency = value
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
                 len(value) == 4 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 65536 for val in value)), \
                "The 'fan_frequency' field must be a set or sequence with length 4 and each value of type 'int' and each unsigned integer in [0, 65535]"
        self._fan_frequency = numpy.array(value, dtype=numpy.uint16)

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
