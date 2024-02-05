# generated from rosidl_generator_py/resource/_idl.py.em
# with input from unitree_go:msg/SportModeState.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'position'
# Member 'velocity'
# Member 'range_obstacle'
# Member 'foot_force'
# Member 'foot_position_body'
# Member 'foot_speed_body'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_SportModeState(type):
    """Metaclass of message 'SportModeState'."""

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
                'unitree_go.msg.SportModeState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__sport_mode_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__sport_mode_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__sport_mode_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__sport_mode_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__sport_mode_state

            from unitree_go.msg import IMUState
            if IMUState.__class__._TYPE_SUPPORT is None:
                IMUState.__class__.__import_type_support__()

            from unitree_go.msg import TimeSpec
            if TimeSpec.__class__._TYPE_SUPPORT is None:
                TimeSpec.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class SportModeState(metaclass=Metaclass_SportModeState):
    """Message class 'SportModeState'."""

    __slots__ = [
        '_stamp',
        '_error_code',
        '_imu_state',
        '_mode',
        '_progress',
        '_gait_type',
        '_foot_raise_height',
        '_position',
        '_body_height',
        '_velocity',
        '_yaw_speed',
        '_range_obstacle',
        '_foot_force',
        '_foot_position_body',
        '_foot_speed_body',
    ]

    _fields_and_field_types = {
        'stamp': 'unitree_go/TimeSpec',
        'error_code': 'uint32',
        'imu_state': 'unitree_go/IMUState',
        'mode': 'uint8',
        'progress': 'float',
        'gait_type': 'uint8',
        'foot_raise_height': 'float',
        'position': 'float[3]',
        'body_height': 'float',
        'velocity': 'float[3]',
        'yaw_speed': 'float',
        'range_obstacle': 'float[4]',
        'foot_force': 'int16[4]',
        'foot_position_body': 'float[12]',
        'foot_speed_body': 'float[12]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['unitree_go', 'msg'], 'TimeSpec'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['unitree_go', 'msg'], 'IMUState'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 3),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 3),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int16'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 12),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 12),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from unitree_go.msg import TimeSpec
        self.stamp = kwargs.get('stamp', TimeSpec())
        self.error_code = kwargs.get('error_code', int())
        from unitree_go.msg import IMUState
        self.imu_state = kwargs.get('imu_state', IMUState())
        self.mode = kwargs.get('mode', int())
        self.progress = kwargs.get('progress', float())
        self.gait_type = kwargs.get('gait_type', int())
        self.foot_raise_height = kwargs.get('foot_raise_height', float())
        if 'position' not in kwargs:
            self.position = numpy.zeros(3, dtype=numpy.float32)
        else:
            self.position = numpy.array(kwargs.get('position'), dtype=numpy.float32)
            assert self.position.shape == (3, )
        self.body_height = kwargs.get('body_height', float())
        if 'velocity' not in kwargs:
            self.velocity = numpy.zeros(3, dtype=numpy.float32)
        else:
            self.velocity = numpy.array(kwargs.get('velocity'), dtype=numpy.float32)
            assert self.velocity.shape == (3, )
        self.yaw_speed = kwargs.get('yaw_speed', float())
        if 'range_obstacle' not in kwargs:
            self.range_obstacle = numpy.zeros(4, dtype=numpy.float32)
        else:
            self.range_obstacle = numpy.array(kwargs.get('range_obstacle'), dtype=numpy.float32)
            assert self.range_obstacle.shape == (4, )
        if 'foot_force' not in kwargs:
            self.foot_force = numpy.zeros(4, dtype=numpy.int16)
        else:
            self.foot_force = numpy.array(kwargs.get('foot_force'), dtype=numpy.int16)
            assert self.foot_force.shape == (4, )
        if 'foot_position_body' not in kwargs:
            self.foot_position_body = numpy.zeros(12, dtype=numpy.float32)
        else:
            self.foot_position_body = numpy.array(kwargs.get('foot_position_body'), dtype=numpy.float32)
            assert self.foot_position_body.shape == (12, )
        if 'foot_speed_body' not in kwargs:
            self.foot_speed_body = numpy.zeros(12, dtype=numpy.float32)
        else:
            self.foot_speed_body = numpy.array(kwargs.get('foot_speed_body'), dtype=numpy.float32)
            assert self.foot_speed_body.shape == (12, )

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
        if self.error_code != other.error_code:
            return False
        if self.imu_state != other.imu_state:
            return False
        if self.mode != other.mode:
            return False
        if self.progress != other.progress:
            return False
        if self.gait_type != other.gait_type:
            return False
        if self.foot_raise_height != other.foot_raise_height:
            return False
        if all(self.position != other.position):
            return False
        if self.body_height != other.body_height:
            return False
        if all(self.velocity != other.velocity):
            return False
        if self.yaw_speed != other.yaw_speed:
            return False
        if all(self.range_obstacle != other.range_obstacle):
            return False
        if all(self.foot_force != other.foot_force):
            return False
        if all(self.foot_position_body != other.foot_position_body):
            return False
        if all(self.foot_speed_body != other.foot_speed_body):
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
            from unitree_go.msg import TimeSpec
            assert \
                isinstance(value, TimeSpec), \
                "The 'stamp' field must be a sub message of type 'TimeSpec'"
        self._stamp = value

    @property
    def error_code(self):
        """Message field 'error_code'."""
        return self._error_code

    @error_code.setter
    def error_code(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'error_code' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'error_code' field must be an unsigned integer in [0, 4294967295]"
        self._error_code = value

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
    def progress(self):
        """Message field 'progress'."""
        return self._progress

    @progress.setter
    def progress(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'progress' field must be of type 'float'"
        self._progress = value

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
    def position(self):
        """Message field 'position'."""
        return self._position

    @position.setter
    def position(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'position' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 3, \
                "The 'position' numpy.ndarray() must have a size of 3"
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
                 len(value) == 3 and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'position' field must be a set or sequence with length 3 and each value of type 'float'"
        self._position = numpy.array(value, dtype=numpy.float32)

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
    def velocity(self):
        """Message field 'velocity'."""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'velocity' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 3, \
                "The 'velocity' numpy.ndarray() must have a size of 3"
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
                 len(value) == 3 and
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'velocity' field must be a set or sequence with length 3 and each value of type 'float'"
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
    def range_obstacle(self):
        """Message field 'range_obstacle'."""
        return self._range_obstacle

    @range_obstacle.setter
    def range_obstacle(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'range_obstacle' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 4, \
                "The 'range_obstacle' numpy.ndarray() must have a size of 4"
            self._range_obstacle = value
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
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'range_obstacle' field must be a set or sequence with length 4 and each value of type 'float'"
        self._range_obstacle = numpy.array(value, dtype=numpy.float32)

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
    def foot_position_body(self):
        """Message field 'foot_position_body'."""
        return self._foot_position_body

    @foot_position_body.setter
    def foot_position_body(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'foot_position_body' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 12, \
                "The 'foot_position_body' numpy.ndarray() must have a size of 12"
            self._foot_position_body = value
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
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'foot_position_body' field must be a set or sequence with length 12 and each value of type 'float'"
        self._foot_position_body = numpy.array(value, dtype=numpy.float32)

    @property
    def foot_speed_body(self):
        """Message field 'foot_speed_body'."""
        return self._foot_speed_body

    @foot_speed_body.setter
    def foot_speed_body(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'foot_speed_body' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 12, \
                "The 'foot_speed_body' numpy.ndarray() must have a size of 12"
            self._foot_speed_body = value
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
                 all(isinstance(v, float) for v in value) and
                 True), \
                "The 'foot_speed_body' field must be a set or sequence with length 12 and each value of type 'float'"
        self._foot_speed_body = numpy.array(value, dtype=numpy.float32)
