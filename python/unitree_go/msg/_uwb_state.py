# generated from rosidl_generator_py/resource/_idl.py.em
# with input from unitree_go:msg/UwbState.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'version'
# Member 'joystick'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_UwbState(type):
    """Metaclass of message 'UwbState'."""

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
                'unitree_go.msg.UwbState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__uwb_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__uwb_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__uwb_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__uwb_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__uwb_state

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class UwbState(metaclass=Metaclass_UwbState):
    """Message class 'UwbState'."""

    __slots__ = [
        '_version',
        '_channel',
        '_joy_mode',
        '_orientation_est',
        '_pitch_est',
        '_distance_est',
        '_yaw_est',
        '_tag_roll',
        '_tag_pitch',
        '_tag_yaw',
        '_base_roll',
        '_base_pitch',
        '_base_yaw',
        '_joystick',
        '_error_state',
        '_buttons',
        '_enabled_from_app',
    ]

    _fields_and_field_types = {
        'version': 'uint8[2]',
        'channel': 'uint8',
        'joy_mode': 'uint8',
        'orientation_est': 'float',
        'pitch_est': 'float',
        'distance_est': 'float',
        'yaw_est': 'float',
        'tag_roll': 'float',
        'tag_pitch': 'float',
        'tag_yaw': 'float',
        'base_roll': 'float',
        'base_pitch': 'float',
        'base_yaw': 'float',
        'joystick': 'float[2]',
        'error_state': 'uint8',
        'buttons': 'uint8',
        'enabled_from_app': 'uint8',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint8'), 2),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 2),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        if 'version' not in kwargs:
            self.version = numpy.zeros(2, dtype=numpy.uint8)
        else:
            self.version = numpy.array(kwargs.get('version'), dtype=numpy.uint8)
            assert self.version.shape == (2, )
        self.channel = kwargs.get('channel', int())
        self.joy_mode = kwargs.get('joy_mode', int())
        self.orientation_est = kwargs.get('orientation_est', float())
        self.pitch_est = kwargs.get('pitch_est', float())
        self.distance_est = kwargs.get('distance_est', float())
        self.yaw_est = kwargs.get('yaw_est', float())
        self.tag_roll = kwargs.get('tag_roll', float())
        self.tag_pitch = kwargs.get('tag_pitch', float())
        self.tag_yaw = kwargs.get('tag_yaw', float())
        self.base_roll = kwargs.get('base_roll', float())
        self.base_pitch = kwargs.get('base_pitch', float())
        self.base_yaw = kwargs.get('base_yaw', float())
        if 'joystick' not in kwargs:
            self.joystick = numpy.zeros(2, dtype=numpy.float32)
        else:
            self.joystick = numpy.array(kwargs.get('joystick'), dtype=numpy.float32)
            assert self.joystick.shape == (2, )
        self.error_state = kwargs.get('error_state', int())
        self.buttons = kwargs.get('buttons', int())
        self.enabled_from_app = kwargs.get('enabled_from_app', int())

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
        if all(self.version != other.version):
            return False
        if self.channel != other.channel:
            return False
        if self.joy_mode != other.joy_mode:
            return False
        if self.orientation_est != other.orientation_est:
            return False
        if self.pitch_est != other.pitch_est:
            return False
        if self.distance_est != other.distance_est:
            return False
        if self.yaw_est != other.yaw_est:
            return False
        if self.tag_roll != other.tag_roll:
            return False
        if self.tag_pitch != other.tag_pitch:
            return False
        if self.tag_yaw != other.tag_yaw:
            return False
        if self.base_roll != other.base_roll:
            return False
        if self.base_pitch != other.base_pitch:
            return False
        if self.base_yaw != other.base_yaw:
            return False
        if all(self.joystick != other.joystick):
            return False
        if self.error_state != other.error_state:
            return False
        if self.buttons != other.buttons:
            return False
        if self.enabled_from_app != other.enabled_from_app:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def version(self):
        """Message field 'version'."""
        return self._version

    @version.setter
    def version(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint8, \
                "The 'version' numpy.ndarray() must have the dtype of 'numpy.uint8'"
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
                 all(val >= 0 and val < 256 for val in value)), \
                "The 'version' field must be a set or sequence with length 2 and each value of type 'int' and each unsigned integer in [0, 255]"
        self._version = numpy.array(value, dtype=numpy.uint8)

    @property
    def channel(self):
        """Message field 'channel'."""
        return self._channel

    @channel.setter
    def channel(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'channel' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'channel' field must be an unsigned integer in [0, 255]"
        self._channel = value

    @property
    def joy_mode(self):
        """Message field 'joy_mode'."""
        return self._joy_mode

    @joy_mode.setter
    def joy_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'joy_mode' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'joy_mode' field must be an unsigned integer in [0, 255]"
        self._joy_mode = value

    @property
    def orientation_est(self):
        """Message field 'orientation_est'."""
        return self._orientation_est

    @orientation_est.setter
    def orientation_est(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'orientation_est' field must be of type 'float'"
        self._orientation_est = value

    @property
    def pitch_est(self):
        """Message field 'pitch_est'."""
        return self._pitch_est

    @pitch_est.setter
    def pitch_est(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'pitch_est' field must be of type 'float'"
        self._pitch_est = value

    @property
    def distance_est(self):
        """Message field 'distance_est'."""
        return self._distance_est

    @distance_est.setter
    def distance_est(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'distance_est' field must be of type 'float'"
        self._distance_est = value

    @property
    def yaw_est(self):
        """Message field 'yaw_est'."""
        return self._yaw_est

    @yaw_est.setter
    def yaw_est(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'yaw_est' field must be of type 'float'"
        self._yaw_est = value

    @property
    def tag_roll(self):
        """Message field 'tag_roll'."""
        return self._tag_roll

    @tag_roll.setter
    def tag_roll(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'tag_roll' field must be of type 'float'"
        self._tag_roll = value

    @property
    def tag_pitch(self):
        """Message field 'tag_pitch'."""
        return self._tag_pitch

    @tag_pitch.setter
    def tag_pitch(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'tag_pitch' field must be of type 'float'"
        self._tag_pitch = value

    @property
    def tag_yaw(self):
        """Message field 'tag_yaw'."""
        return self._tag_yaw

    @tag_yaw.setter
    def tag_yaw(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'tag_yaw' field must be of type 'float'"
        self._tag_yaw = value

    @property
    def base_roll(self):
        """Message field 'base_roll'."""
        return self._base_roll

    @base_roll.setter
    def base_roll(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'base_roll' field must be of type 'float'"
        self._base_roll = value

    @property
    def base_pitch(self):
        """Message field 'base_pitch'."""
        return self._base_pitch

    @base_pitch.setter
    def base_pitch(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'base_pitch' field must be of type 'float'"
        self._base_pitch = value

    @property
    def base_yaw(self):
        """Message field 'base_yaw'."""
        return self._base_yaw

    @base_yaw.setter
    def base_yaw(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'base_yaw' field must be of type 'float'"
        self._base_yaw = value

    @property
    def joystick(self):
        """Message field 'joystick'."""
        return self._joystick

    @joystick.setter
    def joystick(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'joystick' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 2, \
                "The 'joystick' numpy.ndarray() must have a size of 2"
            self._joystick = value
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
                "The 'joystick' field must be a set or sequence with length 2 and each value of type 'float'"
        self._joystick = numpy.array(value, dtype=numpy.float32)

    @property
    def error_state(self):
        """Message field 'error_state'."""
        return self._error_state

    @error_state.setter
    def error_state(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'error_state' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'error_state' field must be an unsigned integer in [0, 255]"
        self._error_state = value

    @property
    def buttons(self):
        """Message field 'buttons'."""
        return self._buttons

    @buttons.setter
    def buttons(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'buttons' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'buttons' field must be an unsigned integer in [0, 255]"
        self._buttons = value

    @property
    def enabled_from_app(self):
        """Message field 'enabled_from_app'."""
        return self._enabled_from_app

    @enabled_from_app.setter
    def enabled_from_app(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'enabled_from_app' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'enabled_from_app' field must be an unsigned integer in [0, 255]"
        self._enabled_from_app = value
