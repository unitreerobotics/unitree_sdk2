# generated from rosidl_generator_py/resource/_idl.py.em
# with input from unitree_go:msg/LidarState.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'imu_rpy'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_LidarState(type):
    """Metaclass of message 'LidarState'."""

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
                'unitree_go.msg.LidarState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__lidar_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__lidar_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__lidar_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__lidar_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__lidar_state

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class LidarState(metaclass=Metaclass_LidarState):
    """Message class 'LidarState'."""

    __slots__ = [
        '_stamp',
        '_firmware_version',
        '_software_version',
        '_sdk_version',
        '_sys_rotation_speed',
        '_com_rotation_speed',
        '_error_state',
        '_cloud_frequency',
        '_cloud_packet_loss_rate',
        '_cloud_size',
        '_cloud_scan_num',
        '_imu_frequency',
        '_imu_packet_loss_rate',
        '_imu_rpy',
        '_serial_recv_stamp',
        '_serial_buffer_size',
        '_serial_buffer_read',
    ]

    _fields_and_field_types = {
        'stamp': 'double',
        'firmware_version': 'string',
        'software_version': 'string',
        'sdk_version': 'string',
        'sys_rotation_speed': 'float',
        'com_rotation_speed': 'float',
        'error_state': 'uint8',
        'cloud_frequency': 'float',
        'cloud_packet_loss_rate': 'float',
        'cloud_size': 'uint32',
        'cloud_scan_num': 'uint32',
        'imu_frequency': 'float',
        'imu_packet_loss_rate': 'float',
        'imu_rpy': 'float[3]',
        'serial_recv_stamp': 'double',
        'serial_buffer_size': 'uint32',
        'serial_buffer_read': 'uint32',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 3),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.stamp = kwargs.get('stamp', float())
        self.firmware_version = kwargs.get('firmware_version', str())
        self.software_version = kwargs.get('software_version', str())
        self.sdk_version = kwargs.get('sdk_version', str())
        self.sys_rotation_speed = kwargs.get('sys_rotation_speed', float())
        self.com_rotation_speed = kwargs.get('com_rotation_speed', float())
        self.error_state = kwargs.get('error_state', int())
        self.cloud_frequency = kwargs.get('cloud_frequency', float())
        self.cloud_packet_loss_rate = kwargs.get('cloud_packet_loss_rate', float())
        self.cloud_size = kwargs.get('cloud_size', int())
        self.cloud_scan_num = kwargs.get('cloud_scan_num', int())
        self.imu_frequency = kwargs.get('imu_frequency', float())
        self.imu_packet_loss_rate = kwargs.get('imu_packet_loss_rate', float())
        if 'imu_rpy' not in kwargs:
            self.imu_rpy = numpy.zeros(3, dtype=numpy.float32)
        else:
            self.imu_rpy = numpy.array(kwargs.get('imu_rpy'), dtype=numpy.float32)
            assert self.imu_rpy.shape == (3, )
        self.serial_recv_stamp = kwargs.get('serial_recv_stamp', float())
        self.serial_buffer_size = kwargs.get('serial_buffer_size', int())
        self.serial_buffer_read = kwargs.get('serial_buffer_read', int())

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
        if self.firmware_version != other.firmware_version:
            return False
        if self.software_version != other.software_version:
            return False
        if self.sdk_version != other.sdk_version:
            return False
        if self.sys_rotation_speed != other.sys_rotation_speed:
            return False
        if self.com_rotation_speed != other.com_rotation_speed:
            return False
        if self.error_state != other.error_state:
            return False
        if self.cloud_frequency != other.cloud_frequency:
            return False
        if self.cloud_packet_loss_rate != other.cloud_packet_loss_rate:
            return False
        if self.cloud_size != other.cloud_size:
            return False
        if self.cloud_scan_num != other.cloud_scan_num:
            return False
        if self.imu_frequency != other.imu_frequency:
            return False
        if self.imu_packet_loss_rate != other.imu_packet_loss_rate:
            return False
        if all(self.imu_rpy != other.imu_rpy):
            return False
        if self.serial_recv_stamp != other.serial_recv_stamp:
            return False
        if self.serial_buffer_size != other.serial_buffer_size:
            return False
        if self.serial_buffer_read != other.serial_buffer_read:
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
    def firmware_version(self):
        """Message field 'firmware_version'."""
        return self._firmware_version

    @firmware_version.setter
    def firmware_version(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'firmware_version' field must be of type 'str'"
        self._firmware_version = value

    @property
    def software_version(self):
        """Message field 'software_version'."""
        return self._software_version

    @software_version.setter
    def software_version(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'software_version' field must be of type 'str'"
        self._software_version = value

    @property
    def sdk_version(self):
        """Message field 'sdk_version'."""
        return self._sdk_version

    @sdk_version.setter
    def sdk_version(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'sdk_version' field must be of type 'str'"
        self._sdk_version = value

    @property
    def sys_rotation_speed(self):
        """Message field 'sys_rotation_speed'."""
        return self._sys_rotation_speed

    @sys_rotation_speed.setter
    def sys_rotation_speed(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'sys_rotation_speed' field must be of type 'float'"
        self._sys_rotation_speed = value

    @property
    def com_rotation_speed(self):
        """Message field 'com_rotation_speed'."""
        return self._com_rotation_speed

    @com_rotation_speed.setter
    def com_rotation_speed(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'com_rotation_speed' field must be of type 'float'"
        self._com_rotation_speed = value

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
    def cloud_frequency(self):
        """Message field 'cloud_frequency'."""
        return self._cloud_frequency

    @cloud_frequency.setter
    def cloud_frequency(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'cloud_frequency' field must be of type 'float'"
        self._cloud_frequency = value

    @property
    def cloud_packet_loss_rate(self):
        """Message field 'cloud_packet_loss_rate'."""
        return self._cloud_packet_loss_rate

    @cloud_packet_loss_rate.setter
    def cloud_packet_loss_rate(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'cloud_packet_loss_rate' field must be of type 'float'"
        self._cloud_packet_loss_rate = value

    @property
    def cloud_size(self):
        """Message field 'cloud_size'."""
        return self._cloud_size

    @cloud_size.setter
    def cloud_size(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'cloud_size' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'cloud_size' field must be an unsigned integer in [0, 4294967295]"
        self._cloud_size = value

    @property
    def cloud_scan_num(self):
        """Message field 'cloud_scan_num'."""
        return self._cloud_scan_num

    @cloud_scan_num.setter
    def cloud_scan_num(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'cloud_scan_num' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'cloud_scan_num' field must be an unsigned integer in [0, 4294967295]"
        self._cloud_scan_num = value

    @property
    def imu_frequency(self):
        """Message field 'imu_frequency'."""
        return self._imu_frequency

    @imu_frequency.setter
    def imu_frequency(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'imu_frequency' field must be of type 'float'"
        self._imu_frequency = value

    @property
    def imu_packet_loss_rate(self):
        """Message field 'imu_packet_loss_rate'."""
        return self._imu_packet_loss_rate

    @imu_packet_loss_rate.setter
    def imu_packet_loss_rate(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'imu_packet_loss_rate' field must be of type 'float'"
        self._imu_packet_loss_rate = value

    @property
    def imu_rpy(self):
        """Message field 'imu_rpy'."""
        return self._imu_rpy

    @imu_rpy.setter
    def imu_rpy(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'imu_rpy' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 3, \
                "The 'imu_rpy' numpy.ndarray() must have a size of 3"
            self._imu_rpy = value
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
                "The 'imu_rpy' field must be a set or sequence with length 3 and each value of type 'float'"
        self._imu_rpy = numpy.array(value, dtype=numpy.float32)

    @property
    def serial_recv_stamp(self):
        """Message field 'serial_recv_stamp'."""
        return self._serial_recv_stamp

    @serial_recv_stamp.setter
    def serial_recv_stamp(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'serial_recv_stamp' field must be of type 'float'"
        self._serial_recv_stamp = value

    @property
    def serial_buffer_size(self):
        """Message field 'serial_buffer_size'."""
        return self._serial_buffer_size

    @serial_buffer_size.setter
    def serial_buffer_size(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'serial_buffer_size' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'serial_buffer_size' field must be an unsigned integer in [0, 4294967295]"
        self._serial_buffer_size = value

    @property
    def serial_buffer_read(self):
        """Message field 'serial_buffer_read'."""
        return self._serial_buffer_read

    @serial_buffer_read.setter
    def serial_buffer_read(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'serial_buffer_read' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'serial_buffer_read' field must be an unsigned integer in [0, 4294967295]"
        self._serial_buffer_read = value
