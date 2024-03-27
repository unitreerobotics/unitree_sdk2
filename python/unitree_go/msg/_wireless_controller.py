# generated from rosidl_generator_py/resource/_idl.py.em
# with input from unitree_go:msg/WirelessController.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_WirelessController(type):
    """Metaclass of message 'WirelessController'."""

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
                'unitree_go.msg.WirelessController')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__wireless_controller
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__wireless_controller
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__wireless_controller
            cls._TYPE_SUPPORT = module.type_support_msg__msg__wireless_controller
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__wireless_controller

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class WirelessController(metaclass=Metaclass_WirelessController):
    """Message class 'WirelessController'."""

    __slots__ = [
        '_lx',
        '_ly',
        '_rx',
        '_ry',
        '_keys',
    ]

    _fields_and_field_types = {
        'lx': 'float',
        'ly': 'float',
        'rx': 'float',
        'ry': 'float',
        'keys': 'uint16',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.lx = kwargs.get('lx', float())
        self.ly = kwargs.get('ly', float())
        self.rx = kwargs.get('rx', float())
        self.ry = kwargs.get('ry', float())
        self.keys = kwargs.get('keys', int())

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
        if self.lx != other.lx:
            return False
        if self.ly != other.ly:
            return False
        if self.rx != other.rx:
            return False
        if self.ry != other.ry:
            return False
        if self.keys != other.keys:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def lx(self):
        """Message field 'lx'."""
        return self._lx

    @lx.setter
    def lx(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'lx' field must be of type 'float'"
        self._lx = value

    @property
    def ly(self):
        """Message field 'ly'."""
        return self._ly

    @ly.setter
    def ly(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ly' field must be of type 'float'"
        self._ly = value

    @property
    def rx(self):
        """Message field 'rx'."""
        return self._rx

    @rx.setter
    def rx(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'rx' field must be of type 'float'"
        self._rx = value

    @property
    def ry(self):
        """Message field 'ry'."""
        return self._ry

    @ry.setter
    def ry(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'ry' field must be of type 'float'"
        self._ry = value

    @property
    def keys(self):
        """Message field 'keys'."""
        return self._keys

    @keys.setter
    def keys(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'keys' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'keys' field must be an unsigned integer in [0, 65535]"
        self._keys = value
