# generated from rosidl_generator_py/resource/_idl.py.em
# with input from unitree_go:msg/PathPoint.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PathPoint(type):
    """Metaclass of message 'PathPoint'."""

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
                'unitree_go.msg.PathPoint')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__path_point
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__path_point
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__path_point
            cls._TYPE_SUPPORT = module.type_support_msg__msg__path_point
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__path_point

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PathPoint(metaclass=Metaclass_PathPoint):
    """Message class 'PathPoint'."""

    __slots__ = [
        '_t_from_start',
        '_x',
        '_y',
        '_yaw',
        '_vx',
        '_vy',
        '_vyaw',
    ]

    _fields_and_field_types = {
        't_from_start': 'float',
        'x': 'float',
        'y': 'float',
        'yaw': 'float',
        'vx': 'float',
        'vy': 'float',
        'vyaw': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.t_from_start = kwargs.get('t_from_start', float())
        self.x = kwargs.get('x', float())
        self.y = kwargs.get('y', float())
        self.yaw = kwargs.get('yaw', float())
        self.vx = kwargs.get('vx', float())
        self.vy = kwargs.get('vy', float())
        self.vyaw = kwargs.get('vyaw', float())

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
        if self.t_from_start != other.t_from_start:
            return False
        if self.x != other.x:
            return False
        if self.y != other.y:
            return False
        if self.yaw != other.yaw:
            return False
        if self.vx != other.vx:
            return False
        if self.vy != other.vy:
            return False
        if self.vyaw != other.vyaw:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def t_from_start(self):
        """Message field 't_from_start'."""
        return self._t_from_start

    @t_from_start.setter
    def t_from_start(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 't_from_start' field must be of type 'float'"
        self._t_from_start = value

    @property
    def x(self):
        """Message field 'x'."""
        return self._x

    @x.setter
    def x(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'x' field must be of type 'float'"
        self._x = value

    @property
    def y(self):
        """Message field 'y'."""
        return self._y

    @y.setter
    def y(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'y' field must be of type 'float'"
        self._y = value

    @property
    def yaw(self):
        """Message field 'yaw'."""
        return self._yaw

    @yaw.setter
    def yaw(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'yaw' field must be of type 'float'"
        self._yaw = value

    @property
    def vx(self):
        """Message field 'vx'."""
        return self._vx

    @vx.setter
    def vx(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'vx' field must be of type 'float'"
        self._vx = value

    @property
    def vy(self):
        """Message field 'vy'."""
        return self._vy

    @vy.setter
    def vy(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'vy' field must be of type 'float'"
        self._vy = value

    @property
    def vyaw(self):
        """Message field 'vyaw'."""
        return self._vyaw

    @vyaw.setter
    def vyaw(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'vyaw' field must be of type 'float'"
        self._vyaw = value
