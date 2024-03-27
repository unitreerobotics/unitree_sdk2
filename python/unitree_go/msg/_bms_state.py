# generated from rosidl_generator_py/resource/_idl.py.em
# with input from unitree_go:msg/BmsState.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'bq_ntc'
# Member 'mcu_ntc'
# Member 'cell_vol'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_BmsState(type):
    """Metaclass of message 'BmsState'."""

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
                'unitree_go.msg.BmsState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__bms_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__bms_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__bms_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__bms_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__bms_state

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class BmsState(metaclass=Metaclass_BmsState):
    """Message class 'BmsState'."""

    __slots__ = [
        '_version_high',
        '_version_low',
        '_status',
        '_soc',
        '_current',
        '_cycle',
        '_bq_ntc',
        '_mcu_ntc',
        '_cell_vol',
    ]

    _fields_and_field_types = {
        'version_high': 'uint8',
        'version_low': 'uint8',
        'status': 'uint8',
        'soc': 'uint8',
        'current': 'int32',
        'cycle': 'uint16',
        'bq_ntc': 'int8[2]',
        'mcu_ntc': 'int8[2]',
        'cell_vol': 'uint16[15]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint8'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('uint16'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int8'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('int8'), 2),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('uint16'), 15),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.version_high = kwargs.get('version_high', int())
        self.version_low = kwargs.get('version_low', int())
        self.status = kwargs.get('status', int())
        self.soc = kwargs.get('soc', int())
        self.current = kwargs.get('current', int())
        self.cycle = kwargs.get('cycle', int())
        if 'bq_ntc' not in kwargs:
            self.bq_ntc = numpy.zeros(2, dtype=numpy.int8)
        else:
            self.bq_ntc = numpy.array(kwargs.get('bq_ntc'), dtype=numpy.int8)
            assert self.bq_ntc.shape == (2, )
        if 'mcu_ntc' not in kwargs:
            self.mcu_ntc = numpy.zeros(2, dtype=numpy.int8)
        else:
            self.mcu_ntc = numpy.array(kwargs.get('mcu_ntc'), dtype=numpy.int8)
            assert self.mcu_ntc.shape == (2, )
        if 'cell_vol' not in kwargs:
            self.cell_vol = numpy.zeros(15, dtype=numpy.uint16)
        else:
            self.cell_vol = numpy.array(kwargs.get('cell_vol'), dtype=numpy.uint16)
            assert self.cell_vol.shape == (15, )

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
        if self.version_high != other.version_high:
            return False
        if self.version_low != other.version_low:
            return False
        if self.status != other.status:
            return False
        if self.soc != other.soc:
            return False
        if self.current != other.current:
            return False
        if self.cycle != other.cycle:
            return False
        if all(self.bq_ntc != other.bq_ntc):
            return False
        if all(self.mcu_ntc != other.mcu_ntc):
            return False
        if all(self.cell_vol != other.cell_vol):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @property
    def version_high(self):
        """Message field 'version_high'."""
        return self._version_high

    @version_high.setter
    def version_high(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'version_high' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'version_high' field must be an unsigned integer in [0, 255]"
        self._version_high = value

    @property
    def version_low(self):
        """Message field 'version_low'."""
        return self._version_low

    @version_low.setter
    def version_low(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'version_low' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'version_low' field must be an unsigned integer in [0, 255]"
        self._version_low = value

    @property
    def status(self):
        """Message field 'status'."""
        return self._status

    @status.setter
    def status(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'status' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'status' field must be an unsigned integer in [0, 255]"
        self._status = value

    @property
    def soc(self):
        """Message field 'soc'."""
        return self._soc

    @soc.setter
    def soc(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'soc' field must be of type 'int'"
            assert value >= 0 and value < 256, \
                "The 'soc' field must be an unsigned integer in [0, 255]"
        self._soc = value

    @property
    def current(self):
        """Message field 'current'."""
        return self._current

    @current.setter
    def current(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'current' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'current' field must be an integer in [-2147483648, 2147483647]"
        self._current = value

    @property
    def cycle(self):
        """Message field 'cycle'."""
        return self._cycle

    @cycle.setter
    def cycle(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'cycle' field must be of type 'int'"
            assert value >= 0 and value < 65536, \
                "The 'cycle' field must be an unsigned integer in [0, 65535]"
        self._cycle = value

    @property
    def bq_ntc(self):
        """Message field 'bq_ntc'."""
        return self._bq_ntc

    @bq_ntc.setter
    def bq_ntc(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.int8, \
                "The 'bq_ntc' numpy.ndarray() must have the dtype of 'numpy.int8'"
            assert value.size == 2, \
                "The 'bq_ntc' numpy.ndarray() must have a size of 2"
            self._bq_ntc = value
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
                 all(val >= -128 and val < 128 for val in value)), \
                "The 'bq_ntc' field must be a set or sequence with length 2 and each value of type 'int' and each integer in [-128, 127]"
        self._bq_ntc = numpy.array(value, dtype=numpy.int8)

    @property
    def mcu_ntc(self):
        """Message field 'mcu_ntc'."""
        return self._mcu_ntc

    @mcu_ntc.setter
    def mcu_ntc(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.int8, \
                "The 'mcu_ntc' numpy.ndarray() must have the dtype of 'numpy.int8'"
            assert value.size == 2, \
                "The 'mcu_ntc' numpy.ndarray() must have a size of 2"
            self._mcu_ntc = value
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
                 all(val >= -128 and val < 128 for val in value)), \
                "The 'mcu_ntc' field must be a set or sequence with length 2 and each value of type 'int' and each integer in [-128, 127]"
        self._mcu_ntc = numpy.array(value, dtype=numpy.int8)

    @property
    def cell_vol(self):
        """Message field 'cell_vol'."""
        return self._cell_vol

    @cell_vol.setter
    def cell_vol(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.uint16, \
                "The 'cell_vol' numpy.ndarray() must have the dtype of 'numpy.uint16'"
            assert value.size == 15, \
                "The 'cell_vol' numpy.ndarray() must have a size of 15"
            self._cell_vol = value
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
                 len(value) == 15 and
                 all(isinstance(v, int) for v in value) and
                 all(val >= 0 and val < 65536 for val in value)), \
                "The 'cell_vol' field must be a set or sequence with length 15 and each value of type 'int' and each unsigned integer in [0, 65535]"
        self._cell_vol = numpy.array(value, dtype=numpy.uint16)
