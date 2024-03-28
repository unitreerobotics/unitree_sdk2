// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from unitree_go:msg/LidarState.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "unitree_go/msg/detail/lidar_state__struct.h"
#include "unitree_go/msg/detail/lidar_state__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool unitree_go__msg__lidar_state__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[39];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("unitree_go.msg._lidar_state.LidarState", full_classname_dest, 38) == 0);
  }
  unitree_go__msg__LidarState * ros_message = _ros_message;
  {  // stamp
    PyObject * field = PyObject_GetAttrString(_pymsg, "stamp");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->stamp = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // firmware_version
    PyObject * field = PyObject_GetAttrString(_pymsg, "firmware_version");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->firmware_version, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // software_version
    PyObject * field = PyObject_GetAttrString(_pymsg, "software_version");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->software_version, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // sdk_version
    PyObject * field = PyObject_GetAttrString(_pymsg, "sdk_version");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->sdk_version, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // sys_rotation_speed
    PyObject * field = PyObject_GetAttrString(_pymsg, "sys_rotation_speed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->sys_rotation_speed = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // com_rotation_speed
    PyObject * field = PyObject_GetAttrString(_pymsg, "com_rotation_speed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->com_rotation_speed = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // error_state
    PyObject * field = PyObject_GetAttrString(_pymsg, "error_state");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->error_state = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // cloud_frequency
    PyObject * field = PyObject_GetAttrString(_pymsg, "cloud_frequency");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->cloud_frequency = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // cloud_packet_loss_rate
    PyObject * field = PyObject_GetAttrString(_pymsg, "cloud_packet_loss_rate");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->cloud_packet_loss_rate = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // cloud_size
    PyObject * field = PyObject_GetAttrString(_pymsg, "cloud_size");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->cloud_size = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // cloud_scan_num
    PyObject * field = PyObject_GetAttrString(_pymsg, "cloud_scan_num");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->cloud_scan_num = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // imu_frequency
    PyObject * field = PyObject_GetAttrString(_pymsg, "imu_frequency");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->imu_frequency = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // imu_packet_loss_rate
    PyObject * field = PyObject_GetAttrString(_pymsg, "imu_packet_loss_rate");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->imu_packet_loss_rate = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // imu_rpy
    PyObject * field = PyObject_GetAttrString(_pymsg, "imu_rpy");
    if (!field) {
      return false;
    }
    {
      // TODO(dirk-thomas) use a better way to check the type before casting
      assert(field->ob_type != NULL);
      assert(field->ob_type->tp_name != NULL);
      assert(strcmp(field->ob_type->tp_name, "numpy.ndarray") == 0);
      PyArrayObject * seq_field = (PyArrayObject *)field;
      Py_INCREF(seq_field);
      assert(PyArray_NDIM(seq_field) == 1);
      assert(PyArray_TYPE(seq_field) == NPY_FLOAT32);
      Py_ssize_t size = 3;
      float * dest = ros_message->imu_rpy;
      for (Py_ssize_t i = 0; i < size; ++i) {
        float tmp = *(npy_float32 *)PyArray_GETPTR1(seq_field, i);
        memcpy(&dest[i], &tmp, sizeof(float));
      }
      Py_DECREF(seq_field);
    }
    Py_DECREF(field);
  }
  {  // serial_recv_stamp
    PyObject * field = PyObject_GetAttrString(_pymsg, "serial_recv_stamp");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->serial_recv_stamp = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // serial_buffer_size
    PyObject * field = PyObject_GetAttrString(_pymsg, "serial_buffer_size");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->serial_buffer_size = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // serial_buffer_read
    PyObject * field = PyObject_GetAttrString(_pymsg, "serial_buffer_read");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->serial_buffer_read = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * unitree_go__msg__lidar_state__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of LidarState */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("unitree_go.msg._lidar_state");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "LidarState");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  unitree_go__msg__LidarState * ros_message = (unitree_go__msg__LidarState *)raw_ros_message;
  {  // stamp
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->stamp);
    {
      int rc = PyObject_SetAttrString(_pymessage, "stamp", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // firmware_version
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->firmware_version.data,
      strlen(ros_message->firmware_version.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "firmware_version", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // software_version
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->software_version.data,
      strlen(ros_message->software_version.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "software_version", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // sdk_version
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->sdk_version.data,
      strlen(ros_message->sdk_version.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "sdk_version", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // sys_rotation_speed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->sys_rotation_speed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "sys_rotation_speed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // com_rotation_speed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->com_rotation_speed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "com_rotation_speed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // error_state
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->error_state);
    {
      int rc = PyObject_SetAttrString(_pymessage, "error_state", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cloud_frequency
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->cloud_frequency);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cloud_frequency", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cloud_packet_loss_rate
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->cloud_packet_loss_rate);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cloud_packet_loss_rate", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cloud_size
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->cloud_size);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cloud_size", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cloud_scan_num
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->cloud_scan_num);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cloud_scan_num", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // imu_frequency
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->imu_frequency);
    {
      int rc = PyObject_SetAttrString(_pymessage, "imu_frequency", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // imu_packet_loss_rate
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->imu_packet_loss_rate);
    {
      int rc = PyObject_SetAttrString(_pymessage, "imu_packet_loss_rate", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // imu_rpy
    PyObject * field = NULL;
    field = PyObject_GetAttrString(_pymessage, "imu_rpy");
    if (!field) {
      return NULL;
    }
    assert(field->ob_type != NULL);
    assert(field->ob_type->tp_name != NULL);
    assert(strcmp(field->ob_type->tp_name, "numpy.ndarray") == 0);
    PyArrayObject * seq_field = (PyArrayObject *)field;
    assert(PyArray_NDIM(seq_field) == 1);
    assert(PyArray_TYPE(seq_field) == NPY_FLOAT32);
    assert(sizeof(npy_float32) == sizeof(float));
    npy_float32 * dst = (npy_float32 *)PyArray_GETPTR1(seq_field, 0);
    float * src = &(ros_message->imu_rpy[0]);
    memcpy(dst, src, 3 * sizeof(float));
    Py_DECREF(field);
  }
  {  // serial_recv_stamp
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->serial_recv_stamp);
    {
      int rc = PyObject_SetAttrString(_pymessage, "serial_recv_stamp", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // serial_buffer_size
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->serial_buffer_size);
    {
      int rc = PyObject_SetAttrString(_pymessage, "serial_buffer_size", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // serial_buffer_read
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->serial_buffer_read);
    {
      int rc = PyObject_SetAttrString(_pymessage, "serial_buffer_read", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
