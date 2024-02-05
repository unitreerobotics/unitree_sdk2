// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from unitree_go:msg/UwbState.idl
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
#include "unitree_go/msg/detail/uwb_state__struct.h"
#include "unitree_go/msg/detail/uwb_state__functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool unitree_go__msg__uwb_state__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[35];
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
    assert(strncmp("unitree_go.msg._uwb_state.UwbState", full_classname_dest, 34) == 0);
  }
  unitree_go__msg__UwbState * ros_message = _ros_message;
  {  // version
    PyObject * field = PyObject_GetAttrString(_pymsg, "version");
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
      assert(PyArray_TYPE(seq_field) == NPY_UINT8);
      Py_ssize_t size = 2;
      uint8_t * dest = ros_message->version;
      for (Py_ssize_t i = 0; i < size; ++i) {
        uint8_t tmp = *(npy_uint8 *)PyArray_GETPTR1(seq_field, i);
        memcpy(&dest[i], &tmp, sizeof(uint8_t));
      }
      Py_DECREF(seq_field);
    }
    Py_DECREF(field);
  }
  {  // channel
    PyObject * field = PyObject_GetAttrString(_pymsg, "channel");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->channel = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // joy_mode
    PyObject * field = PyObject_GetAttrString(_pymsg, "joy_mode");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->joy_mode = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // orientation_est
    PyObject * field = PyObject_GetAttrString(_pymsg, "orientation_est");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->orientation_est = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // pitch_est
    PyObject * field = PyObject_GetAttrString(_pymsg, "pitch_est");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->pitch_est = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // distance_est
    PyObject * field = PyObject_GetAttrString(_pymsg, "distance_est");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->distance_est = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // yaw_est
    PyObject * field = PyObject_GetAttrString(_pymsg, "yaw_est");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->yaw_est = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // tag_roll
    PyObject * field = PyObject_GetAttrString(_pymsg, "tag_roll");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->tag_roll = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // tag_pitch
    PyObject * field = PyObject_GetAttrString(_pymsg, "tag_pitch");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->tag_pitch = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // tag_yaw
    PyObject * field = PyObject_GetAttrString(_pymsg, "tag_yaw");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->tag_yaw = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // base_roll
    PyObject * field = PyObject_GetAttrString(_pymsg, "base_roll");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->base_roll = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // base_pitch
    PyObject * field = PyObject_GetAttrString(_pymsg, "base_pitch");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->base_pitch = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // base_yaw
    PyObject * field = PyObject_GetAttrString(_pymsg, "base_yaw");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->base_yaw = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // joystick
    PyObject * field = PyObject_GetAttrString(_pymsg, "joystick");
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
      Py_ssize_t size = 2;
      float * dest = ros_message->joystick;
      for (Py_ssize_t i = 0; i < size; ++i) {
        float tmp = *(npy_float32 *)PyArray_GETPTR1(seq_field, i);
        memcpy(&dest[i], &tmp, sizeof(float));
      }
      Py_DECREF(seq_field);
    }
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
  {  // buttons
    PyObject * field = PyObject_GetAttrString(_pymsg, "buttons");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->buttons = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // enabled_from_app
    PyObject * field = PyObject_GetAttrString(_pymsg, "enabled_from_app");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->enabled_from_app = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * unitree_go__msg__uwb_state__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of UwbState */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("unitree_go.msg._uwb_state");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "UwbState");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  unitree_go__msg__UwbState * ros_message = (unitree_go__msg__UwbState *)raw_ros_message;
  {  // version
    PyObject * field = NULL;
    field = PyObject_GetAttrString(_pymessage, "version");
    if (!field) {
      return NULL;
    }
    assert(field->ob_type != NULL);
    assert(field->ob_type->tp_name != NULL);
    assert(strcmp(field->ob_type->tp_name, "numpy.ndarray") == 0);
    PyArrayObject * seq_field = (PyArrayObject *)field;
    assert(PyArray_NDIM(seq_field) == 1);
    assert(PyArray_TYPE(seq_field) == NPY_UINT8);
    assert(sizeof(npy_uint8) == sizeof(uint8_t));
    npy_uint8 * dst = (npy_uint8 *)PyArray_GETPTR1(seq_field, 0);
    uint8_t * src = &(ros_message->version[0]);
    memcpy(dst, src, 2 * sizeof(uint8_t));
    Py_DECREF(field);
  }
  {  // channel
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->channel);
    {
      int rc = PyObject_SetAttrString(_pymessage, "channel", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // joy_mode
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->joy_mode);
    {
      int rc = PyObject_SetAttrString(_pymessage, "joy_mode", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // orientation_est
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->orientation_est);
    {
      int rc = PyObject_SetAttrString(_pymessage, "orientation_est", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pitch_est
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->pitch_est);
    {
      int rc = PyObject_SetAttrString(_pymessage, "pitch_est", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // distance_est
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->distance_est);
    {
      int rc = PyObject_SetAttrString(_pymessage, "distance_est", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // yaw_est
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->yaw_est);
    {
      int rc = PyObject_SetAttrString(_pymessage, "yaw_est", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // tag_roll
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->tag_roll);
    {
      int rc = PyObject_SetAttrString(_pymessage, "tag_roll", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // tag_pitch
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->tag_pitch);
    {
      int rc = PyObject_SetAttrString(_pymessage, "tag_pitch", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // tag_yaw
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->tag_yaw);
    {
      int rc = PyObject_SetAttrString(_pymessage, "tag_yaw", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // base_roll
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->base_roll);
    {
      int rc = PyObject_SetAttrString(_pymessage, "base_roll", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // base_pitch
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->base_pitch);
    {
      int rc = PyObject_SetAttrString(_pymessage, "base_pitch", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // base_yaw
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->base_yaw);
    {
      int rc = PyObject_SetAttrString(_pymessage, "base_yaw", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // joystick
    PyObject * field = NULL;
    field = PyObject_GetAttrString(_pymessage, "joystick");
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
    float * src = &(ros_message->joystick[0]);
    memcpy(dst, src, 2 * sizeof(float));
    Py_DECREF(field);
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
  {  // buttons
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->buttons);
    {
      int rc = PyObject_SetAttrString(_pymessage, "buttons", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // enabled_from_app
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->enabled_from_app);
    {
      int rc = PyObject_SetAttrString(_pymessage, "enabled_from_app", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
