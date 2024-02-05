// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from unitree_go:msg/SportModeCmd.idl
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
#include "unitree_go/msg/detail/sport_mode_cmd__struct.h"
#include "unitree_go/msg/detail/sport_mode_cmd__functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"

// Nested array functions includes
#include "unitree_go/msg/detail/path_point__functions.h"
// end nested array functions include
bool unitree_go__msg__bms_cmd__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * unitree_go__msg__bms_cmd__convert_to_py(void * raw_ros_message);
bool unitree_go__msg__path_point__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * unitree_go__msg__path_point__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool unitree_go__msg__sport_mode_cmd__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[44];
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
    assert(strncmp("unitree_go.msg._sport_mode_cmd.SportModeCmd", full_classname_dest, 43) == 0);
  }
  unitree_go__msg__SportModeCmd * ros_message = _ros_message;
  {  // mode
    PyObject * field = PyObject_GetAttrString(_pymsg, "mode");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->mode = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // gait_type
    PyObject * field = PyObject_GetAttrString(_pymsg, "gait_type");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->gait_type = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // speed_level
    PyObject * field = PyObject_GetAttrString(_pymsg, "speed_level");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->speed_level = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // foot_raise_height
    PyObject * field = PyObject_GetAttrString(_pymsg, "foot_raise_height");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->foot_raise_height = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // body_height
    PyObject * field = PyObject_GetAttrString(_pymsg, "body_height");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->body_height = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // position
    PyObject * field = PyObject_GetAttrString(_pymsg, "position");
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
      float * dest = ros_message->position;
      for (Py_ssize_t i = 0; i < size; ++i) {
        float tmp = *(npy_float32 *)PyArray_GETPTR1(seq_field, i);
        memcpy(&dest[i], &tmp, sizeof(float));
      }
      Py_DECREF(seq_field);
    }
    Py_DECREF(field);
  }
  {  // euler
    PyObject * field = PyObject_GetAttrString(_pymsg, "euler");
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
      float * dest = ros_message->euler;
      for (Py_ssize_t i = 0; i < size; ++i) {
        float tmp = *(npy_float32 *)PyArray_GETPTR1(seq_field, i);
        memcpy(&dest[i], &tmp, sizeof(float));
      }
      Py_DECREF(seq_field);
    }
    Py_DECREF(field);
  }
  {  // velocity
    PyObject * field = PyObject_GetAttrString(_pymsg, "velocity");
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
      float * dest = ros_message->velocity;
      for (Py_ssize_t i = 0; i < size; ++i) {
        float tmp = *(npy_float32 *)PyArray_GETPTR1(seq_field, i);
        memcpy(&dest[i], &tmp, sizeof(float));
      }
      Py_DECREF(seq_field);
    }
    Py_DECREF(field);
  }
  {  // yaw_speed
    PyObject * field = PyObject_GetAttrString(_pymsg, "yaw_speed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->yaw_speed = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // bms_cmd
    PyObject * field = PyObject_GetAttrString(_pymsg, "bms_cmd");
    if (!field) {
      return false;
    }
    if (!unitree_go__msg__bms_cmd__convert_from_py(field, &ros_message->bms_cmd)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // path_point
    PyObject * field = PyObject_GetAttrString(_pymsg, "path_point");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'path_point'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = 30;
    unitree_go__msg__PathPoint * dest = ros_message->path_point;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!unitree_go__msg__path_point__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * unitree_go__msg__sport_mode_cmd__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of SportModeCmd */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("unitree_go.msg._sport_mode_cmd");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "SportModeCmd");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  unitree_go__msg__SportModeCmd * ros_message = (unitree_go__msg__SportModeCmd *)raw_ros_message;
  {  // mode
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->mode);
    {
      int rc = PyObject_SetAttrString(_pymessage, "mode", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // gait_type
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->gait_type);
    {
      int rc = PyObject_SetAttrString(_pymessage, "gait_type", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // speed_level
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->speed_level);
    {
      int rc = PyObject_SetAttrString(_pymessage, "speed_level", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // foot_raise_height
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->foot_raise_height);
    {
      int rc = PyObject_SetAttrString(_pymessage, "foot_raise_height", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // body_height
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->body_height);
    {
      int rc = PyObject_SetAttrString(_pymessage, "body_height", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // position
    PyObject * field = NULL;
    field = PyObject_GetAttrString(_pymessage, "position");
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
    float * src = &(ros_message->position[0]);
    memcpy(dst, src, 2 * sizeof(float));
    Py_DECREF(field);
  }
  {  // euler
    PyObject * field = NULL;
    field = PyObject_GetAttrString(_pymessage, "euler");
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
    float * src = &(ros_message->euler[0]);
    memcpy(dst, src, 3 * sizeof(float));
    Py_DECREF(field);
  }
  {  // velocity
    PyObject * field = NULL;
    field = PyObject_GetAttrString(_pymessage, "velocity");
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
    float * src = &(ros_message->velocity[0]);
    memcpy(dst, src, 2 * sizeof(float));
    Py_DECREF(field);
  }
  {  // yaw_speed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->yaw_speed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "yaw_speed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // bms_cmd
    PyObject * field = NULL;
    field = unitree_go__msg__bms_cmd__convert_to_py(&ros_message->bms_cmd);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "bms_cmd", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // path_point
    PyObject * field = NULL;
    size_t size = 30;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    unitree_go__msg__PathPoint * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->path_point[i]);
      PyObject * pyitem = unitree_go__msg__path_point__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "path_point", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
