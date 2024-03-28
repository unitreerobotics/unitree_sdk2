// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from unitree_go:msg/MotorState.idl
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
#include "unitree_go/msg/detail/motor_state__struct.h"
#include "unitree_go/msg/detail/motor_state__functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool unitree_go__msg__motor_state__convert_from_py(PyObject * _pymsg, void * _ros_message)
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
    assert(strncmp("unitree_go.msg._motor_state.MotorState", full_classname_dest, 38) == 0);
  }
  unitree_go__msg__MotorState * ros_message = _ros_message;
  {  // mode
    PyObject * field = PyObject_GetAttrString(_pymsg, "mode");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->mode = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // q
    PyObject * field = PyObject_GetAttrString(_pymsg, "q");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->q = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // dq
    PyObject * field = PyObject_GetAttrString(_pymsg, "dq");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->dq = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ddq
    PyObject * field = PyObject_GetAttrString(_pymsg, "ddq");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ddq = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // tau_est
    PyObject * field = PyObject_GetAttrString(_pymsg, "tau_est");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->tau_est = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // q_raw
    PyObject * field = PyObject_GetAttrString(_pymsg, "q_raw");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->q_raw = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // dq_raw
    PyObject * field = PyObject_GetAttrString(_pymsg, "dq_raw");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->dq_raw = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ddq_raw
    PyObject * field = PyObject_GetAttrString(_pymsg, "ddq_raw");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ddq_raw = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // temperature
    PyObject * field = PyObject_GetAttrString(_pymsg, "temperature");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->temperature = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // lost
    PyObject * field = PyObject_GetAttrString(_pymsg, "lost");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->lost = PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // reserve
    PyObject * field = PyObject_GetAttrString(_pymsg, "reserve");
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
      assert(PyArray_TYPE(seq_field) == NPY_UINT32);
      Py_ssize_t size = 2;
      uint32_t * dest = ros_message->reserve;
      for (Py_ssize_t i = 0; i < size; ++i) {
        uint32_t tmp = *(npy_uint32 *)PyArray_GETPTR1(seq_field, i);
        memcpy(&dest[i], &tmp, sizeof(uint32_t));
      }
      Py_DECREF(seq_field);
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * unitree_go__msg__motor_state__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of MotorState */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("unitree_go.msg._motor_state");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "MotorState");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  unitree_go__msg__MotorState * ros_message = (unitree_go__msg__MotorState *)raw_ros_message;
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
  {  // q
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->q);
    {
      int rc = PyObject_SetAttrString(_pymessage, "q", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // dq
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->dq);
    {
      int rc = PyObject_SetAttrString(_pymessage, "dq", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ddq
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ddq);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ddq", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // tau_est
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->tau_est);
    {
      int rc = PyObject_SetAttrString(_pymessage, "tau_est", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // q_raw
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->q_raw);
    {
      int rc = PyObject_SetAttrString(_pymessage, "q_raw", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // dq_raw
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->dq_raw);
    {
      int rc = PyObject_SetAttrString(_pymessage, "dq_raw", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ddq_raw
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ddq_raw);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ddq_raw", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // temperature
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->temperature);
    {
      int rc = PyObject_SetAttrString(_pymessage, "temperature", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // lost
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->lost);
    {
      int rc = PyObject_SetAttrString(_pymessage, "lost", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // reserve
    PyObject * field = NULL;
    field = PyObject_GetAttrString(_pymessage, "reserve");
    if (!field) {
      return NULL;
    }
    assert(field->ob_type != NULL);
    assert(field->ob_type->tp_name != NULL);
    assert(strcmp(field->ob_type->tp_name, "numpy.ndarray") == 0);
    PyArrayObject * seq_field = (PyArrayObject *)field;
    assert(PyArray_NDIM(seq_field) == 1);
    assert(PyArray_TYPE(seq_field) == NPY_UINT32);
    assert(sizeof(npy_uint32) == sizeof(uint32_t));
    npy_uint32 * dst = (npy_uint32 *)PyArray_GETPTR1(seq_field, 0);
    uint32_t * src = &(ros_message->reserve[0]);
    memcpy(dst, src, 2 * sizeof(uint32_t));
    Py_DECREF(field);
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
