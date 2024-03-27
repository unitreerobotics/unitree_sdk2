// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from unitree_go:msg/WirelessController.idl
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
#include "unitree_go/msg/detail/wireless_controller__struct.h"
#include "unitree_go/msg/detail/wireless_controller__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool unitree_go__msg__wireless_controller__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[55];
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
    assert(strncmp("unitree_go.msg._wireless_controller.WirelessController", full_classname_dest, 54) == 0);
  }
  unitree_go__msg__WirelessController * ros_message = _ros_message;
  {  // lx
    PyObject * field = PyObject_GetAttrString(_pymsg, "lx");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->lx = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ly
    PyObject * field = PyObject_GetAttrString(_pymsg, "ly");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ly = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // rx
    PyObject * field = PyObject_GetAttrString(_pymsg, "rx");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->rx = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ry
    PyObject * field = PyObject_GetAttrString(_pymsg, "ry");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ry = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // keys
    PyObject * field = PyObject_GetAttrString(_pymsg, "keys");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->keys = (uint16_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * unitree_go__msg__wireless_controller__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of WirelessController */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("unitree_go.msg._wireless_controller");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "WirelessController");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  unitree_go__msg__WirelessController * ros_message = (unitree_go__msg__WirelessController *)raw_ros_message;
  {  // lx
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->lx);
    {
      int rc = PyObject_SetAttrString(_pymessage, "lx", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ly
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ly);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ly", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // rx
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->rx);
    {
      int rc = PyObject_SetAttrString(_pymessage, "rx", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ry
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ry);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ry", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // keys
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->keys);
    {
      int rc = PyObject_SetAttrString(_pymessage, "keys", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
