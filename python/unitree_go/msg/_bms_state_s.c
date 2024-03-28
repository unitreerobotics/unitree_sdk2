// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from unitree_go:msg/BmsState.idl
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
#include "unitree_go/msg/detail/bms_state__struct.h"
#include "unitree_go/msg/detail/bms_state__functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool unitree_go__msg__bms_state__convert_from_py(PyObject * _pymsg, void * _ros_message)
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
    assert(strncmp("unitree_go.msg._bms_state.BmsState", full_classname_dest, 34) == 0);
  }
  unitree_go__msg__BmsState * ros_message = _ros_message;
  {  // version_high
    PyObject * field = PyObject_GetAttrString(_pymsg, "version_high");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->version_high = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // version_low
    PyObject * field = PyObject_GetAttrString(_pymsg, "version_low");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->version_low = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // status
    PyObject * field = PyObject_GetAttrString(_pymsg, "status");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->status = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // soc
    PyObject * field = PyObject_GetAttrString(_pymsg, "soc");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->soc = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // current
    PyObject * field = PyObject_GetAttrString(_pymsg, "current");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->current = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // cycle
    PyObject * field = PyObject_GetAttrString(_pymsg, "cycle");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->cycle = (uint16_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // bq_ntc
    PyObject * field = PyObject_GetAttrString(_pymsg, "bq_ntc");
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
      assert(PyArray_TYPE(seq_field) == NPY_INT8);
      Py_ssize_t size = 2;
      int8_t * dest = ros_message->bq_ntc;
      for (Py_ssize_t i = 0; i < size; ++i) {
        int8_t tmp = *(npy_int8 *)PyArray_GETPTR1(seq_field, i);
        memcpy(&dest[i], &tmp, sizeof(int8_t));
      }
      Py_DECREF(seq_field);
    }
    Py_DECREF(field);
  }
  {  // mcu_ntc
    PyObject * field = PyObject_GetAttrString(_pymsg, "mcu_ntc");
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
      assert(PyArray_TYPE(seq_field) == NPY_INT8);
      Py_ssize_t size = 2;
      int8_t * dest = ros_message->mcu_ntc;
      for (Py_ssize_t i = 0; i < size; ++i) {
        int8_t tmp = *(npy_int8 *)PyArray_GETPTR1(seq_field, i);
        memcpy(&dest[i], &tmp, sizeof(int8_t));
      }
      Py_DECREF(seq_field);
    }
    Py_DECREF(field);
  }
  {  // cell_vol
    PyObject * field = PyObject_GetAttrString(_pymsg, "cell_vol");
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
      assert(PyArray_TYPE(seq_field) == NPY_UINT16);
      Py_ssize_t size = 15;
      uint16_t * dest = ros_message->cell_vol;
      for (Py_ssize_t i = 0; i < size; ++i) {
        uint16_t tmp = *(npy_uint16 *)PyArray_GETPTR1(seq_field, i);
        memcpy(&dest[i], &tmp, sizeof(uint16_t));
      }
      Py_DECREF(seq_field);
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * unitree_go__msg__bms_state__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of BmsState */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("unitree_go.msg._bms_state");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "BmsState");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  unitree_go__msg__BmsState * ros_message = (unitree_go__msg__BmsState *)raw_ros_message;
  {  // version_high
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->version_high);
    {
      int rc = PyObject_SetAttrString(_pymessage, "version_high", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // version_low
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->version_low);
    {
      int rc = PyObject_SetAttrString(_pymessage, "version_low", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // status
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->status);
    {
      int rc = PyObject_SetAttrString(_pymessage, "status", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // soc
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->soc);
    {
      int rc = PyObject_SetAttrString(_pymessage, "soc", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // current
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->current);
    {
      int rc = PyObject_SetAttrString(_pymessage, "current", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cycle
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->cycle);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cycle", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // bq_ntc
    PyObject * field = NULL;
    field = PyObject_GetAttrString(_pymessage, "bq_ntc");
    if (!field) {
      return NULL;
    }
    assert(field->ob_type != NULL);
    assert(field->ob_type->tp_name != NULL);
    assert(strcmp(field->ob_type->tp_name, "numpy.ndarray") == 0);
    PyArrayObject * seq_field = (PyArrayObject *)field;
    assert(PyArray_NDIM(seq_field) == 1);
    assert(PyArray_TYPE(seq_field) == NPY_INT8);
    assert(sizeof(npy_int8) == sizeof(int8_t));
    npy_int8 * dst = (npy_int8 *)PyArray_GETPTR1(seq_field, 0);
    int8_t * src = &(ros_message->bq_ntc[0]);
    memcpy(dst, src, 2 * sizeof(int8_t));
    Py_DECREF(field);
  }
  {  // mcu_ntc
    PyObject * field = NULL;
    field = PyObject_GetAttrString(_pymessage, "mcu_ntc");
    if (!field) {
      return NULL;
    }
    assert(field->ob_type != NULL);
    assert(field->ob_type->tp_name != NULL);
    assert(strcmp(field->ob_type->tp_name, "numpy.ndarray") == 0);
    PyArrayObject * seq_field = (PyArrayObject *)field;
    assert(PyArray_NDIM(seq_field) == 1);
    assert(PyArray_TYPE(seq_field) == NPY_INT8);
    assert(sizeof(npy_int8) == sizeof(int8_t));
    npy_int8 * dst = (npy_int8 *)PyArray_GETPTR1(seq_field, 0);
    int8_t * src = &(ros_message->mcu_ntc[0]);
    memcpy(dst, src, 2 * sizeof(int8_t));
    Py_DECREF(field);
  }
  {  // cell_vol
    PyObject * field = NULL;
    field = PyObject_GetAttrString(_pymessage, "cell_vol");
    if (!field) {
      return NULL;
    }
    assert(field->ob_type != NULL);
    assert(field->ob_type->tp_name != NULL);
    assert(strcmp(field->ob_type->tp_name, "numpy.ndarray") == 0);
    PyArrayObject * seq_field = (PyArrayObject *)field;
    assert(PyArray_NDIM(seq_field) == 1);
    assert(PyArray_TYPE(seq_field) == NPY_UINT16);
    assert(sizeof(npy_uint16) == sizeof(uint16_t));
    npy_uint16 * dst = (npy_uint16 *)PyArray_GETPTR1(seq_field, 0);
    uint16_t * src = &(ros_message->cell_vol[0]);
    memcpy(dst, src, 15 * sizeof(uint16_t));
    Py_DECREF(field);
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
