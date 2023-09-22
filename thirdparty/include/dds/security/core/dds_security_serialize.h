/*
 * Copyright(c) 2006 to 2019 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */

#ifndef DDS_SECURITY_CDR_SER_H
#define DDS_SECURITY_CDR_SER_H

#include "dds/export.h"
#include "dds/security/dds_security_api.h"
#include "dds/security/core/dds_security_types.h"
#include "stddef.h"

#if defined (__cplusplus)
extern "C" {
#endif



typedef struct DDS_Security_Serializer *DDS_Security_Serializer;
typedef struct DDS_Security_Deserializer *DDS_Security_Deserializer;


DDS_EXPORT DDS_Security_Serializer
DDS_Security_Serializer_new(
     size_t size,
     size_t increment);

DDS_EXPORT void
DDS_Security_Serializer_free(
     DDS_Security_Serializer serializer);

DDS_EXPORT void
DDS_Security_Serializer_buffer(
     DDS_Security_Serializer ser,
     unsigned char **buffer,
     size_t *size);

DDS_EXPORT void
DDS_Security_Serialize_PropertySeq(
     DDS_Security_Serializer serializer,
     const DDS_Security_PropertySeq *seq);

DDS_EXPORT void
DDS_Security_Serialize_BinaryPropertyArray(
     DDS_Security_Serializer serializer,
     const DDS_Security_BinaryProperty_t **properties,
     const uint32_t length);

DDS_EXPORT void
DDS_Security_Serialize_BinaryPropertySeq(
     DDS_Security_Serializer serializer,
     const DDS_Security_BinaryPropertySeq *seq);

DDS_EXPORT void
DDS_Security_Serialize_DataHolderSeq(
     DDS_Security_Serializer serializer,
     const DDS_Security_DataHolderSeq *seq);

DDS_EXPORT void
DDS_Security_Serialize_ParticipantBuiltinTopicData(
     DDS_Security_Serializer ser,
     DDS_Security_ParticipantBuiltinTopicData *pdata);

DDS_EXPORT void
DDS_Security_Serialize_KeyMaterial_AES_GCM_GMAC(
     DDS_Security_Serializer ser,
     const DDS_Security_KeyMaterial_AES_GCM_GMAC *data);

DDS_EXPORT DDS_Security_Deserializer
DDS_Security_Deserializer_new(
     const unsigned char *data,
     size_t size);

DDS_EXPORT void
DDS_Security_Deserializer_free(
     DDS_Security_Deserializer deserializer);

DDS_EXPORT int
DDS_Security_Deserialize_ParticipantBuiltinTopicData(
     DDS_Security_Deserializer deserializer,
     DDS_Security_ParticipantBuiltinTopicData *pdata,
     DDS_Security_SecurityException *ex);

DDS_EXPORT void
DDS_Security_BuiltinTopicKeyBE(
     DDS_Security_BuiltinTopicKey_t dst,
     const  DDS_Security_BuiltinTopicKey_t src);

DDS_EXPORT int
DDS_Security_Deserialize_KeyMaterial_AES_GCM_GMAC(
     DDS_Security_Deserializer dser,
     DDS_Security_KeyMaterial_AES_GCM_GMAC *data);

#if defined (__cplusplus)
}
#endif

#endif /* DDS_SECURITY_CDR_SER_H */
