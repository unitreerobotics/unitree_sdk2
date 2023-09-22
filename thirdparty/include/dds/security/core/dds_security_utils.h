/*
 * Copyright(c) 2006 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */

#ifndef DSCMN_SECURITY_UTILS_H_
#define DSCMN_SECURITY_UTILS_H_

#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>

#include "dds/export.h"
#include "dds/ddsrt/strtol.h"
#include "dds/ddsrt/time.h"
#include "dds/security/core/dds_security_types.h"
#include "dds/security/dds_security_api.h"

typedef DDS_Security_long_long DDS_Security_Handle;
typedef DDS_Security_LongLongSeq DDS_Security_HandleSeq;

#define DDS_SECURITY_SEQUENCE_INIT   {0, 0, NULL}
#define DDS_SECURITY_TOKEN_INIT      {NULL, DDS_SECURITY_SEQUENCE_INIT, DDS_SECURITY_SEQUENCE_INIT}
#define DDS_SECURITY_EXCEPTION_INIT  {NULL, 0, 0}

typedef enum {
    DDS_SECURITY_CONFIG_ITEM_PREFIX_UNKNOWN,
    DDS_SECURITY_CONFIG_ITEM_PREFIX_FILE,
    DDS_SECURITY_CONFIG_ITEM_PREFIX_DATA,
    DDS_SECURITY_CONFIG_ITEM_PREFIX_PKCS11
} DDS_Security_config_item_prefix_t;


DDS_EXPORT DDS_Security_BinaryProperty_t *
DDS_Security_BinaryProperty_alloc(
        void);

DDS_EXPORT void
DDS_Security_BinaryProperty_deinit(
        DDS_Security_BinaryProperty_t *p);

DDS_EXPORT void
DDS_Security_BinaryProperty_free(
        DDS_Security_BinaryProperty_t *p);

DDS_EXPORT void
DDS_Security_BinaryProperty_copy(
         DDS_Security_BinaryProperty_t *dst,
         const DDS_Security_BinaryProperty_t *src);

DDS_EXPORT bool
DDS_Security_BinaryProperty_equal(
         const DDS_Security_BinaryProperty_t *pa,
         const DDS_Security_BinaryProperty_t *pb);

DDS_EXPORT void
DDS_Security_BinaryProperty_set_by_value(
         DDS_Security_BinaryProperty_t *bp,
         const char *name,
         const unsigned char *data,
         uint32_t length);

DDS_EXPORT void
DDS_Security_BinaryProperty_set_by_string(
         DDS_Security_BinaryProperty_t *bp,
         const char *name,
         const char *data);

DDS_EXPORT void
DDS_Security_BinaryProperty_set_by_ref(
         DDS_Security_BinaryProperty_t *bp,
         const char *name,
         unsigned char *data,
         uint32_t length);

DDS_EXPORT DDS_Security_BinaryPropertySeq *
DDS_Security_BinaryPropertySeq_alloc(
        void);

DDS_EXPORT DDS_Security_BinaryProperty_t *
DDS_Security_BinaryPropertySeq_allocbuf(
         DDS_Security_unsigned_long len);

DDS_EXPORT void
DDS_Security_BinaryPropertySeq_deinit(
         DDS_Security_BinaryPropertySeq *seq);

DDS_EXPORT void
DDS_Security_BinaryPropertySeq_free(
         DDS_Security_BinaryPropertySeq *seq);

DDS_EXPORT DDS_Security_Property_t *
DDS_Security_Property_alloc(
        void);

DDS_EXPORT void
DDS_Security_Property_free(
        DDS_Security_Property_t *p);

DDS_EXPORT void
DDS_Security_Property_deinit(
        DDS_Security_Property_t *p);

DDS_EXPORT void
DDS_Security_Property_copy(
         DDS_Security_Property_t *dst,
         const DDS_Security_Property_t *src);

DDS_EXPORT bool
DDS_Security_Property_equal(
         const DDS_Security_Property_t *pa,
         const DDS_Security_Property_t *pb);

DDS_EXPORT char *
DDS_Security_Property_get_value(
         const DDS_Security_PropertySeq *properties,
         const char *name);

DDS_EXPORT DDS_Security_PropertySeq *
DDS_Security_PropertySeq_alloc(
        void);

DDS_EXPORT DDS_Security_Property_t *
DDS_Security_PropertySeq_allocbuf(
         DDS_Security_unsigned_long len);

DDS_EXPORT void
DDS_Security_PropertySeq_freebuf(
         DDS_Security_PropertySeq *seq);

DDS_EXPORT void
DDS_Security_PropertySeq_free(
         DDS_Security_PropertySeq *seq);

DDS_EXPORT void
DDS_Security_PropertySeq_deinit(
        DDS_Security_PropertySeq *seq);

DDS_EXPORT const DDS_Security_Property_t *
DDS_Security_PropertySeq_find_property (
         const DDS_Security_PropertySeq *property_seq,
         const char *name );

DDS_EXPORT DDS_Security_DataHolder *
DDS_Security_DataHolder_alloc(
        void);

DDS_EXPORT void
DDS_Security_DataHolder_free(
        DDS_Security_DataHolder *holder);

DDS_EXPORT void
DDS_Security_DataHolder_deinit(
        DDS_Security_DataHolder *holder);

DDS_EXPORT void
DDS_Security_DataHolder_copy(
         DDS_Security_DataHolder *dst,
         const DDS_Security_DataHolder *src);

DDS_EXPORT bool
DDS_Security_DataHolder_equal(
         const DDS_Security_DataHolder *psa,
         const DDS_Security_DataHolder *psb);

DDS_EXPORT const DDS_Security_Property_t *
DDS_Security_DataHolder_find_property(
         const DDS_Security_DataHolder *holder,
         const char *name);

DDS_EXPORT const DDS_Security_BinaryProperty_t *
DDS_Security_DataHolder_find_binary_property(
         const DDS_Security_DataHolder *holder,
         const char *name);

DDS_EXPORT DDS_Security_DataHolderSeq *
DDS_Security_DataHolderSeq_alloc(
        void);

DDS_EXPORT DDS_Security_DataHolder *
DDS_Security_DataHolderSeq_allocbuf(
         DDS_Security_unsigned_long len);

DDS_EXPORT void
DDS_Security_DataHolderSeq_freebuf(
         DDS_Security_DataHolderSeq *seq);

DDS_EXPORT void
DDS_Security_DataHolderSeq_free(
         DDS_Security_DataHolderSeq *seq);

DDS_EXPORT void
DDS_Security_DataHolderSeq_deinit(
         DDS_Security_DataHolderSeq *seq);

DDS_EXPORT void
DDS_Security_DataHolderSeq_copy(
         DDS_Security_DataHolderSeq *dst,
         const DDS_Security_DataHolderSeq *src);

DDS_EXPORT DDS_Security_ParticipantBuiltinTopicData *
DDS_Security_ParticipantBuiltinTopicData_alloc(
        void);

DDS_EXPORT void
DDS_Security_ParticipantBuiltinTopicData_free(
         DDS_Security_ParticipantBuiltinTopicData *data);

DDS_EXPORT void
DDS_Security_ParticipantBuiltinTopicData_deinit(
         DDS_Security_ParticipantBuiltinTopicData *data);

DDS_EXPORT DDS_Security_OctetSeq *
DDS_Security_OctetSeq_alloc(
        void);

DDS_EXPORT DDS_Security_octet *
DDS_Security_OctetSeq_allocbuf(
         DDS_Security_unsigned_long len);

DDS_EXPORT void
DDS_Security_OctetSeq_freebuf(
         DDS_Security_OctetSeq *seq);

DDS_EXPORT void
DDS_Security_OctetSeq_free(
         DDS_Security_OctetSeq *seq);

DDS_EXPORT void
DDS_Security_OctetSeq_deinit(
         DDS_Security_OctetSeq *seq);

DDS_EXPORT void
DDS_Security_OctetSeq_copy(
         DDS_Security_OctetSeq *dst,
         const DDS_Security_OctetSeq *src);

DDS_EXPORT DDS_Security_HandleSeq *
DDS_Security_HandleSeq_alloc(
        void);

DDS_EXPORT DDS_Security_long_long *
DDS_Security_HandleSeq_allocbuf(
         DDS_Security_unsigned_long length);

DDS_EXPORT void
DDS_Security_HandleSeq_freebuf(
         DDS_Security_HandleSeq *seq);

DDS_EXPORT void
DDS_Security_HandleSeq_free(
         DDS_Security_HandleSeq *seq);

DDS_EXPORT void
DDS_Security_HandleSeq_deinit(
         DDS_Security_HandleSeq *seq);

DDS_EXPORT void
DDS_Security_Exception_vset(
         DDS_Security_SecurityException *ex,
         const char *context,
         int code,
         int minor_code,
         const char *fmt,
         va_list ap);

DDS_EXPORT void
DDS_Security_Exception_set(
         DDS_Security_SecurityException *ex,
         const char *context,
         int code,
         int minor_code,
         const char *fmt,
         ...)
  ddsrt_attribute_format_printf(5, 6)
  ddsrt_nonnull_all;

DDS_EXPORT void
DDS_Security_Exception_reset(
         DDS_Security_SecurityException *ex);

DDS_EXPORT void
DDS_Security_Exception_clean(
         DDS_Security_SecurityException *ex);

DDS_EXPORT void
DDS_Security_PropertyQosPolicy_deinit(
         DDS_Security_PropertyQosPolicy *policy);

DDS_EXPORT void
DDS_Security_PropertyQosPolicy_free(
         DDS_Security_PropertyQosPolicy *policy);

DDS_EXPORT void
DDS_Security_set_token_nil(
         DDS_Security_DataHolder *token);

DDS_EXPORT void
DDS_Security_KeyMaterial_AES_GCM_GMAC_deinit(
         DDS_Security_KeyMaterial_AES_GCM_GMAC *key_material);

DDS_EXPORT DDS_Security_CryptoTransformKind_Enum
DDS_Security_basicprotectionkind2transformationkind(
         const DDS_Security_PropertySeq *properties,
         DDS_Security_BasicProtectionKind protection);

DDS_EXPORT DDS_Security_CryptoTransformKind_Enum
DDS_Security_protectionkind2transformationkind(
         const DDS_Security_PropertySeq *properties,
         DDS_Security_ProtectionKind protection);

DDS_EXPORT DDS_Security_config_item_prefix_t
DDS_Security_get_conf_item_type(
         const char *str,
         char **data);

DDS_EXPORT char *
DDS_Security_normalize_file(
    const char *filepath);

DDS_EXPORT dds_time_t
DDS_Security_parse_xml_date(
        char *buf);


#define DDS_Security_ParticipantCryptoTokenSeq_alloc() \
                    DDS_Security_DataHolderSeq_alloc()
#define DDS_Security_ParticipantCryptoTokenSeq_freebuf(s) \
                    DDS_Security_DataHolderSeq_freebuf(s)
#define DDS_Security_ParticipantCryptoTokenSeq_free(s) \
                    DDS_Security_DataHolderSeq_free(s)
#define DDS_Security_ParticipantCryptoTokenSeq_deinit(s) \
                    DDS_Security_DataHolderSeq_deinit(s)
#define DDS_Security_ParticipantCryptoTokenSeq_copy(d,s) \
                    DDS_Security_DataHolderSeq_copy((d), (s))


#define DDS_Security_ParticipantCryptoHandleSeq_alloc()     DDS_Security_HandleSeq_alloc()
#define DDS_Security_ParticipantCryptoHandleSeq_allocbuf(l) DDS_Security_HandleSeq_allocbuf(l)
#define DDS_Security_ParticipantCryptoHandleSeq_freebuf(s)  DDS_Security_HandleSeq_freebuf(s)
#define DDS_Security_ParticipantCryptoHandleSeq_free(s)     DDS_Security_HandleSeq_free(s)
#define DDS_Security_ParticipantCryptoHandleSeq_deinit(s)   DDS_Security_HandleSeq_deinit(s)

#define DDS_Security_DatawriterCryptoHandleSeq_alloc()      DDS_Security_HandleSeq_alloc()
#define DDS_Security_DatawriterCryptoHandleSeq_allocbuf(l)  DDS_Security_HandleSeq_allocbuf(l)
#define DDS_Security_DatawriterCryptoHandleSeq_freebuf(s)   DDS_Security_HandleSeq_freebuf(s)
#define DDS_Security_DatawriterCryptoHandleSeq_free(s)      DDS_Security_HandleSeq_free(s)
#define DDS_Security_DatawriterCryptoHandleSeq_deinit(s)    DDS_Security_HandleSeq_deinit(s)

#define DDS_Security_DatareaderCryptoHandleSeq_alloc()      DDS_Security_HandleSeq_alloc()
#define DDS_Security_DatareaderCryptoHandleSeq_allocbuf(l)  DDS_Security_HandleSeq_allocbuf(l)
#define DDS_Security_DatareaderCryptoHandleSeq_freebuf(s)   DDS_Security_HandleSeq_freebuf(s)
#define DDS_Security_DatareaderCryptoHandleSeq_free(s)      DDS_Security_HandleSeq_free(s)
#define DDS_Security_DatareaderCryptoHandleSeq_deinit(s)    DDS_Security_HandleSeq_deinit(s)

#define DDS_Security_CryptoTokenSeq_alloc()     DDS_Security_DataHolderSeq_alloc()
#define DDS_Security_CryptoTokenSeq_allocbuf(l) DDS_Security_DataHolderSeq_allocbuf(l)
#define DDS_Security_CryptoTokenSeq_freebuf(s)  DDS_Security_DataHolderSeq_freebuf(s)
#define DDS_Security_CryptoTokenSeq_free(s)     DDS_Security_DataHolderSeq_free(s)


/* for DEBUG purposes */
DDS_EXPORT void
print_binary_debug(
         char* name,
         unsigned char *value,
         uint32_t size);

DDS_EXPORT void
print_binary_properties_debug(
         const DDS_Security_DataHolder *token);


#endif /* DSCMN_SECURITY_UTILS_H_ */
