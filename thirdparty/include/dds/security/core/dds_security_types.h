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

#ifndef DSS_SECURITY_PLUGIN_TYPES_H_
#define DSS_SECURITY_PLUGIN_TYPES_H_

#include "dds/security/dds_security_api_types.h"

typedef DDS_Security_octet DDS_Security_CryptoTransformKind[4];
typedef DDS_Security_octet DDS_Security_CryptoTransformKeyId[4];

/* enumeration for CryptoTransformKind.
 * ds_security_assign_CryptoTransformKind function should be used for assigning to CryptoTransformKind
 */
typedef enum
{
/* No encryption, no authentication tag */
  CRYPTO_TRANSFORMATION_KIND_NONE = 0,
 /* No encryption.
 One AES128-GMAC authentication tag using the sender_key
 Zero or more AES128-GMAC auth. tags with receiver specfic keys */
  CRYPTO_TRANSFORMATION_KIND_AES128_GMAC = 1,
 /* Authenticated Encryption using AES-128 in Galois Counter Mode
 (GCM) using the sender key.
 The authentication tag using the sender_key obtained from GCM
 Zero or more AES128-GMAC auth. tags with receiver specfic keys */
  CRYPTO_TRANSFORMATION_KIND_AES128_GCM = 2,
 /* No encryption.
 One AES256-GMAC authentication tag using the sender_key
 Zero or more AES256-GMAC auth. tags with receiver specfic keys */
  CRYPTO_TRANSFORMATION_KIND_AES256_GMAC = 3,
 /* Authenticated Encryption using AES-256 in Galois Counter Mode
 (GCM) using the sender key.
 The authentication tag using the sender_key obtained from GCM
 Zero or more AES256-GMAC auth. tags with receiver specfic keys */
  CRYPTO_TRANSFORMATION_KIND_AES256_GCM = 4,

  /* INVALID ENUM*/
  CRYPTO_TRANSFORMATION_KIND_INVALID = 127,
} DDS_Security_CryptoTransformKind_Enum;

typedef struct DDS_Security_KeyMaterial_AES_GCM_GMAC {
  DDS_Security_CryptoTransformKind transformation_kind;
  DDS_Security_OctetSeq master_salt;         /*size shall be 16 or 32*/
  DDS_Security_CryptoTransformKeyId sender_key_id;
  DDS_Security_OctetSeq master_sender_key;   /*size shall be 16 or 32*/
  DDS_Security_CryptoTransformKeyId receiver_specific_key_id;
  DDS_Security_OctetSeq master_receiver_specific_key; /*size shall be 0, 16 or 32*/
} DDS_Security_KeyMaterial_AES_GCM_GMAC;

struct CryptoTransformIdentifier {
  DDS_Security_CryptoTransformKind transformation_kind;
  DDS_Security_CryptoTransformKeyId transformation_key_id;
};

/** temporary address decleration until it is ready in ddsrt */
typedef uintptr_t          ddsrt_address;   /* word length of the platform */


#endif /* DSS_SECURITY_PLUGIN_TYPES_H_ */
