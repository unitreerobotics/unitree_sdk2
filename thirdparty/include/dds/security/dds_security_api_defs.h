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

#ifndef DDS_SECURITY_API_DEF_H
#define DDS_SECURITY_API_DEF_H

#include "dds_security_api_err.h"

#if defined (__cplusplus)
extern "C" {
#endif



/**************************************************************************
 *                                                                        *
 * Return values.                                                         *
 *                                                                        *
 **************************************************************************/
typedef enum {
    DDS_SECURITY_VALIDATION_OK,
    DDS_SECURITY_VALIDATION_FAILED,
    DDS_SECURITY_VALIDATION_PENDING_RETRY,
    DDS_SECURITY_VALIDATION_PENDING_HANDSHAKE_REQUEST,
    DDS_SECURITY_VALIDATION_PENDING_HANDSHAKE_MESSAGE,
    DDS_SECURITY_VALIDATION_OK_FINAL_MESSAGE
} DDS_Security_ValidationResult_t;

#define DDS_SECURITY_HANDLE_NIL (0)

#define DDS_SECURITY_SUCCESS (0)
#define DDS_SECURITY_FAILED (-1)


/**************************************************************************
 *                                                                        *
 * Attribute flags.                                                       *
 *                                                                        *
 **************************************************************************/
#define DDS_SECURITY_PARTICIPANT_ATTRIBUTES_FLAG_IS_RTPS_PROTECTED                      (1u      )
#define DDS_SECURITY_PARTICIPANT_ATTRIBUTES_FLAG_IS_DISCOVERY_PROTECTED                 (1u <<  1)
#define DDS_SECURITY_PARTICIPANT_ATTRIBUTES_FLAG_IS_LIVELINESS_PROTECTED                (1u <<  2)
#define DDS_SECURITY_PARTICIPANT_ATTRIBUTES_FLAG_IS_VALID                               (1u << 31)

#define DDS_SECURITY_PLUGIN_PARTICIPANT_ATTRIBUTES_FLAG_IS_RTPS_ENCRYPTED               (1u      )
#define DDS_SECURITY_PLUGIN_PARTICIPANT_ATTRIBUTES_FLAG_IS_DISCOVERY_ENCRYPTED          (1u <<  1)
#define DDS_SECURITY_PLUGIN_PARTICIPANT_ATTRIBUTES_FLAG_IS_LIVELINESS_ENCRYPTED         (1u <<  2)
#define DDS_SECURITY_PLUGIN_PARTICIPANT_ATTRIBUTES_FLAG_IS_RTPS_AUTHENTICATED           (1u <<  3)
#define DDS_SECURITY_PLUGIN_PARTICIPANT_ATTRIBUTES_FLAG_IS_DISCOVERY_AUTHENTICATED      (1u <<  4)
#define DDS_SECURITY_PLUGIN_PARTICIPANT_ATTRIBUTES_FLAG_IS_LIVELINESS_AUTHENTICATED     (1u <<  5)

#define DDS_SECURITY_ENDPOINT_ATTRIBUTES_FLAG_IS_READ_PROTECTED                         (1u      )
#define DDS_SECURITY_ENDPOINT_ATTRIBUTES_FLAG_IS_WRITE_PROTECTED                        (1u <<  1)
#define DDS_SECURITY_ENDPOINT_ATTRIBUTES_FLAG_IS_DISCOVERY_PROTECTED                    (1u <<  2)
#define DDS_SECURITY_ENDPOINT_ATTRIBUTES_FLAG_IS_SUBMESSAGE_PROTECTED                   (1u <<  3)
#define DDS_SECURITY_ENDPOINT_ATTRIBUTES_FLAG_IS_PAYLOAD_PROTECTED                      (1u <<  4)
#define DDS_SECURITY_ENDPOINT_ATTRIBUTES_FLAG_IS_KEY_PROTECTED                          (1u <<  5)
#define DDS_SECURITY_ENDPOINT_ATTRIBUTES_FLAG_IS_LIVELINESS_PROTECTED                   (1u <<  6)
#define DDS_SECURITY_ENDPOINT_ATTRIBUTES_FLAG_IS_VALID                                  (1u << 31)

#define DDS_SECURITY_PLUGIN_ENDPOINT_ATTRIBUTES_FLAG_IS_SUBMESSAGE_ENCRYPTED            (1u      )
#define DDS_SECURITY_PLUGIN_ENDPOINT_ATTRIBUTES_FLAG_IS_PAYLOAD_ENCRYPTED               (1u <<  1)
#define DDS_SECURITY_PLUGIN_ENDPOINT_ATTRIBUTES_FLAG_IS_SUBMESSAGE_ORIGIN_AUTHENTICATED (1u <<  2)




/**************************************************************************
 *                                                                        *
 * Protection types.                                                      *
 *                                                                        *
 **************************************************************************/
typedef enum {
    DDS_SECURITY_PROTECTION_KIND_ENCRYPT_WITH_ORIGIN_AUTHENTICATION,
    DDS_SECURITY_PROTECTION_KIND_SIGN_WITH_ORIGIN_AUTHENTICATION,
    DDS_SECURITY_PROTECTION_KIND_ENCRYPT,
    DDS_SECURITY_PROTECTION_KIND_SIGN,
    DDS_SECURITY_PROTECTION_KIND_NONE
} DDS_Security_ProtectionKind;

typedef enum {
    DDS_SECURITY_BASICPROTECTION_KIND_ENCRYPT,
    DDS_SECURITY_BASICPROTECTION_KIND_SIGN,
    DDS_SECURITY_BASICPROTECTION_KIND_NONE
} DDS_Security_BasicProtectionKind;




/**************************************************************************
 *                                                                        *
 * Submessage categories.                                                 *
 *                                                                        *
 **************************************************************************/
typedef enum {
    DDS_SECURITY_INFO_SUBMESSAGE,
    DDS_SECURITY_DATAWRITER_SUBMESSAGE,
    DDS_SECURITY_DATAREADER_SUBMESSAGE
} DDS_Security_SecureSubmessageCategory_t;




/**************************************************************************
 *                                                                        *
 * QoS Policies content.                                                  *
 *                                                                        *
 **************************************************************************/
typedef enum {
    DDS_SECURITY_AUTOMATIC_LIVELINESS_QOS,
    DDS_SECURITY_MANUAL_BY_PARTICIPANT_LIVELINESS_QOS,
    DDS_SECURITY_MANUAL_BY_TOPIC_LIVELINESS_QOS
} DDS_Security_LivelinessQosPolicyKind;

typedef enum {
    DDS_SECURITY_BEST_EFFORT_RELIABILITY_QOS,
    DDS_SECURITY_RELIABLE_RELIABILITY_QOS
} DDS_Security_ReliabilityQosPolicyKind;

typedef enum  {
    DDS_SECURITY_BY_RECEPTION_TIMESTAMP_DESTINATIONORDER_QOS,
    DDS_SECURITY_BY_SOURCE_TIMESTAMP_DESTINATIONORDER_QOS
} DDS_Security_DestinationOrderQosPolicyKind;

typedef enum {
    DDS_SECURITY_INSTANCE_PRESENTATION_QOS,
    DDS_SECURITY_TOPIC_PRESENTATION_QOS,
    DDS_SECURITY_GROUP_PRESENTATION_QOS
} DDS_Security_PresentationQosPolicyAccessScopeKind;

typedef enum {
    DDS_SECURITY_KEEP_LAST_HISTORY_QOS,
    DDS_SECURITY_KEEP_ALL_HISTORY_QOS
} DDS_Security_HistoryQosPolicyKind;

typedef enum {
    DDS_SECURITY_VOLATILE_DURABILITY_QOS,
    DDS_SECURITY_TRANSIENT_LOCAL_DURABILITY_QOS,
    DDS_SECURITY_TRANSIENT_DURABILITY_QOS,
    DDS_SECURITY_PERSISTENT_DURABILITY_QOS
} DDS_Security_DurabilityQosPolicyKind;

typedef enum {
    DDS_SECURITY_SHARED_OWNERSHIP_QOS,
    DDS_SECURITY_EXCLUSIVE_OWNERSHIP_QOS
} DDS_Security_OwnershipQosPolicyKind;




/**************************************************************************
 *                                                                        *
 * Listener information.                                                  *
 *                                                                        *
 **************************************************************************/
typedef enum {
    DDS_SECURITY_IDENTITY_STATUS
} DDS_Security_AuthStatusKind;




/**************************************************************************
 *                                                                        *
 * Some byte array sizes.                                                 *
 *                                                                        *
 **************************************************************************/
#define DDS_SECURITY_AUTHENTICATION_CHALLENGE_SIZE 32

#define DDS_SECURITY_MASTER_SALT_SIZE_128 16
#define DDS_SECURITY_MASTER_SALT_SIZE_256 32
#define DDS_SECURITY_MASTER_SENDER_KEY_SIZE_128 16
#define DDS_SECURITY_MASTER_SENDER_KEY_SIZE_256 32
#define DDS_SECURITY_MASTER_RECEIVER_SPECIFIC_KEY_SIZE_128 16
#define DDS_SECURITY_MASTER_RECEIVER_SPECIFIC_KEY_SIZE_256 32

/**************************************************************************
 *                                                                        *
 * Security Property Key Names                                                     *
 *                                                                        *
 *************************************************************************/
#define DDS_SEC_PROP_PREFIX "dds.sec."
#define ORG_ECLIPSE_CYCLONEDDS_SEC_PREFIX "org.eclipse.cyclonedds.sec."

#define DDS_SEC_PROP_AUTH_LIBRARY_PATH DDS_SEC_PROP_PREFIX "auth.library.path"
#define DDS_SEC_PROP_AUTH_LIBRARY_INIT DDS_SEC_PROP_PREFIX "auth.library.init"
#define DDS_SEC_PROP_AUTH_LIBRARY_FINALIZE DDS_SEC_PROP_PREFIX "auth.library.finalize"
#define DDS_SEC_PROP_CRYPTO_LIBRARY_PATH DDS_SEC_PROP_PREFIX "crypto.library.path"
#define DDS_SEC_PROP_CRYPTO_LIBRARY_INIT DDS_SEC_PROP_PREFIX "crypto.library.init"
#define DDS_SEC_PROP_CRYPTO_LIBRARY_FINALIZE DDS_SEC_PROP_PREFIX "crypto.library.finalize"
#define DDS_SEC_PROP_ACCESS_LIBRARY_PATH DDS_SEC_PROP_PREFIX "access.library.path"
#define DDS_SEC_PROP_ACCESS_LIBRARY_INIT DDS_SEC_PROP_PREFIX "access.library.init"
#define DDS_SEC_PROP_ACCESS_LIBRARY_FINALIZE DDS_SEC_PROP_PREFIX "access.library.finalize"

#define DDS_SEC_PROP_AUTH_IDENTITY_CA DDS_SEC_PROP_PREFIX "auth.identity_ca"
#define DDS_SEC_PROP_AUTH_PRIV_KEY DDS_SEC_PROP_PREFIX "auth.private_key"
#define DDS_SEC_PROP_AUTH_IDENTITY_CERT DDS_SEC_PROP_PREFIX "auth.identity_certificate"
#define DDS_SEC_PROP_AUTH_PASSWORD DDS_SEC_PROP_PREFIX "auth.password"
#define ORG_ECLIPSE_CYCLONEDDS_SEC_AUTH_CRL ORG_ECLIPSE_CYCLONEDDS_SEC_PREFIX "auth.crl"
#define DDS_SEC_PROP_ACCESS_PERMISSIONS_CA DDS_SEC_PROP_PREFIX "access.permissions_ca"
#define DDS_SEC_PROP_ACCESS_GOVERNANCE DDS_SEC_PROP_PREFIX "access.governance"
#define DDS_SEC_PROP_ACCESS_PERMISSIONS DDS_SEC_PROP_PREFIX "access.permissions"
#define DDS_SEC_PROP_ACCESS_TRUSTED_CA_DIR DDS_SEC_PROP_PREFIX "auth.trusted_ca_dir"

#define DDS_SEC_PROP_BUILTIN_ENDPOINT_NAME DDS_SEC_PROP_PREFIX "builtin_endpoint_name"
#define DDS_SEC_PROP_CRYPTO_KEYSIZE DDS_SEC_PROP_PREFIX "crypto.keysize"

#if defined (__cplusplus)
}
#endif


#endif /* DDS_SECURITY_API_DEF_H */
