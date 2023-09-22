/*
 * Copyright(c) 2006 to 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */

#ifndef DDS_SECURITY_API_CRYPTOGRAPHY_H
#define DDS_SECURITY_API_CRYPTOGRAPHY_H

#include "dds_security_api_types.h"

#if defined(__cplusplus)
extern "C"
{
#endif

/* Crypto Component */
struct dds_security_crypto_key_factory;
typedef struct dds_security_crypto_key_factory dds_security_crypto_key_factory;

struct dds_security_crypto_key_exchange;
typedef struct dds_security_crypto_key_exchange dds_security_crypto_key_exchange;

struct dds_security_crypto_transform;
typedef struct dds_security_crypto_transform dds_security_crypto_transform;

/* CryptoKeyFactory interface */
typedef DDS_Security_ParticipantCryptoHandle (*DDS_Security_crypto_key_factory_register_local_participant)(
    dds_security_crypto_key_factory *instance,
    const DDS_Security_IdentityHandle participant_identity,
    const DDS_Security_PermissionsHandle participant_permissions,
    const DDS_Security_PropertySeq *participant_properties,
    const DDS_Security_ParticipantSecurityAttributes *participant_security_attributes,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_ParticipantCryptoHandle (*DDS_Security_crypto_key_factory_register_matched_remote_participant)(
    dds_security_crypto_key_factory *instance,
    const DDS_Security_ParticipantCryptoHandle local_participant_crypto_handle,
    const DDS_Security_IdentityHandle remote_participant_identity,
    const DDS_Security_PermissionsHandle remote_participant_permissions,
    const DDS_Security_SharedSecretHandle shared_secret,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_DatawriterCryptoHandle (*DDS_Security_crypto_key_factory_register_local_datawriter)(
    dds_security_crypto_key_factory *instance,
    const DDS_Security_ParticipantCryptoHandle participant_crypto,
    const DDS_Security_PropertySeq *datawriter_properties,
    const DDS_Security_EndpointSecurityAttributes *datawriter_security_attributes,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_DatareaderCryptoHandle (*DDS_Security_crypto_key_factory_register_matched_remote_datareader)(
    dds_security_crypto_key_factory *instance,
    const DDS_Security_DatawriterCryptoHandle local_datawriter_crypto_handle,
    const DDS_Security_ParticipantCryptoHandle remote_participant_crypto,
    const DDS_Security_SharedSecretHandle shared_secret,
    const DDS_Security_boolean relay_only,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_DatareaderCryptoHandle (*DDS_Security_crypto_key_factory_register_local_datareader)(
    dds_security_crypto_key_factory *instance,
    const DDS_Security_ParticipantCryptoHandle participant_crypto_handle,
    const DDS_Security_PropertySeq *datareader_properties,
    const DDS_Security_EndpointSecurityAttributes *datareader_security_attributes,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_DatawriterCryptoHandle (*DDS_Security_crypto_key_factory_register_matched_remote_datawriter)(
    dds_security_crypto_key_factory *instance,
    const DDS_Security_DatareaderCryptoHandle local_datareader_crypto_handle,
    const DDS_Security_ParticipantCryptoHandle remote_participant_crypt,
    const DDS_Security_SharedSecretHandle shared_secret,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_key_factory_unregister_participant)(
    dds_security_crypto_key_factory *instance,
    const DDS_Security_ParticipantCryptoHandle participant_crypto_handle,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_key_factory_unregister_datawriter)(
    dds_security_crypto_key_factory *instance,
    const DDS_Security_DatawriterCryptoHandle datawriter_crypto_handle,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_key_factory_unregister_datareader)(
    dds_security_crypto_key_factory *instance,
    const DDS_Security_DatareaderCryptoHandle datareader_crypto_handle,
    DDS_Security_SecurityException *ex);

struct dds_security_crypto_key_factory
{
  DDS_Security_crypto_key_factory_register_local_participant register_local_participant;
  DDS_Security_crypto_key_factory_register_matched_remote_participant register_matched_remote_participant;
  DDS_Security_crypto_key_factory_register_local_datawriter register_local_datawriter;
  DDS_Security_crypto_key_factory_register_matched_remote_datareader register_matched_remote_datareader;
  DDS_Security_crypto_key_factory_register_local_datareader register_local_datareader;
  DDS_Security_crypto_key_factory_register_matched_remote_datawriter register_matched_remote_datawriter;
  DDS_Security_crypto_key_factory_unregister_participant unregister_participant;
  DDS_Security_crypto_key_factory_unregister_datawriter unregister_datawriter;
  DDS_Security_crypto_key_factory_unregister_datareader unregister_datareader;
};

/* CryptoKeyExchange Interface */
typedef DDS_Security_boolean (*DDS_Security_crypto_key_exchange_create_local_participant_crypto_tokens)(
    dds_security_crypto_key_exchange *instance,
    DDS_Security_ParticipantCryptoTokenSeq *local_participant_crypto_tokens,
    const DDS_Security_ParticipantCryptoHandle local_participant_crypto,
    const DDS_Security_ParticipantCryptoHandle remote_participant_crypto,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_key_exchange_set_remote_participant_crypto_tokens)(
    dds_security_crypto_key_exchange *instance,
    const DDS_Security_ParticipantCryptoHandle local_participant_crypto,
    const DDS_Security_ParticipantCryptoHandle remote_participant_crypto,
    const DDS_Security_ParticipantCryptoTokenSeq *remote_participant_tokens,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_key_exchange_create_local_datawriter_crypto_tokens)(
    dds_security_crypto_key_exchange *instance,
    DDS_Security_DatawriterCryptoTokenSeq *local_datawriter_crypto_tokens,
    const DDS_Security_DatawriterCryptoHandle local_datawriter_crypto,
    const DDS_Security_DatareaderCryptoHandle remote_datareader_crypto,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_key_exchange_set_remote_datawriter_crypto_tokens)(
    dds_security_crypto_key_exchange *instance,
    const DDS_Security_DatareaderCryptoHandle local_datareader_crypto,
    const DDS_Security_DatawriterCryptoHandle remote_datawriter_crypto,
    const DDS_Security_DatawriterCryptoTokenSeq *remote_datawriter_tokens,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_key_exchange_create_local_datareader_crypto_tokens)(
    dds_security_crypto_key_exchange *instance,
    DDS_Security_DatareaderCryptoTokenSeq *local_datareader_cryto_tokens,
    const DDS_Security_DatareaderCryptoHandle local_datareader_crypto,
    const DDS_Security_DatawriterCryptoHandle remote_datawriter_crypto,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_key_exchange_set_remote_datareader_crypto_tokens)(
    dds_security_crypto_key_exchange *instance,
    const DDS_Security_DatawriterCryptoHandle local_datawriter_crypto,
    const DDS_Security_DatareaderCryptoHandle remote_datareader_crypto,
    const DDS_Security_DatareaderCryptoTokenSeq *remote_datareader_tokens,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_key_exchange_return_crypto_tokens)(
    dds_security_crypto_key_exchange *instance,
    DDS_Security_CryptoTokenSeq *crypto_tokens,
    DDS_Security_SecurityException *ex);

struct dds_security_crypto_key_exchange
{
  DDS_Security_crypto_key_exchange_create_local_participant_crypto_tokens create_local_participant_crypto_tokens;
  DDS_Security_crypto_key_exchange_set_remote_participant_crypto_tokens set_remote_participant_crypto_tokens;
  DDS_Security_crypto_key_exchange_create_local_datawriter_crypto_tokens create_local_datawriter_crypto_tokens;
  DDS_Security_crypto_key_exchange_set_remote_datawriter_crypto_tokens set_remote_datawriter_crypto_tokens;
  DDS_Security_crypto_key_exchange_create_local_datareader_crypto_tokens create_local_datareader_crypto_tokens;
  DDS_Security_crypto_key_exchange_set_remote_datareader_crypto_tokens set_remote_datareader_crypto_tokens;
  DDS_Security_crypto_key_exchange_return_crypto_tokens return_crypto_tokens;
};

/* CryptoTransform Interface */
typedef DDS_Security_boolean (*DDS_Security_crypto_transform_encode_serialized_payload)(
    dds_security_crypto_transform *instance,
    DDS_Security_OctetSeq *encoded_buffer,
    DDS_Security_OctetSeq *extra_inline_qos,
    const DDS_Security_OctetSeq *plain_buffer,
    const DDS_Security_DatawriterCryptoHandle sending_datawriter_crypto,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_transform_encode_datawriter_submessage)(
    dds_security_crypto_transform *instance,
    DDS_Security_OctetSeq *encoded_rtps_submessage,
    const DDS_Security_OctetSeq *plain_rtps_submessage,
    const DDS_Security_DatawriterCryptoHandle sending_datawriter_crypto,
    const DDS_Security_DatareaderCryptoHandleSeq *receiving_datareader_crypto_list,
    DDS_Security_long *receiving_datareader_crypto_list_index,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_transform_encode_datareader_submessage)(
    dds_security_crypto_transform *instance,
    DDS_Security_OctetSeq *encoded_rtps_submessage,
    const DDS_Security_OctetSeq *plain_rtps_submessage,
    const DDS_Security_DatareaderCryptoHandle sending_datareader_crypto,
    const DDS_Security_DatawriterCryptoHandleSeq *receiving_datawriter_crypto_list,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_transform_encode_rtps_message)(
    dds_security_crypto_transform *instance,
    DDS_Security_OctetSeq *encoded_rtps_message,
    const DDS_Security_OctetSeq *plain_rtps_message,
    const DDS_Security_ParticipantCryptoHandle sending_participant_crypto,
    const DDS_Security_ParticipantCryptoHandleSeq *receiving_participant_crypto_list,
    DDS_Security_long *receiving_participant_crypto_list_index,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_transform_decode_rtps_message)(
    dds_security_crypto_transform *instance,
    DDS_Security_OctetSeq *plain_buffer,
    const DDS_Security_OctetSeq *encoded_buffer,
    const DDS_Security_ParticipantCryptoHandle receiving_participant_crypto,
    const DDS_Security_ParticipantCryptoHandle sending_participant_crypto,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_transform_preprocess_secure_submsg)(
    dds_security_crypto_transform *instance,
    DDS_Security_DatawriterCryptoHandle *datawriter_crypto,
    DDS_Security_DatareaderCryptoHandle *datareader_crypto,
    DDS_Security_SecureSubmessageCategory_t *secure_submessage_category,
    const DDS_Security_OctetSeq *encoded_rtps_submessage,
    const DDS_Security_ParticipantCryptoHandle receiving_participant_crypto,
    const DDS_Security_ParticipantCryptoHandle sending_participant_crypto,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_transform_decode_datawriter_submessage)(
    dds_security_crypto_transform *instance,
    DDS_Security_OctetSeq *plain_rtps_submessage,
    const DDS_Security_OctetSeq *encoded_rtps_submessage,
    const DDS_Security_DatareaderCryptoHandle receiving_datareader_crypto,
    const DDS_Security_DatawriterCryptoHandle sending_datawriter_crypto,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_transform_decode_datareader_submessage)(
    dds_security_crypto_transform *instance,
    DDS_Security_OctetSeq *plain_rtps_message,
    const DDS_Security_OctetSeq *encoded_rtps_message,
    const DDS_Security_DatawriterCryptoHandle receiving_datawriter_crypto,
    const DDS_Security_DatareaderCryptoHandle sending_datareader_crypto,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_crypto_transform_decode_serialized_payload)(
    dds_security_crypto_transform *instance,
    DDS_Security_OctetSeq *plain_buffer,
    const DDS_Security_OctetSeq *encoded_buffer,
    const DDS_Security_OctetSeq *inline_qos,
    const DDS_Security_DatareaderCryptoHandle receiving_datareader_crypto,
    const DDS_Security_DatawriterCryptoHandle sending_datawriter_crypto,
    DDS_Security_SecurityException *ex);

struct dds_security_crypto_transform
{
  DDS_Security_crypto_transform_encode_serialized_payload encode_serialized_payload;
  DDS_Security_crypto_transform_encode_datawriter_submessage encode_datawriter_submessage;
  DDS_Security_crypto_transform_encode_datareader_submessage encode_datareader_submessage;
  DDS_Security_crypto_transform_encode_rtps_message encode_rtps_message;
  DDS_Security_crypto_transform_decode_rtps_message decode_rtps_message;
  DDS_Security_crypto_transform_preprocess_secure_submsg preprocess_secure_submsg;
  DDS_Security_crypto_transform_decode_datawriter_submessage decode_datawriter_submessage;
  DDS_Security_crypto_transform_decode_datareader_submessage decode_datareader_submessage;
  DDS_Security_crypto_transform_decode_serialized_payload decode_serialized_payload;
};

typedef struct dds_security_cryptography
{
  struct ddsi_domaingv *gv;

  dds_security_crypto_transform *crypto_transform;
  dds_security_crypto_key_factory *crypto_key_factory;
  dds_security_crypto_key_exchange *crypto_key_exchange;
} dds_security_cryptography;

#if defined(__cplusplus)
}
#endif

#endif /* DDS_SECURITY_API_CRYPTOGRAPHY_H */
