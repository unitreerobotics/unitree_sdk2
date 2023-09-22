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

#ifndef DDS_SECURITY_AUTHENTICATION_API_H
#define DDS_SECURITY_AUTHENTICATION_API_H

#include "dds_security_api_types.h"

#if defined(__cplusplus)
extern "C"
{
#endif

/* Authentication Component */
struct dds_security_authentication;
typedef struct dds_security_authentication dds_security_authentication;

struct dds_security_authentication_listener;
typedef struct dds_security_authentication_listener dds_security_authentication_listener;

/* AuthenticationListener interface */
typedef DDS_Security_boolean (*DDS_Security_authentication_listener_on_revoke_identity)(
    const dds_security_authentication *plugin,
    const DDS_Security_IdentityHandle handle);

typedef DDS_Security_boolean (*DDS_Security_authentication_listener_on_status_changed)(
    dds_security_authentication_listener *context,
    const dds_security_authentication *plugin,
    const DDS_Security_IdentityHandle handle,
    const DDS_Security_AuthStatusKind status_kind);

struct dds_security_authentication_listener
{
  DDS_Security_authentication_listener_on_revoke_identity on_revoke_identity;
  DDS_Security_authentication_listener_on_status_changed on_status_changed;
};

typedef DDS_Security_ValidationResult_t (*DDS_Security_authentication_validate_local_identity)(
    dds_security_authentication *instance,
    DDS_Security_IdentityHandle *local_identity_handle,
    DDS_Security_GUID_t *adjusted_participant_guid,
    const DDS_Security_DomainId domain_id,
    const DDS_Security_Qos *participant_qos,
    const DDS_Security_GUID_t *candidate_participant_guid,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_authentication_get_identity_token)(
    dds_security_authentication *instance,
    DDS_Security_IdentityToken *identity_token,
    const DDS_Security_IdentityHandle handle,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_authentication_get_identity_status_token)(
    dds_security_authentication *instance,
    DDS_Security_IdentityStatusToken *identity_status_token,
    const DDS_Security_IdentityHandle handle,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_authentication_set_permissions_credential_and_token)(
    dds_security_authentication *instance,
    const DDS_Security_IdentityHandle handle,
    const DDS_Security_PermissionsCredentialToken *permissions_credential,
    const DDS_Security_PermissionsToken *permissions_token,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_ValidationResult_t (*DDS_Security_authentication_validate_remote_identity)(
    dds_security_authentication *instance,
    DDS_Security_IdentityHandle *remote_identity_handle,
    DDS_Security_AuthRequestMessageToken *local_auth_request_token,
    const DDS_Security_AuthRequestMessageToken *remote_auth_request_token,
    const DDS_Security_IdentityHandle local_identity_handle,
    const DDS_Security_IdentityToken *remote_identity_token,
    const DDS_Security_GUID_t *remote_participant_guid,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_ValidationResult_t (*DDS_Security_authentication_begin_handshake_request)(
    dds_security_authentication *instance,
    DDS_Security_HandshakeHandle *handshake_handle,
    DDS_Security_HandshakeMessageToken *handshake_message,
    const DDS_Security_IdentityHandle initiator_identity_handle,
    const DDS_Security_IdentityHandle replier_identity_handle,
    const DDS_Security_OctetSeq *serialized_local_participant_data,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_ValidationResult_t (*DDS_Security_authentication_begin_handshake_reply)(
    dds_security_authentication *instance,
    DDS_Security_HandshakeHandle *handshake_handle,
    DDS_Security_HandshakeMessageToken *handshake_message_out,
    const DDS_Security_HandshakeMessageToken *handshake_message_in,
    const DDS_Security_IdentityHandle initiator_identity_handle,
    const DDS_Security_IdentityHandle replier_identity_handle,
    const DDS_Security_OctetSeq *serialized_local_participant_data,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_ValidationResult_t (*DDS_Security_authentication_process_handshake)(
    dds_security_authentication *instance,
    DDS_Security_HandshakeMessageToken *handshake_message_out,
    const DDS_Security_HandshakeMessageToken *handshake_message_in,
    const DDS_Security_HandshakeHandle handshake_handle,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_SharedSecretHandle (*DDS_Security_authentication_get_shared_secret)(
    dds_security_authentication *instance,
    const DDS_Security_HandshakeHandle handshake_handle,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_authentication_get_authenticated_peer_credential_token)(
    dds_security_authentication *instance,
    DDS_Security_AuthenticatedPeerCredentialToken *peer_credential_token,
    const DDS_Security_HandshakeHandle handshake_handle,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_authentication_set_listener)(
    dds_security_authentication *instance,
    const dds_security_authentication_listener *listener,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_authentication_return_identity_token)(
    dds_security_authentication *instance,
    const DDS_Security_IdentityToken *token,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_authentication_return_identity_status_token)(
    dds_security_authentication *instance,
    const DDS_Security_IdentityStatusToken *token,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_authentication_return_authenticated_peer_credential_token)(
    dds_security_authentication *instance,
    const DDS_Security_AuthenticatedPeerCredentialToken *peer_credential_token,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_authentication_return_handshake_handle)(
    dds_security_authentication *instance,
    const DDS_Security_HandshakeHandle handshake_handle,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_authentication_return_identity_handle)(
    dds_security_authentication *instance,
    const DDS_Security_IdentityHandle identity_handle,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_authentication_return_sharedsecret_handle)(
    dds_security_authentication *instance,
    const DDS_Security_SharedSecretHandle sharedsecret_handle,
    DDS_Security_SecurityException *ex);

struct dds_security_authentication
{
  struct ddsi_domaingv *gv;

  DDS_Security_authentication_validate_local_identity validate_local_identity;
  DDS_Security_authentication_get_identity_token get_identity_token;
  DDS_Security_authentication_get_identity_status_token get_identity_status_token;
  DDS_Security_authentication_set_permissions_credential_and_token set_permissions_credential_and_token;
  DDS_Security_authentication_validate_remote_identity validate_remote_identity;
  DDS_Security_authentication_begin_handshake_request begin_handshake_request;
  DDS_Security_authentication_begin_handshake_reply begin_handshake_reply;
  DDS_Security_authentication_process_handshake process_handshake;
  DDS_Security_authentication_get_shared_secret get_shared_secret;
  DDS_Security_authentication_get_authenticated_peer_credential_token get_authenticated_peer_credential_token;
  DDS_Security_authentication_set_listener set_listener;
  DDS_Security_authentication_return_identity_token return_identity_token;
  DDS_Security_authentication_return_identity_status_token return_identity_status_token;
  DDS_Security_authentication_return_authenticated_peer_credential_token return_authenticated_peer_credential_token;
  DDS_Security_authentication_return_handshake_handle return_handshake_handle;
  DDS_Security_authentication_return_identity_handle return_identity_handle;
  DDS_Security_authentication_return_sharedsecret_handle return_sharedsecret_handle;
};

#if defined(__cplusplus)
}
#endif

#endif /* DDS_SECURITY_AUTHENTICATION_API_H */
