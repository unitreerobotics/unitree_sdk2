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

#ifndef DDS_SECURITY_ACCESS_CONTROL_API_H
#define DDS_SECURITY_ACCESS_CONTROL_API_H

#include "dds_security_api_types.h"
#include "dds_security_api_authentication.h"

#if defined(__cplusplus)
extern "C"
{
#endif

/* AccessControl Component */
struct dds_security_access_control;
typedef struct dds_security_access_control dds_security_access_control;


/* AccessControlListener Interface */
struct dds_security_access_control_listener;
typedef struct dds_security_access_control_listener dds_security_access_control_listener;

typedef DDS_Security_boolean (*DDS_Security_access_control_listener_on_revoke_permissions)(
    const dds_security_access_control *plugin,
    const DDS_Security_PermissionsHandle handle);

struct dds_security_access_control_listener
{
  DDS_Security_access_control_listener_on_revoke_permissions on_revoke_permissions;
};

/* AccessControl Interface */
typedef DDS_Security_PermissionsHandle (*DDS_Security_access_control_validate_local_permissions)(
    dds_security_access_control *instance,
    const dds_security_authentication *auth_plugin,
    const DDS_Security_IdentityHandle identity,
    const DDS_Security_DomainId domain_id,
    const DDS_Security_Qos *participant_qos,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_PermissionsHandle (*DDS_Security_access_control_validate_remote_permissions)(
    dds_security_access_control *instance,
    const dds_security_authentication *auth_plugin,
    const DDS_Security_IdentityHandle local_identity_handle,
    const DDS_Security_IdentityHandle remote_identity_handle,
    const DDS_Security_PermissionsToken *remote_permissions_token,
    const DDS_Security_AuthenticatedPeerCredentialToken *remote_credential_token,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_create_participant)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_DomainId domain_id,
    const DDS_Security_Qos *participant_qos,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_create_datawriter)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_DomainId domain_id,
    const DDS_Security_char *topic_name,
    const DDS_Security_Qos *writer_qos,
    const DDS_Security_PartitionQosPolicy *partition,
    const DDS_Security_DataTags *data_tag,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_create_datareader)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_DomainId domain_id,
    const DDS_Security_char *topic_name,
    const DDS_Security_Qos *reader_qos,
    const DDS_Security_PartitionQosPolicy *partition,
    const DDS_Security_DataTags *data_tag,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_create_topic)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_DomainId domain_id,
    const DDS_Security_char *topic_name,
    const DDS_Security_Qos *topic_qos,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_local_datawriter_register_instance)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_Entity *writer,
    const DDS_Security_DynamicData *key,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_local_datawriter_dispose_instance)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_Entity *writer,
    const DDS_Security_DynamicData key,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_remote_participant)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_DomainId domain_id,
    const DDS_Security_ParticipantBuiltinTopicDataSecure *participant_data,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_remote_datawriter)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_DomainId domain_id,
    const DDS_Security_PublicationBuiltinTopicDataSecure *publication_data,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_remote_datareader)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_DomainId domain_id,
    const DDS_Security_SubscriptionBuiltinTopicDataSecure *subscription_data,
    DDS_Security_boolean *relay_only,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_remote_topic)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_DomainId domain_id,
    const DDS_Security_TopicBuiltinTopicData *topic_data,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_local_datawriter_match)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle writer_permissions_handle,
    const DDS_Security_PermissionsHandle reader_permissions_handle,
    const DDS_Security_PublicationBuiltinTopicDataSecure *publication_data,
    const DDS_Security_SubscriptionBuiltinTopicDataSecure *subscription_data,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_local_datareader_match)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle reader_permissions_handle,
    const DDS_Security_PermissionsHandle writer_permissions_handle,
    const DDS_Security_SubscriptionBuiltinTopicDataSecure *subscription_data,
    const DDS_Security_PublicationBuiltinTopicDataSecure *publication_data,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_remote_datawriter_register_instance)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_Entity *reader,
    const DDS_Security_InstanceHandle publication_handle,
    const DDS_Security_DynamicData key,
    const DDS_Security_InstanceHandle instance_handle,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_check_remote_datawriter_dispose_instance)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_Entity *reader,
    const DDS_Security_InstanceHandle publication_handle,
    const DDS_Security_DynamicData key,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_get_permissions_token)(
    dds_security_access_control *instance,
    DDS_Security_PermissionsToken *permissions_token,
    const DDS_Security_PermissionsHandle handle,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_get_permissions_credential_token)(
    dds_security_access_control *instance,
    DDS_Security_PermissionsCredentialToken *permissions_credential_token,
    const DDS_Security_PermissionsHandle handle,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_set_listener)(
    dds_security_access_control *instance,
    const dds_security_access_control_listener *listener,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_return_permissions_token)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsToken *token,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_return_permissions_credential_token)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsCredentialToken *permissions_credential_token,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_get_participant_sec_attributes)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    DDS_Security_ParticipantSecurityAttributes *attributes,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_get_topic_sec_attributes)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_char *topic_name,
    DDS_Security_TopicSecurityAttributes *attributes,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_get_datawriter_sec_attributes)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_char *topic_name,
    const DDS_Security_PartitionQosPolicy *partition,
    const DDS_Security_DataTagQosPolicy *data_tag,
    DDS_Security_EndpointSecurityAttributes *attributes,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_get_datareader_sec_attributes)(
    dds_security_access_control *instance,
    const DDS_Security_PermissionsHandle permissions_handle,
    const DDS_Security_char *topic_name,
    const DDS_Security_PartitionQosPolicy *partition,
    const DDS_Security_DataTagQosPolicy *data_tag,
    DDS_Security_EndpointSecurityAttributes *attributes,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_return_participant_sec_attributes)(
    dds_security_access_control *instance,
    const DDS_Security_ParticipantSecurityAttributes *attributes,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_return_topic_sec_attributes)(
    dds_security_access_control *instance,
    const DDS_Security_TopicSecurityAttributes *attributes,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_return_datawriter_sec_attributes)(
    dds_security_access_control *instance,
    const DDS_Security_EndpointSecurityAttributes *attributes,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_return_datareader_sec_attributes)(
    dds_security_access_control *instance,
    const DDS_Security_EndpointSecurityAttributes *attributes,
    DDS_Security_SecurityException *ex);

typedef DDS_Security_boolean (*DDS_Security_access_control_return_permissions_handle)(
    dds_security_access_control *instance,
    DDS_Security_PermissionsHandle permissions_handle,
    DDS_Security_SecurityException *ex);

struct dds_security_access_control
{
  struct ddsi_domaingv *gv;

  DDS_Security_access_control_validate_local_permissions validate_local_permissions;
  DDS_Security_access_control_validate_remote_permissions validate_remote_permissions;
  DDS_Security_access_control_check_create_participant check_create_participant;
  DDS_Security_access_control_check_create_datawriter check_create_datawriter;
  DDS_Security_access_control_check_create_datareader check_create_datareader;
  DDS_Security_access_control_check_create_topic check_create_topic;
  DDS_Security_access_control_check_local_datawriter_register_instance check_local_datawriter_register_instance;
  DDS_Security_access_control_check_local_datawriter_dispose_instance check_local_datawriter_dispose_instance;
  DDS_Security_access_control_check_remote_participant check_remote_participant;
  DDS_Security_access_control_check_remote_datawriter check_remote_datawriter;
  DDS_Security_access_control_check_remote_datareader check_remote_datareader;
  DDS_Security_access_control_check_remote_topic check_remote_topic;
  DDS_Security_access_control_check_local_datawriter_match check_local_datawriter_match;
  DDS_Security_access_control_check_local_datareader_match check_local_datareader_match;
  DDS_Security_access_control_check_remote_datawriter_register_instance check_remote_datawriter_register_instance;
  DDS_Security_access_control_check_remote_datawriter_dispose_instance check_remote_datawriter_dispose_instance;
  DDS_Security_access_control_get_permissions_token get_permissions_token;
  DDS_Security_access_control_get_permissions_credential_token get_permissions_credential_token;
  DDS_Security_access_control_set_listener set_listener;
  DDS_Security_access_control_return_permissions_token return_permissions_token;
  DDS_Security_access_control_return_permissions_credential_token return_permissions_credential_token;
  DDS_Security_access_control_get_participant_sec_attributes get_participant_sec_attributes;
  DDS_Security_access_control_get_topic_sec_attributes get_topic_sec_attributes;
  DDS_Security_access_control_get_datawriter_sec_attributes get_datawriter_sec_attributes;
  DDS_Security_access_control_get_datareader_sec_attributes get_datareader_sec_attributes;
  DDS_Security_access_control_return_participant_sec_attributes return_participant_sec_attributes;
  DDS_Security_access_control_return_topic_sec_attributes return_topic_sec_attributes;
  DDS_Security_access_control_return_datawriter_sec_attributes return_datawriter_sec_attributes;
  DDS_Security_access_control_return_datareader_sec_attributes return_datareader_sec_attributes;
  DDS_Security_access_control_return_permissions_handle return_permissions_handle;
};

#if defined(__cplusplus)
}
#endif

#endif /* DDS_SECURITY_ACCESS_CONTROL_API_H */
