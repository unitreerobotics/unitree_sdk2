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

#ifndef DDS_SECURITY_API_TYPES_H
#define DDS_SECURITY_API_TYPES_H

#include "dds_security_api_defs.h"
#include "stdint.h"

#if defined (__cplusplus)
extern "C" {
#endif

/**************************************************************************
 *                                                                        *
 * Primitive types.                                                       *
 *                                                                        *
 **************************************************************************/
typedef int16_t                 DDS_Security_short;
typedef int32_t                 DDS_Security_long;
typedef int64_t                 DDS_Security_long_long;
typedef uint16_t                DDS_Security_unsigned_short;
typedef uint32_t                DDS_Security_unsigned_long;
typedef uint64_t                DDS_Security_unsigned_long_long;
typedef float                   DDS_Security_float;
typedef double                  DDS_Security_double;
typedef long double             DDS_Security_long_double;
typedef char                    DDS_Security_char;
typedef unsigned char           DDS_Security_octet;
typedef unsigned char           DDS_Security_boolean;
typedef DDS_Security_char *     DDS_Security_string;
typedef void *                  DDS_Security_Object;

/* Sequences */
typedef struct {
    DDS_Security_unsigned_long _maximum;
    DDS_Security_unsigned_long _length;
    DDS_Security_octet *_buffer;
} DDS_Security_OctetSeq;

typedef struct {
    DDS_Security_unsigned_long _maximum;
    DDS_Security_unsigned_long _length;
    DDS_Security_string *_buffer;
} DDS_Security_StringSeq;

typedef struct {
    DDS_Security_unsigned_long _maximum;
    DDS_Security_unsigned_long _length;
    DDS_Security_long_long *_buffer;
} DDS_Security_LongLongSeq;




/**************************************************************************
 *                                                                        *
 * Simple types.                                                          *
 *                                                                        *
 **************************************************************************/
typedef DDS_Security_long_long DDS_Security_IdentityHandle;
typedef DDS_Security_long_long DDS_Security_InstanceHandle;
typedef DDS_Security_long_long DDS_Security_HandshakeHandle;
typedef DDS_Security_long_long DDS_Security_SharedSecretHandle;
typedef DDS_Security_long_long DDS_Security_PermissionsHandle;
typedef DDS_Security_long_long DDS_Security_CryptoHandle;
typedef DDS_Security_long_long DDS_Security_ParticipantCryptoHandle;
typedef DDS_Security_long_long DDS_Security_DatawriterCryptoHandle;
typedef DDS_Security_long_long DDS_Security_DatareaderCryptoHandle;

typedef DDS_Security_long DDS_Security_DynamicData;

typedef DDS_Security_long DDS_Security_DomainId;  /* Valid values 0 <= id <= 230 */

typedef DDS_Security_long DDS_Security_Entity;

typedef DDS_Security_unsigned_long DDS_Security_BuiltinTopicKey_t[3];

typedef DDS_Security_octet DDS_Security_GuidPrefix_t[12];

/* Sequences */
typedef DDS_Security_LongLongSeq DDS_Security_CryptoHandleSeq;
typedef DDS_Security_LongLongSeq DDS_Security_ParticipantCryptoHandleSeq;
typedef DDS_Security_LongLongSeq DDS_Security_DatawriterCryptoHandleSeq;
typedef DDS_Security_LongLongSeq DDS_Security_DatareaderCryptoHandleSeq;




/**************************************************************************
 *                                                                        *
 * Simple structures.                                                     *
 *                                                                        *
 **************************************************************************/
typedef struct  {
    DDS_Security_string message;
    DDS_Security_long code;
    DDS_Security_long minor_code;
} DDS_Security_SecurityException;

typedef struct {
    DDS_Security_octet entityKey[3];
    DDS_Security_octet entityKind;
} DDS_Security_EntityId_t;

typedef struct {
    DDS_Security_GuidPrefix_t prefix;
    DDS_Security_EntityId_t entityId;
} DDS_Security_GUID_t;

typedef struct  {
    DDS_Security_long sec;
    DDS_Security_unsigned_long nanosec;
} DDS_Security_Duration_t;




/**************************************************************************
 *                                                                        *
 * Properties.                                                            *
 *                                                                        *
 **************************************************************************/
typedef struct {
    DDS_Security_string name;
    DDS_Security_string value;
    DDS_Security_boolean propagate;
} DDS_Security_Property_t;

typedef struct {
    DDS_Security_unsigned_long _maximum;
    DDS_Security_unsigned_long _length;
    DDS_Security_Property_t *_buffer;
} DDS_Security_PropertySeq;

typedef struct {
    DDS_Security_string name;
    DDS_Security_OctetSeq value;
    DDS_Security_boolean propagate;
} DDS_Security_BinaryProperty_t;

typedef struct {
    DDS_Security_unsigned_long _maximum;
    DDS_Security_unsigned_long _length;
    DDS_Security_BinaryProperty_t *_buffer;
} DDS_Security_BinaryPropertySeq;




/**************************************************************************
 *                                                                        *
 * DataTags.                                                              *
 *                                                                        *
 **************************************************************************/
typedef struct {
    DDS_Security_string name;
    DDS_Security_string value;
} DDS_Security_Tag;

typedef struct {
    DDS_Security_unsigned_long _maximum;
    DDS_Security_unsigned_long _length;
    DDS_Security_Tag *_buffer;
} DDS_Security_TagSeq;

typedef struct {
    DDS_Security_TagSeq tags;
} DDS_Security_DataTags;




/**************************************************************************
 *                                                                        *
 * Attributes.                                                            *
 *                                                                        *
 **************************************************************************/
typedef DDS_Security_unsigned_long DDS_Security_EndpointSecurityAttributesMask;
typedef DDS_Security_unsigned_long DDS_Security_PluginEndpointSecurityAttributesMask;

typedef DDS_Security_unsigned_long DDS_Security_ParticipantSecurityAttributesMask;
typedef DDS_Security_unsigned_long DDS_Security_PluginParticipantSecurityAttributesMask;

typedef struct {
    DDS_Security_boolean allow_unauthenticated_participants;
    DDS_Security_boolean is_access_protected;
    DDS_Security_boolean is_rtps_protected;
    DDS_Security_boolean is_discovery_protected;
    DDS_Security_boolean is_liveliness_protected;
    DDS_Security_ParticipantSecurityAttributesMask plugin_participant_attributes;
    DDS_Security_PropertySeq ac_endpoint_properties;
} DDS_Security_ParticipantSecurityAttributes;

typedef struct {
    DDS_Security_boolean is_read_protected;
    DDS_Security_boolean is_write_protected;
    DDS_Security_boolean is_discovery_protected;
    DDS_Security_boolean is_liveliness_protected;
} DDS_Security_TopicSecurityAttributes;

typedef struct {
    DDS_Security_boolean is_read_protected;
    DDS_Security_boolean is_write_protected;
    DDS_Security_boolean is_discovery_protected;
    DDS_Security_boolean is_liveliness_protected;
    DDS_Security_boolean is_submessage_protected;
    DDS_Security_boolean is_payload_protected;
    DDS_Security_boolean is_key_protected;
    DDS_Security_PluginEndpointSecurityAttributesMask plugin_endpoint_attributes;
    DDS_Security_PropertySeq ac_endpoint_properties;
} DDS_Security_EndpointSecurityAttributes;

typedef struct {
    DDS_Security_ParticipantSecurityAttributesMask participant_security_attributes;
    DDS_Security_PluginParticipantSecurityAttributesMask plugin_participant_security_attributes;
} DDS_Security_ParticipantSecurityInfo;

typedef struct {
    DDS_Security_EndpointSecurityAttributesMask endpoint_security_mask;
    DDS_Security_PluginEndpointSecurityAttributesMask plugin_endpoint_security_mask;
} DDS_Security_EndpointSecurityInfo;




/**************************************************************************
 *                                                                        *
 * Tokens.                                                                *
 *                                                                        *
 **************************************************************************/
typedef struct {
    DDS_Security_string class_id;
    DDS_Security_PropertySeq properties;
    DDS_Security_BinaryPropertySeq binary_properties;
} DDS_Security_DataHolder;

typedef struct {
    DDS_Security_unsigned_long _maximum;
    DDS_Security_unsigned_long _length;
    DDS_Security_DataHolder *_buffer;
} DDS_Security_DataHolderSeq;

typedef DDS_Security_DataHolder DDS_Security_Token;
typedef DDS_Security_DataHolder DDS_Security_MessageToken;
typedef DDS_Security_DataHolder DDS_Security_IdentityToken;
typedef DDS_Security_DataHolder DDS_Security_PermissionsToken;
typedef DDS_Security_DataHolder DDS_Security_IdentityStatusToken;
typedef DDS_Security_DataHolder DDS_Security_AuthRequestMessageToken;
typedef DDS_Security_DataHolder DDS_Security_HandshakeMessageToken;
typedef DDS_Security_DataHolder DDS_Security_AuthenticatedPeerCredentialToken;
typedef DDS_Security_DataHolder DDS_Security_PermissionsCredentialToken;
typedef DDS_Security_DataHolder DDS_Security_CryptoToken;
typedef DDS_Security_DataHolder DDS_Security_ParticipantCryptoToken;
typedef DDS_Security_DataHolder DDS_Security_DatawriterCryptoToken;
typedef DDS_Security_DataHolder DDS_Security_DatareaderCryptoToken;

typedef DDS_Security_DataHolderSeq DDS_Security_CryptoTokenSeq;

typedef DDS_Security_CryptoTokenSeq DDS_Security_ParticipantCryptoTokenSeq;
typedef DDS_Security_CryptoTokenSeq DDS_Security_DatareaderCryptoTokenSeq;
typedef DDS_Security_CryptoTokenSeq DDS_Security_DatawriterCryptoTokenSeq;




/**************************************************************************
 *                                                                        *
 * Policies.                                                              *
 *                                                                        *
 **************************************************************************/
typedef DDS_Security_DataTags DDS_Security_DataTagQosPolicy;

typedef struct {
    DDS_Security_PropertySeq value;
    DDS_Security_BinaryPropertySeq binary_value;
} DDS_Security_PropertyQosPolicy;

typedef struct  {
     DDS_Security_DurabilityQosPolicyKind kind;
} DDS_Security_DurabilityQosPolicy;

typedef struct {
    DDS_Security_Duration_t period;
} DDS_Security_DeadlineQosPolicy;

typedef struct {
    DDS_Security_Duration_t duration;
} DDS_Security_LatencyBudgetQosPolicy;

typedef struct {
    DDS_Security_OwnershipQosPolicyKind kind;
} DDS_Security_OwnershipQosPolicy;

typedef struct {
    DDS_Security_LivelinessQosPolicyKind kind;
    DDS_Security_Duration_t lease_duration;
} DDS_Security_LivelinessQosPolicy;

typedef struct {
    DDS_Security_ReliabilityQosPolicyKind kind;
    DDS_Security_Duration_t max_blocking_time;
    DDS_Security_boolean synchronous;
} DDS_Security_ReliabilityQosPolicy;

typedef struct {
    DDS_Security_Duration_t duration;
} DDS_Security_LifespanQosPolicy;

typedef struct {
    DDS_Security_DestinationOrderQosPolicyKind kind;
} DDS_Security_DestinationOrderQosPolicy;

typedef struct  {
    DDS_Security_OctetSeq value;
} DDS_Security_UserDataQosPolicy;

typedef struct {
    DDS_Security_long value;
} DDS_Security_OwnershipStrengthQosPolicy;

typedef struct {
    DDS_Security_PresentationQosPolicyAccessScopeKind access_scope;
    DDS_Security_boolean coherent_access;
    DDS_Security_boolean ordered_access;
} DDS_Security_PresentationQosPolicy;

typedef struct {
    DDS_Security_StringSeq name;
} DDS_Security_PartitionQosPolicy;

typedef struct {
    DDS_Security_OctetSeq value;
} DDS_Security_TopicDataQosPolicy;

typedef struct {
    DDS_Security_OctetSeq value;
} DDS_Security_GroupDataQosPolicy;

typedef struct {
    DDS_Security_Duration_t minimum_separation;
} DDS_Security_TimeBasedFilterQosPolicy;

typedef struct {
    DDS_Security_Duration_t service_cleanup_delay;
    DDS_Security_HistoryQosPolicyKind history_kind;
    DDS_Security_long history_depth;
    DDS_Security_long max_samples;
    DDS_Security_long max_instances;
    DDS_Security_long max_samples_per_instance;
} DDS_Security_DurabilityServiceQosPolicy;

typedef struct {
    DDS_Security_long value;
} DDS_Security_TransportPriorityQosPolicy;

typedef struct {
    DDS_Security_HistoryQosPolicyKind kind;
    DDS_Security_long depth;
} DDS_Security_HistoryQosPolicy;

typedef struct {
    DDS_Security_long max_samples;
    DDS_Security_long max_instances;
    DDS_Security_long max_samples_per_instance;
} DDS_Security_ResourceLimitsQosPolicy;




/**************************************************************************
 *                                                                        *
 * QoS.                                                                   *
 *                                                                        *
 **************************************************************************/
typedef struct {
    // Existing policies from the DDS specification are ignored.
    DDS_Security_PropertyQosPolicy property;
    DDS_Security_DataTagQosPolicy data_tags;
} DDS_Security_Qos;




/**************************************************************************
 *                                                                        *
 * Messages.                                                              *
 *                                                                        *
 **************************************************************************/
typedef struct  {
    DDS_Security_BuiltinTopicKey_t key;
    DDS_Security_BuiltinTopicKey_t participant_key;
    DDS_Security_string topic_name;
    DDS_Security_string type_name;
    DDS_Security_DurabilityQosPolicy durability;
    DDS_Security_DeadlineQosPolicy deadline;
    DDS_Security_LatencyBudgetQosPolicy latency_budget;
    DDS_Security_LivelinessQosPolicy liveliness;
    DDS_Security_ReliabilityQosPolicy reliability;
    DDS_Security_LifespanQosPolicy lifespan;
    DDS_Security_DestinationOrderQosPolicy destination_order;
    DDS_Security_UserDataQosPolicy user_data;
    DDS_Security_OwnershipQosPolicy ownership;
    DDS_Security_OwnershipStrengthQosPolicy ownership_strength;
    DDS_Security_PresentationQosPolicy presentation;
    DDS_Security_PartitionQosPolicy partition;
    DDS_Security_TopicDataQosPolicy topic_data;
    DDS_Security_GroupDataQosPolicy group_data;
    DDS_Security_EndpointSecurityInfo security_info;
    DDS_Security_DataTags data_tags;
} DDS_Security_PublicationBuiltinTopicDataSecure;

typedef struct {
    DDS_Security_BuiltinTopicKey_t key;
    DDS_Security_BuiltinTopicKey_t participant_key;
    DDS_Security_string topic_name;
    DDS_Security_string type_name;
    DDS_Security_DurabilityQosPolicy durability;
    DDS_Security_DeadlineQosPolicy deadline;
    DDS_Security_LatencyBudgetQosPolicy latency_budget;
    DDS_Security_LivelinessQosPolicy liveliness;
    DDS_Security_ReliabilityQosPolicy reliability;
    DDS_Security_OwnershipQosPolicy ownership;
    DDS_Security_DestinationOrderQosPolicy destination_order;
    DDS_Security_UserDataQosPolicy user_data;
    DDS_Security_TimeBasedFilterQosPolicy time_based_filter;
    DDS_Security_PresentationQosPolicy presentation;
    DDS_Security_PartitionQosPolicy partition;
    DDS_Security_TopicDataQosPolicy topic_data;
    DDS_Security_GroupDataQosPolicy group_data;
    DDS_Security_EndpointSecurityInfo security_info;
    DDS_Security_DataTags data_tags;
} DDS_Security_SubscriptionBuiltinTopicDataSecure;

typedef struct {
    DDS_Security_BuiltinTopicKey_t key;
    DDS_Security_string name;
    DDS_Security_string type_name;
    DDS_Security_DurabilityQosPolicy durability;
    DDS_Security_DurabilityServiceQosPolicy durability_service;
    DDS_Security_DeadlineQosPolicy deadline;
    DDS_Security_LatencyBudgetQosPolicy latency_budget;
    DDS_Security_LivelinessQosPolicy liveliness;
    DDS_Security_ReliabilityQosPolicy reliability;
    DDS_Security_TransportPriorityQosPolicy transport_priority;
    DDS_Security_LifespanQosPolicy lifespan;
    DDS_Security_DestinationOrderQosPolicy destination_order;
    DDS_Security_HistoryQosPolicy history;
    DDS_Security_ResourceLimitsQosPolicy resource_limits;
    DDS_Security_OwnershipQosPolicy ownership;
    DDS_Security_TopicDataQosPolicy topic_data;
} DDS_Security_TopicBuiltinTopicData;

typedef struct {
    DDS_Security_BuiltinTopicKey_t key;
    DDS_Security_UserDataQosPolicy user_data;
    DDS_Security_IdentityToken identity_token;
    DDS_Security_PermissionsToken permissions_token;
    DDS_Security_PropertyQosPolicy property;
    DDS_Security_ParticipantSecurityInfo security_info;
} DDS_Security_ParticipantBuiltinTopicData;

typedef struct {
    DDS_Security_BuiltinTopicKey_t key;
    DDS_Security_UserDataQosPolicy user_data;
    DDS_Security_IdentityToken identity_token;
    DDS_Security_PermissionsToken permissions_token;
    DDS_Security_PropertyQosPolicy property;
    DDS_Security_ParticipantSecurityInfo security_info;
    DDS_Security_IdentityStatusToken identity_status_token;
} DDS_Security_ParticipantBuiltinTopicDataSecure;



#if defined (__cplusplus)
}
#endif

#endif /* DDS_SECURITY_API_TYPES_H */

