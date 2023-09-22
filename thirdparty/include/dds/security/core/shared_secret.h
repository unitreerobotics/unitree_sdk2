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
#ifndef SRC_SECURITY_CORE_INCLUDE_SHARED_SECRET_HANDLE_H_
#define SRC_SECURITY_CORE_INCLUDE_SHARED_SECRET_HANDLE_H_

#include <stddef.h>
#include <stdint.h>

#include "dds/export.h"
#include "dds/security/dds_security_api.h"

typedef struct DDS_Security_SharedSecretHandleImpl {
  DDS_Security_octet* shared_secret;
  DDS_Security_long shared_secret_size;
  DDS_Security_octet challenge1[DDS_SECURITY_AUTHENTICATION_CHALLENGE_SIZE];
  DDS_Security_octet challenge2[DDS_SECURITY_AUTHENTICATION_CHALLENGE_SIZE];
} DDS_Security_SharedSecretHandleImpl;

DDS_EXPORT const DDS_Security_octet* get_challenge1_from_secret_handle (DDS_Security_SharedSecretHandle handle);
DDS_EXPORT const DDS_Security_octet* get_challenge2_from_secret_handle (DDS_Security_SharedSecretHandle handle);
DDS_EXPORT const DDS_Security_octet* get_secret_from_secret_handle (DDS_Security_SharedSecretHandle handle);
DDS_EXPORT size_t get_secret_size_from_secret_handle (DDS_Security_SharedSecretHandle handle);

#endif /* SRC_SECURITY_CORE_INCLUDE_SHARED_SECRET_H_ */
