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

#ifndef DDS_SECURITY_FSM_H
#define DDS_SECURITY_FSM_H

#include "dds/ddsrt/time.h"
#include "dds/ddsi/ddsi_domaingv.h"

#if defined (__cplusplus)
extern "C" {
#endif

#define DDS_SECURITY_FSM_EVENT_AUTO            (-1)
#define DDS_SECURITY_FSM_EVENT_TIMEOUT         (-2)
#define DDS_SECURITY_FSM_EVENT_DELETE          (-3)

struct dds_security_fsm;
struct dds_security_fsm_control;

typedef enum {
  DDS_SECURITY_FSM_DEBUG_ACT_DISPATCH,
  DDS_SECURITY_FSM_DEBUG_ACT_DISPATCH_DIRECT,
  DDS_SECURITY_FSM_DEBUG_ACT_HANDLING
} DDS_SECURITY_FSM_DEBUG_ACT;

/**
 * Template for user-defined state methods.
 * It is allowed to call dds_security_fsm_dispatch() from within a dispatch function.
 */
typedef void (*dds_security_fsm_action)(struct dds_security_fsm *fsm, void *arg);

/**
 * State struct
 *
 * - func    : optional user defined function, invoked by when reaching this state
 * - timeout : optional timeout which is controlled by the fsm
 */
typedef struct dds_security_fsm_state {
  const dds_security_fsm_action func;
  dds_duration_t timeout;
} dds_security_fsm_state;

/**
 * Template for user-defined debug methods.
 * It'll be called for every dispatched event, regardless of which state it
 * is in (which is also provided).
 * This can be used to get extra information about the behaviour of the
 * state machine.
 * It is not allowed to call any fsm API functions from within this
 * debug callback.
 */
typedef void (*dds_security_fsm_debug)(struct dds_security_fsm *fsm, DDS_SECURITY_FSM_DEBUG_ACT act, const dds_security_fsm_state *current, int event_id, void *arg);

/**
 * Transition definitions
 *
 * begin    : start state (to transition from)
 * event_id : indicate the event responsible for the transition
 * func     : user defined function, invoked during transition
 * end      : end state (to transition to)
 */
typedef struct dds_security_fsm_transition {
  const dds_security_fsm_state *begin;
  const int event_id;
  const dds_security_fsm_action func;
  const dds_security_fsm_state *end;
} dds_security_fsm_transition;


/**
 * Create a new fsm
 * Initializes a new fsm. Fsm does not start.
 *
 * @param transitions   array of transitions which the defines the functioning of the state machine
 * @param size          number of transitions
 * @param arg           Extra data to pass to the fsm. Will be passed to all user defined callback
 *                      methods.
 *
 * @return              Returns the new created state machine on success. Null on failure.
 */
DDS_EXPORT struct dds_security_fsm *
dds_security_fsm_create(struct dds_security_fsm_control *control, const dds_security_fsm_transition *transitions, uint32_t size, void *arg);


/**
 * Start a fsm
 * Starts the fsm, start with the firs transition
 *
 * @param fsm fsm to start.
 */
DDS_EXPORT void
dds_security_fsm_start(struct dds_security_fsm *fsm);

/**
 * Set an overall timeout for the given state machine
 * Will be monitoring the overall timeout of the given state machine,
 * invoking a user defined callback when the given timeout expires.
 * Timeout will be aborted upon a cleanup of the state machine.
 *
 * @param fsm       fsm to set the overall timeout for
 * @param func      user defined function which is called when the
 *                  overall timeout expires.
 * @param timeout   indicates the overall timeout
 */
DDS_EXPORT void
dds_security_fsm_set_timeout(struct dds_security_fsm *fsm, dds_security_fsm_action func, dds_time_t timeout);

/**
 * Set an debug callback for the given state machine.
 *
 * @param fsm       fsm to set the overall timeout for
 * @param func      user defined function which is called for every
 *                  event, whether being dispatched or actually
 *                  handled.
 */
DDS_EXPORT void
dds_security_fsm_set_debug(struct dds_security_fsm *fsm, dds_security_fsm_debug func);

/**
 * Dispatches the next event
 * Assignment for the state machine to transisiton to the next state.
 *
 * @param fsm       The state machine
 * @param event_id  Indicate where to transisition to (outcome of current state)
 * @param prio      Indicates if the event has to be scheduled with priority.
 */
DDS_EXPORT void
dds_security_fsm_dispatch(struct dds_security_fsm *fsm, int32_t event_id, bool prio);

/**
 * Retrieve the current state of a given state machine
 *
 * @param fsm       The state machine
 *
 * @return          true iff fsm not in initial or final state
 */
DDS_EXPORT bool
dds_security_fsm_running(struct dds_security_fsm *fsm);

/**
 * Stops the state machine.
 * Stops all running timeouts and events and cleaning all memory
 * related to this machine.
 *
 * When calling this from another thread, then it may block until
 * a possible concurrent event has finished. After this call, the
 * fsm may not be used anymore.
 *
 * When in the fsm action callback function context, this will
 * not block. It will garbage collect when the event has been
 * handled.
 *
 * @param fsm   The state machine to b stopped
 */
DDS_EXPORT void
dds_security_fsm_stop(struct dds_security_fsm *fsm);

/**
 * Free the state machine.
 * Stops all running timeouts and events and cleaning all memory
 * related to this machine.
 *
 * When calling this from another thread, then it may block until
 * a possible concurrent event has finished. After this call, the
 * fsm may not be used anymore.
 *
 * When in the fsm action callback function context, this will
 * not block. It will garbage collect when the event has been
 * handled.
 *
 * @param fsm   The state machine to be removed
 */
DDS_EXPORT void
dds_security_fsm_free(struct dds_security_fsm *fsm);

/**
 * Create a new fsm control context,
 * The fsm control context manages the global state of the fsm's created within
 * this context. The fsm control a thread to control the state machined allocated
 * to this control.
 *
 * @param gv  The global settings.
 *
 * @return Returns the new fsm control on success. Null on failure.
 */
DDS_EXPORT struct dds_security_fsm_control *
dds_security_fsm_control_create (struct ddsi_domaingv *gv);

/**
 * Frees the fsm control and the allocated fsm's.
 * A precondition is that the fsm control is stopped.
 *
 * @param control The fsm control to be freed.
 */
DDS_EXPORT void
dds_security_fsm_control_free(struct dds_security_fsm_control *control);

/**
 * Starts the thread that handles the events and timeouts associated
 * with the fsm that are managed by this fsm control.
 *
 * @param control The fsm control to be started.
 */
DDS_EXPORT dds_return_t
dds_security_fsm_control_start (struct dds_security_fsm_control *control, const char *name);

/**
 * Stops the thread that handles the events and timeouts.
 *
 * @param control The fsm control to be started.
 */
DDS_EXPORT void
dds_security_fsm_control_stop(struct dds_security_fsm_control *control);


#if defined (__cplusplus)
}
#endif

#endif /* DDS_SECURITY_FSM_H */
