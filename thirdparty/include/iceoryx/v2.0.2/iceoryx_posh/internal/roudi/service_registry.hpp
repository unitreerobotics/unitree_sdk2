// Copyright (c) 2019 by Robert Bosch GmbH. All rights reserved.
// Copyright (c) 2021 - 2022 by Apex.AI Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
#ifndef IOX_POSH_ROUDI_SERVICE_REGISTRY_HPP
#define IOX_POSH_ROUDI_SERVICE_REGISTRY_HPP

#include "iceoryx_hoofs/cxx/expected.hpp"
#include "iceoryx_hoofs/cxx/function_ref.hpp"
#include "iceoryx_hoofs/cxx/optional.hpp"
#include "iceoryx_hoofs/cxx/vector.hpp"
#include "iceoryx_posh/capro/service_description.hpp"
#include "iceoryx_posh/iceoryx_posh_types.hpp"


#include <cstdint>
#include <utility>

namespace iox
{
namespace roudi
{
class ServiceRegistry
{
  public:
    enum class Error
    {
        SERVICE_REGISTRY_FULL,
    };

    static constexpr uint32_t CAPACITY = iox::SERVICE_REGISTRY_CAPACITY;

    using ReferenceCounter_t = uint64_t;

    struct ServiceDescriptionEntry
    {
        ServiceDescriptionEntry(const capro::ServiceDescription& serviceDescription);

        capro::ServiceDescription serviceDescription;

        // note that we can have publishers and servers with the same ServiceDescription
        // and using the counters we save space
        ReferenceCounter_t publisherCount{0U};
        ReferenceCounter_t serverCount{0U};
    };

    /// @brief Adds a given publisher service description to registry
    /// @param[in] serviceDescription, service to be added
    /// @return ServiceRegistryError, error wrapped in cxx::expected
    cxx::expected<Error> addPublisher(const capro::ServiceDescription& serviceDescription) noexcept;

    /// @brief Removes a given publisher service description from registry if service is found,
    ///        in case of multiple occurrences only one occurrence is removed
    /// @param[in] serviceDescription, service to be removed
    void removePublisher(const capro::ServiceDescription& serviceDescription) noexcept;

    /// @brief Adds a given server service description to registry
    /// @param[in] serviceDescription, service to be added
    /// @return ServiceRegistryError, error wrapped in cxx::expected
    cxx::expected<Error> addServer(const capro::ServiceDescription& serviceDescription) noexcept;

    /// @brief Removes a given server service description from registry if service is found,
    ///        in case of multiple occurrences only one occurrence is removed
    /// @param[in] serviceDescription, service to be removed
    void removeServer(const capro::ServiceDescription& serviceDescription) noexcept;

    /// @brief Removes given service description from registry if service is found,
    ///        all occurrences are removed
    /// @param[in] serviceDescription, service to be removed
    void purge(const capro::ServiceDescription& serviceDescription) noexcept;

    /// @brief Searches for given service description in registry
    /// @param[in] service, string or wildcard (= iox::cxx::nullopt) to search for
    /// @param[in] instance, string or wildcard (= iox::cxx::nullopt) to search for
    /// @param[in] event, string or wildcard (= iox::cxx::nullopt) to search for
    /// @param[in] callable, callable to apply to each matching entry
    void find(const cxx::optional<capro::IdString_t>& service,
              const cxx::optional<capro::IdString_t>& instance,
              const cxx::optional<capro::IdString_t>& event,
              cxx::function_ref<void(const ServiceDescriptionEntry&)> callable) const noexcept;

    /// @brief Applies a callable to all entries
    /// @param[in] callable, callable to apply to each entry
    /// @note Can be used to obtain all entries or count them
    void forEach(cxx::function_ref<void(const ServiceDescriptionEntry&)> callable) const noexcept;

  private:
    using Entry_t = cxx::optional<ServiceDescriptionEntry>;
    using ServiceDescriptionContainer_t = cxx::vector<Entry_t, CAPACITY>;

    static constexpr uint32_t NO_INDEX = CAPACITY;

    ServiceDescriptionContainer_t m_serviceDescriptions;

    // store the last known free Index (if any is known)
    // we could use a queue (or stack) here since they are not optimal
    // for the filling pattern of a vector (prefer entries close to the front)
    uint32_t m_freeIndex{NO_INDEX};

  private:
    uint32_t findIndex(const capro::ServiceDescription& serviceDescription) const noexcept;


    cxx::expected<Error> add(const capro::ServiceDescription& serviceDescription,
                             ReferenceCounter_t ServiceDescriptionEntry::*count);
};

} // namespace roudi
} // namespace iox

#endif // IOX_POSH_ROUDI_SERVICE_REGISTRY_HPP
