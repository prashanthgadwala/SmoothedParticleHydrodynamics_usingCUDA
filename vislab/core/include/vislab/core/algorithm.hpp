#pragma once

#include "base_algorithm.hpp"

namespace vislab
{
    /**
     * @brief Using for interface algorithms.
     * @tparam TDerived Derived type.
     */
    template <typename TDerived>
    using InterfaceAlgorithm = Interface<TDerived, BaseAlgorithm<TDerived>>;

    /**
     * @brief Using for concrete algorithms.
     * @tparam TDerived Derived type.
     */
    template <typename TDerived>
    using ConcreteAlgorithm = Concrete<TDerived, BaseAlgorithm<TDerived>>;
}
