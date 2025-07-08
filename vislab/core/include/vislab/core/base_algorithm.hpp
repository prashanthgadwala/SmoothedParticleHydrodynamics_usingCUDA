#pragma once

#include "ialgorithm.hpp"

namespace vislab
{
    /**
     * @brief Base class for algorithms, which implements the functions for reflecting the input ports, output ports, and parameters.
     */
    template <typename TDerivedType>
    class BaseAlgorithm : public Interface<BaseAlgorithm<TDerivedType>, IAlgorithm>
    {
    public:
        /**
         * @brief Default constructor.
         */
        BaseAlgorithm() = default;

        /**
         * @brief Gets a map containing all input ports, indexed by the unique name of the port.
         * @return Map of all input ports.
         */
        [[nodiscard]] inline std::map<std::string, const vislab::IInputPort*> getInputPorts() const
        {
            return this->template getProperties<const vislab::IInputPort>(static_cast<const TDerivedType*>(this));
        }

        /**
         * @brief Gets a map containing all input ports, indexed by the unique name of the port.
         * @return Map of all input ports.
         */
        [[nodiscard]] inline std::map<std::string, vislab::IInputPort*> getInputPorts()
        {
            return this->template getProperties<vislab::IInputPort>(static_cast<TDerivedType*>(this));
        }

        /**
         * @brief Gets a map containing all output ports, indexed by the unique name of the port.
         * @return Map of all output ports.
         */
        [[nodiscard]] inline std::map<std::string, const vislab::IOutputPort*> getOutputPorts() const
        {
            return this->template getProperties<const vislab::IOutputPort>(static_cast<const TDerivedType*>(this));
        }

        /**
         * @brief Gets a map containing all output ports, indexed by the unique name of the port.
         * @return Map of all output ports.
         */
        [[nodiscard]] inline std::map<std::string, vislab::IOutputPort*> getOutputPorts()
        {
            return this->template getProperties<vislab::IOutputPort>(static_cast<TDerivedType*>(this));
        }

        /**
         * @brief Gets a map containing all parameters, indexed by the unique name.
         * @return Map of all parameters.
         */
        [[nodiscard]] inline std::map<std::string, const vislab::IParameter*> getParameters() const
        {
            return this->template getProperties<const vislab::IParameter>(static_cast<const TDerivedType*>(this));
        }

        /**
         * @brief Gets a map containing all parameters, indexed by the unique name.
         * @return Map of all parameters.
         */
        [[nodiscard]] inline std::map<std::string, vislab::IParameter*> getParameters()
        {
            return this->template getProperties<vislab::IParameter>(static_cast<TDerivedType*>(this));
        }
    };

}
