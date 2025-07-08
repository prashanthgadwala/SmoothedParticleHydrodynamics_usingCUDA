#pragma once

#include "data.hpp"
#include "input_port.hpp"
#include "update_info.hpp"
#include "output_port.hpp"
#include "parameter.hpp"
#include "progress_info.hpp"

#include <future>
#include <iostream>

namespace vislab
{
    /**
     * @brief Interface for algorithms.
     */
    class IAlgorithm : public Interface<IAlgorithm, Object, ISerializable>
    {

    public:

        /**
         * @brief Default constructor.
         */
        IAlgorithm();

        /**
         * @brief Default destructor.
         */
        ~IAlgorithm() override = default;

        /**
         * @brief Gets a map containing all input ports, indexed by the unique name of the port.
         * @return Map of all input ports.
         */
        [[nodiscard]] virtual inline std::map<std::string, const IInputPort*> getInputPorts() const = 0;

        /**
         * @brief Gets a map containing all input ports, indexed by the unique name of the port.
         * @return Map of all input ports.
         */
        [[nodiscard]] virtual inline std::map<std::string, IInputPort*> getInputPorts() = 0;

        /**
         * @brief Gets a map containing all output ports, indexed by the unique name of the port.
         * @return Map of all output ports.
         */
        [[nodiscard]] virtual inline std::map<std::string, const IOutputPort*> getOutputPorts() const = 0;

        /**
         * @brief Gets a map containing all output ports, indexed by the unique name of the port.
         * @return Map of all output ports.
         */
        [[nodiscard]] virtual inline std::map<std::string, IOutputPort*> getOutputPorts() = 0;

        /**
         * @brief Gets a map containing all parameters, indexed by the unique name.
         * @return Map of all parameters.
         */
        [[nodiscard]] virtual inline std::map<std::string, const IParameter*> getParameters() const = 0;

        /**
         * @brief Gets a map containing all parameters, indexed by the unique name.
         * @return Map of all parameters.
         */
        [[nodiscard]] virtual inline std::map<std::string, IParameter*> getParameters() = 0;

        /**
         * @brief Invokes the execution of the algorithm.
         * @param progress Can be used to monitor the progress.
         * @return Information object with potential error message.
         */
        [[nodiscard]] UpdateInfo update(ProgressInfo& progress);

        /**
         * @brief Invokes the execution of the algorithm asynchronously.
         * @param progress Can be used to monitor the progress.
         * @return Information object with potential error message.
         */
        [[nodiscard]] std::future<UpdateInfo> updateAsync(ProgressInfo& progress);

        /**
         * @brief Invokes the execution of the algorithm.
         * @return Information object with potential error message.
         */
        [[nodiscard]] UpdateInfo update();

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Default allocates data in all output ports.
         */
        void allocateOutputs();

        /**
         * @brief Flag that determines whether this algorithm is currently active.
         */
        BoolParameter active;


    protected:

        /**
         * @brief Internal computation function
         * @param progress Optional progress info.
         * @return Information about the completion of the computation, including a potential error message.
         */
        [[nodiscard]] virtual UpdateInfo internalUpdate(ProgressInfo& progress) = 0;

        /**
         * @brief Helper function that retrieves all properties of a certain data type.
         * @tparam TPropertyType Type of properties to look for.
         * @tparam TDerivedType Derived type of the algorithm.
         * @param derived Reference to the derived algorithm, i.e., this.
         * @return Map of properties that derived from the TPropertyType.
         */
        template <typename TPropertyType, typename TDerivedType>
        [[nodiscard]] auto getProperties(TDerivedType* derived) const
        {
            std::hash<std::string_view> hash{};
            std::map<std::string, TPropertyType*> properties;
            Type metaPropertyType = meta::resolve<TPropertyType>();
            // FIXME: We store alias Types by their identifier name - does this matter?
            std::string metaPropertyTypeName = metaPropertyType.prop(hash("name")).value().template cast<std::string>();
            meta::resolve<TDerivedType>().data([&hash, &derived, &properties, &metaPropertyType, &metaPropertyTypeName](meta::data data)
                                               {
                                                   std::string test = data.prop(hash("name")).value(). template cast<std::string>();
                 Type metaMemberType = data.type();
                 if ((metaMemberType == metaPropertyType || metaMemberType.base(hash(metaPropertyTypeName))))
                 {
                    // FIXME: We store alias Types by their identifier name - does this matter?
                    std::string name{data.prop(hash("name")).value().template cast<std::string>()};
                    auto& dVal = *const_cast<typename std::remove_const<TDerivedType>::type*>(derived); // FIXME: This looks ugly - but is required and should be fine...
                    meta::any anyVal = data.get(dVal);
                    properties[name] = static_cast<TPropertyType*>(anyVal.data());
                 } });
            return properties;
        }
    };

}
