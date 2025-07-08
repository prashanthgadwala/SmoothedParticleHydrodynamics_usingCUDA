#pragma once

#include "type.hpp"

#include "input_port.hpp"
#include "output_port.hpp"

namespace vislab
{
    template <typename TDerived>
    class FactoryWrapper
    {
    public:
        FactoryWrapper(meta::factory<TDerived> _factory)
            : factory(_factory)
        {
        }

        template <auto TMember>
        FactoryWrapper& member(const std::string& name, const std::string& description, const std::string& label)
        {
            std::hash<std::string> hash;
            // clang-format off
            factory.template data<TMember, meta::as_alias_t>(hash(name),
                meta::make_name_property(name),
                meta::make_property("Description", description),
                meta::make_property("Label", label));
            // clang-format on
            return *this;
        }

    private:
        meta::factory<TDerived> factory;
    };

    template <typename TDerived>
    FactoryWrapper<TDerived> reflect(const std::string& name, const std::vector<std::string>& aliases = {})
    {
        std::hash<std::string> hash;
        meta::factory<TDerived> m = aliases.empty() ? meta::named_reflect<TDerived>(name) : meta::named_reflect<TDerived>(name, aliases);
        if constexpr (std::is_constructible_v<TDerived>)
        {
            m = m.ctor();
        }
        if constexpr (std::is_base_of_v<Data, TDerived>)
        {
            meta::reflect<InputPort<TDerived>>().template base<IInputPort>().template func<&vislab::InputPort<TDerived>::getDataTypeInput>(hash("getDataTypeInput"));
            meta::reflect<OutputPort<TDerived>>().template base<IOutputPort>().template func<&vislab::OutputPort<TDerived>::getDataTypeOutput>(hash("getDataTypeOutput"));
        }
        return FactoryWrapper<TDerived>(m);
    }

    template <typename TDerived, typename TBase>
    FactoryWrapper<TDerived> reflect(const std::string& name, const std::vector<std::string>& aliases = {})
    {
        std::hash<std::string> hash;
        meta::factory<TDerived> m = aliases.empty() ? meta::named_reflect<TDerived>(name) : meta::named_reflect<TDerived>(name, aliases);
        m                         = m.template base<TBase>();
        if constexpr (std::is_constructible_v<TDerived>)
        {
            m = m.ctor();
        }
        if constexpr (std::is_base_of_v<Data, TDerived>)
        {
            meta::reflect<InputPort<TDerived>>().template base<IInputPort>().template func<&vislab::InputPort<TDerived>::getDataTypeInput>(hash("getDataTypeInput"));
            meta::reflect<OutputPort<TDerived>>().template base<IOutputPort>().template func<&vislab::OutputPort<TDerived>::getDataTypeOutput>(hash("getDataTypeOutput"));
        }
        return FactoryWrapper<TDerived>(m);
    }

    template <typename TDerived, typename TBase1, typename TBase2>
    FactoryWrapper<TDerived> reflect(const std::string& name, const std::vector<std::string>& aliases = {})
    {
        std::hash<std::string> hash;
        meta::factory<TDerived> m = aliases.empty() ? meta::named_reflect<TDerived>(name) : meta::named_reflect<TDerived>(name, aliases);
        m                         = m.template base<TBase1>();
        m                         = m.template base<TBase2>();
        if constexpr (std::is_constructible_v<TDerived>)
        {
            m = m.ctor();
        }
        if constexpr (std::is_base_of_v<Data, TDerived>)
        {
            meta::reflect<InputPort<TDerived>>().template base<IInputPort>().template func<&vislab::InputPort<TDerived>::getDataTypeInput>(hash("getDataTypeInput"));
            meta::reflect<OutputPort<TDerived>>().template base<IOutputPort>().template func<&vislab::OutputPort<TDerived>::getDataTypeOutput>(hash("getDataTypeOutput"));
        }
        return FactoryWrapper<TDerived>(m);
    }

    template <typename TDerived, typename TBase1, typename TBase2, typename TBase3>
    FactoryWrapper<TDerived> reflect(const std::string& name, const std::vector<std::string>& aliases = {})
    {
        std::hash<std::string> hash;
        meta::factory<TDerived> m = aliases.empty() ? meta::named_reflect<TDerived>(name) : meta::named_reflect<TDerived>(name, aliases);
        m                         = m.template base<TBase1>();
        m                         = m.template base<TBase2>();
        m                         = m.template base<TBase3>();
        if constexpr (std::is_constructible_v<TDerived>)
        {
            m = m.ctor();
        }
        if constexpr (std::is_base_of_v<Data, TDerived>)
        {
            meta::reflect<InputPort<TDerived>>().template base<IInputPort>().template func<&vislab::InputPort<TDerived>::getDataTypeInput>(hash("getDataTypeInput"));
            meta::reflect<OutputPort<TDerived>>().template base<IOutputPort>().template func<&vislab::OutputPort<TDerived>::getDataTypeOutput>(hash("getDataTypeOutput"));
        }
        return FactoryWrapper<TDerived>(m);
    }

    /*template <typename TDerived>
    FactoryWrapper<TDerived> reflect_algorithm(const std::string& name)
    {
        return FactoryWrapper<TDerived>(reflect<TDerived, vislab::IAlgorithm>(name));
    }*/
}
