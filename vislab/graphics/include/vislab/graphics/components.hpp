#pragma once

#include <vislab/core/event.hpp>
#include <vislab/core/iserializable.hpp>
#include <vislab/core/object.hpp>

#include <vislab/graphics/component.hpp>

#include <unordered_map>

namespace vislab
{
    /**
     * @brief Implements a basic collection of components.
     */
    class Components : public Concrete<Components, Object, ISerializable>
    {
    public:
        /**
         * @brief Adds a component.
         * @tparam TComponent Type of the component to add.
         * @param component Component to add.
         */
        template <typename TComponentType>
        void add(std::shared_ptr<TComponentType> component)
        {
            if (has<TComponentType>())
                throw std::logic_error("There already is a component with this type!");
            mComponents.insert(std::make_pair(getIdentifier<TComponentType>(), component));
            // when adding a new component, we signal that this component has changed (it therefore gets updated)
            component->tags.template add<ComponentChangedTag>();
            onAdd.notify(this, component.get());
        }
        

        /**
         * @brief Removes a component.
         * @tparam TComponentType Type of the component to remove. Note that the type must match exactly.
         */
        template <typename TComponentType>
        void remove()
        {
            auto it = mComponents.find(getIdentifier<TComponentType>());
            if (it != mComponents.end())
            {
                std::shared_ptr<Component> to_remove = it->second;
                mComponents.erase(it->first);
                onRemove.notify(this, to_remove.get());
            }
        }

        /**
         * @brief Removes a certain tag from all components.
         * @tparam TTag Type of tag to remove.
         */
        template <typename TTag>
        void removeTag()
        {
            for (auto& component : mComponents)
            {
                component.second->tags.remove<TTag>();
            }
        }

        /**
         * @brief Tests if a component is present with a specific type is present or if there is a component that inherits from the requested type.
         * @tparam TComponentType Type of component to test.
         * @return True if the component exists.
         */
        template <typename TComponentType>
        [[nodiscard]] bool has() const
        {
            // get the identifier for the requested type
            std::size_t query_id = getIdentifier<TComponentType>();

            // is there a component which has exactly this type?
            if (mComponents.find(query_id) != mComponents.end())
                return true;

            // iterate all components and see if a base class was requested
            for (auto& id : mComponents)
            {
                if (meta::resolve(id.first).base(query_id).operator bool())
                {
                    return true;
                }
            }

            // neither is there a component with the requested type, nor is there a component that inherits from the requested type.
            return false;
        }

        /**
         * @brief Gets a typed component for a specific type.
         * @tparam TComponentType Type to retrieve.
         * @return The component or an empty shared_ptr if the component does not exist.
         */
        template <typename TComponentType>
        [[nodiscard]] std::shared_ptr<const TComponentType> get() const
        {
            // get the identifier for the requested type
            std::size_t query_id = getIdentifier<TComponentType>();

            // is there a component which has exactly this type?
            auto it = mComponents.find(query_id);
            if (it != mComponents.end())
            {
                return std::static_pointer_cast<TComponentType>(it->second);
            }

            // iterate all components and see if a base class was requested
            for (auto& id : mComponents)
            {
                if (meta::resolve(id.first).base(query_id).operator bool())
                {
                    return std::static_pointer_cast<TComponentType>(id.second);
                }
            }

            // neither is there a component with the requested type, nor is there a component that inherits from the requested type.
            return nullptr;
        }

        /**
         * @brief Gets a typed component for a specific type.
         * @tparam TComponentType Type to retrieve.
         * @return The component or an empty shared_ptr if the component does not exist.
         */
        template <typename TComponentType>
        [[nodiscard]] std::shared_ptr<TComponentType> get()
        {
            // get the identifier for the requested type
            std::size_t query_id = getIdentifier<TComponentType>();

            // is there a component which has exactly this type?
            auto it = mComponents.find(query_id);
            if (it != mComponents.end())
            {
                return std::static_pointer_cast<TComponentType>(it->second);
            }

            // iterate all components and see if a base class was requested
            for (auto& id : mComponents)
            {
                if (meta::resolve(id.first).base(query_id).operator bool())
                {
                    return std::static_pointer_cast<TComponentType>(id.second);
                }
            }

            // neither is there a component with the requested type, nor is there a component that inherits from the requested type.
            return nullptr;
        }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Event that is raised when a component was added.
         */
        TEvent<Components, Component> onAdd;

        /**
         * @brief Event that is raised when a component was removed.
         */
        TEvent<Components, Component> onRemove;

    private:
        /**
         * @brief Retrieves the unique identifier of a given type. This messages throws an std::logic_error if the type was not registered.
         * @tparam TComponentType Type to get the identifier from.
         * @return Unique identfier.
         */
        template <typename TComponentType>
        [[nodiscard]] std::size_t getIdentifier() const
        {
            std::size_t identifier = TComponentType::type().identifier();
            if (identifier == 0)
                throw std::logic_error("A type identifier has the numerical value zero. Did you forget to register the type in the reflection system?");
            return identifier;
        }

        /**
         * @brief Collection of components, indexed by their unique identifiers.
         */
        std::unordered_map<std::size_t, std::shared_ptr<Component>> mComponents;
    };
}
