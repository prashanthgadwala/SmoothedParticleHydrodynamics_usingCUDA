#pragma once

#include <vislab/core/iserializable.hpp>
#include <vislab/core/object.hpp>

#include "resource.hpp"
#include "tag.hpp"
#include "tags.hpp"

namespace vislab
{
    /**
     * @brief Tag that can be added to a transform to indicate that the transform has changed.
     */
    class ComponentChangedTag : public Concrete<ComponentChangedTag, Tag>
    {
    };

    /**
     * @brief Base class for components.
     */
    class Component : public Interface<Component, Object, ISerializable, Resource>
    {
    public:
        /**
         * @brief Checks whether the component has changed.
         * @return True if the component has changed.
         */
        [[nodiscard]] bool hasChanged() const
        {
            return tags.has<ComponentChangedTag>();
        }

        /**
         * @brief Marks that this component has changed. Derived systems such as renderers will use this information to update their internal state.
         */
        void markChanged()
        {
            tags.add<ComponentChangedTag>();
        }

        /**
         * @brief Marks that this component has not changed.
         */
        void markUnchanged()
        {
            tags.remove<ComponentChangedTag>();
        }

        /**
         * @brief Tags that are attached to this component. Tags are used to put flags onto an object, for example to inform different systems that a state has changed.
         */
        Tags tags;
    };
}
