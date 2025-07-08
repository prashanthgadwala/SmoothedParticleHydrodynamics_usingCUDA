#pragma once

#include "components.hpp"
#include "resource.hpp"
#include "tag.hpp"
#include "tags.hpp"

#include <vislab/core/data.hpp>

namespace vislab
{
    /**
     * @brief Tag that can be added to a transform to indicate that the transform has changed.
     */
    class ActorChangedTag : public Concrete<ActorChangedTag, Tag>
    {
    };

    /**
     * @brief Actor that can be placed in the scene.
     */
    class Actor : public Concrete<Actor, Data, Resource>
    {
    public:
        /**
         * @brief Constructor.
         */
        Actor();

        /**
         * @brief Constructor.
         * @param name Name of the object.
         */
        Actor(const std::string& name);

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Name of this component.
         */
        std::string name;

        /**
         * @brief Components that are attached to this actor. Components might contain geometry, materials, physics, etc.
         */
        Components components;

        /**
         * @brief Tags that are attached to this actor. Tags are used to put flags onto an object, for example to inform different systems that a state has changed.
         */
        Tags tags;
    };
}
