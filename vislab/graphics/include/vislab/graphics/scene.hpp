#pragma once

#include <vislab/core/data.hpp>

#include <vector>

namespace vislab
{
    class Actor;
    class Camera;

    /**
     * @brief Representation of the scene to render.
     */
    class Scene : public Concrete<Scene, Data>
    {
    public:
        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Collection of actors in the scene.
         */
        std::vector<std::shared_ptr<Actor>> actors;

        /**
         * @brief Gets an actor by name.
         * @param name Name of the actor to get.
         * @return Actor or nullptr if no actor has the requested name.
         */
        [[nodiscard]] std::shared_ptr<const Actor> getActor(const std::string& name) const;

        /**
         * @brief Gets an actor by name.
         * @param name Name of the actor to get.
         * @return Actor or nullptr if no actor has the requested name.
         */
        [[nodiscard]] std::shared_ptr<Actor> getActor(const std::string& name);
    };
}
