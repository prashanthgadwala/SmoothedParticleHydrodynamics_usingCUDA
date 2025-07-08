#pragma once

#include <memory>

namespace vislab
{
    class Scene;
}

namespace physsim
{
    /**
     * @brief Base class for simulations in the physsim course.
     */
    class Simulation
    {
    public:
        /**
         * @brief Initializes the scene.
         */
        virtual void init() = 0;

        /**
         * @brief Restarts the simulation.
         */
        virtual void restart() = 0;

        /**
         * @brief Advances the simulation one time step forward.
         * @param elapsedTime Elapsed time in milliseconds during the last frame.
         * @param totalTime Total time in milliseconds since the beginning of the first frame.
         * @param timeStep Time step of the simulation. Restarts when resetting the simulation.
         * @param timeStep Time step of the simulation. Restarts when resetting the simulation.
         */
        virtual void advance(double elapsedTime, double totalTime, int64_t timeStep) = 0;

        /**
         * @brief Adds graphical user interface elements.
         */
        virtual void gui() = 0;

        /**
         * @brief Reference of the scene, which contains all the geometry to render.
         */
        std::shared_ptr<vislab::Scene> scene;
    };
}
