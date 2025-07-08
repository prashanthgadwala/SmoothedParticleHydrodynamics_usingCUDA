#pragma once

namespace physsim
{
    class RigidBody;

    /**
     * @brief Explicit Euler integration for a rigid body.
     * @param body Rigid body to integrate.
     * @param stepSize Numerical integration step size.
     */
    void explicitEuler(RigidBody& body, double stepSize);

    /**
     * @brief Symplectic Euler integration for a rigid body.
     * @param body Rigid body to integrate.
     * @param stepSize Numerical integration step size.
     */
    void symplecticEuler(RigidBody& body, double stepSize);

    /**
     * @brief Implicit Euler integration for a rigid body.
     * @param body Rigid body to integrate.
     * @param stepSize Numerical integration step size.
     */
    void implicitEuler(RigidBody& body, double stepSize);
}
