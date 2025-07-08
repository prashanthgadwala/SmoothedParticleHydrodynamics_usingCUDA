#pragma once

#include <vislab/core/types.hpp>
#include <memory>

namespace vislab
{
    class Actor;
}

namespace physsim
{
    /**
     * @brief Class that represents the state of a rigid body.
     */
    class RigidBody
    {
    public:
        /**
         * @brief Identifies whether a rigid body is static or dynamic.
         */
        enum class EType
        {
            /**
             * @brief Remains in a fixed position.
             */
            Static,

            /**
             * @brief Is allowed to move.
             */
            Dynamic
        };

        /**
         * @brief Constructor that initializes a unit body at the origin.
         */
        RigidBody();

        /**
         * @brief Gets the position of the center of mass in world space.
         * @return Position of the center of mass in world space.
         */
        const Eigen::Vector3d& position() const;

        /**
         * @brief Sets the position of the center of mass in world space.
         * @param x Position of the center of mass in world space.
         */
        void setPosition(const Eigen::Vector3d& x);

        /**
         * @brief Gets the rotation quaternion of the center of mass to world space.
         * @return Rotation quaternion of the center of mass to world space.
         */
        const Eigen::Quaterniond& rotation() const;

        /**
         * @brief Sets the rotation quaternion of the center of mass to world space.
         * @param q Rotation quaternion of the center of mass to world space.
         */
        void setRotation(const Eigen::Quaterniond& q);

        /**
         * @brief Gets the linear momentum of the center of mass.
         * @return Linear momentum of the center of mass.
         */
        Eigen::Vector3d linearMomentum() const;

        /**
         * @brief Gets the angular momentum of the center of mass.
         * @return Angular momentum of the center of mass.
         */
        Eigen::Vector3d angularMomentum() const;

        /**
         * @brief Gets the moment of inertia tensor in local body coordinates.
         * @return Moment of inertia tensor in local body coordinates.
         */
        const Eigen::Matrix3d& inertiaBody() const;

        /**
         * @brief Sets the moment of inertia tensor in local body coordinates.
         * @param Ib Moment of inertia tensor in local body coordinates.
         */
        void setInertiaBody(const Eigen::Matrix3d& Ib);

        /**
         * @brief Gets the inverse of the moment of interia tensor in local body coordinates.
         * @return Inverse of the moment of interia tensor in local body coordinates.
         */
        const Eigen::Matrix3d& inertiaBodyInverse() const;

        /**
         * @brief Gets the mass of the rigid body.
         * @return Mass of the rigid body.
         */
        const double& mass() const;

        /**
         * @brief Sets the mass of the rigid body.
         * @param m Mass of the rigid body.
         */
        void setMass(const double& m);

        /**
         * @brief Gets the inverse of the mass of the rigid body.
         * @return Inverse of the mass of the rigid body.
         */
        const double& massInverse() const;

        /**
         * @brief Gets the body force that is currently applied to the body.
         * @return Body force that is currently applied to the body.
         */
        const Eigen::Vector3d& force() const;

        /**
         * @brief Gets the body torque that is currently applied to the body.
         * @return Body torque that is currently applied to the body.
         */
        const Eigen::Vector3d& torque() const;

        /**
         * @brief Gets the type of object.
         * @return Type of object.
         */
        const EType& type() const;

        /**
         * @brief Sets the type of object.
         * @param type Type of object.
         */
        void setType(const EType& type);

        /**
         * @brief Gets the moment of inertia tensor in world coordinates.
         * @return q * Ib * q^-1
         */
        Eigen::Matrix3d inertiaWorld() const;

        /**
         * @brief Gets the inverse of the moment of inertia tensor in world coordinates.
         * @return q * Ib_inv * q^-1
         */
        Eigen::Matrix3d inertiaWorldInverse() const;

        /**
         * @brief Gets the linear velocity of the center of mass.
         * @return m_inv * p
         */
        const Eigen::Vector3d& linearVelocity() const;

        /**
         * @brief Sets the linear velocity of the center of mass.
         */
        void setLinearVelocity(const Eigen::Vector3d& v);

        /**
         * @brief Gets the angular velocity of the center of mass.
         * @return I_inv * l
         */
        const Eigen::Vector3d& angularVelocity() const;

        /**
         * @brief Sets the angular velocity of the center of mass.
         */
        void setAngularVelocity(const Eigen::Vector3d& w);

        /**
         * @brief Gets the velocity of a certain point on the body.
         * @param point Point on the body.
         * @return Velocity of the point.
         */
        Eigen::Vector3d velocity(const Eigen::Vector3d& point) const;

        /**
         * @brief Apply a force to center of mass.
         * @param f Force to apply to center of mass.
         */
        void applyForceToCenterOfMass(const Eigen::Vector3d& f);

        /**
         * @brief Apply force f at point p.
         * @param f Force to apply.
         * @param p Point to apply force at.
         */
        void applyForce(const Eigen::Vector3d& f, const Eigen::Vector3d& p);

        /**
         * @brief Apply a torque to the body.
         * @param t Torque to apply.
         */
        void applyTorque(const Eigen::Vector3d& t);

        /**
         * @brief Resets the force to zero.
         */
        void resetForce();

        /**
         * @brief Resets the torque to zero.
         */
        void resetTorque();

        /**
         * @brief Gets the actor that is associated with this rigid body.
         * @return Actor that is associated with this rigid body.
         */
        std::shared_ptr<const vislab::Actor> actor() const;

        /**
         * @brief Sets the actor that is associated with this rigid body.
         * @param actor Actor that is associated with this rigid body.
         */
        void setActor(std::shared_ptr<vislab::Actor> actor);

        /**
         * @brief Gets the uniform scaling of the shape.
         */
        const double& scale() const;

        /**
         * @brief Sets the uniform scaling of the shape.
         */
        void setScale(const double& scale);

        /**
         * @brief Gets the bounding box of the object in world space.
         * @return World-space bounding box.
        */
        Eigen::AlignedBox3d worldBounds() const;

    private:
        /**
         * @brief Updates the transformation of the associated shape.
         */
        void updateTransform();

        /**
         * @brief Position of the center of mass in world space.
         */
        Eigen::Vector3d mPosition;

        /**
         * @brief Rotation quaternion of the center of mass to world space.
         */
        Eigen::Quaterniond mRotation;

        /**
         * @brief Gets the linear velocity of the center of mass.
         */
        Eigen::Vector3d mLinearVelocity;

        /**
         * @brief Gets the angular velocity of the center of mass.
         */
        Eigen::Vector3d mAngularVelocity;

        /**
         * @brief Moment of inertia tensor in local body coordinates.
         */
        Eigen::Matrix3d mInertiaBody;

        /**
         * @brief Inverse of the moment of interia tensor in local body coordinates.
         */
        Eigen::Matrix3d mInertiaBodyInverse;

        /**
         * @brief Mass of the rigid body.
         */
        double mMass;

        /**
         * @brief Inverse of the mass of the rigid body.
         */
        double mMassInverse;

        /**
         * @brief Body force that is currently applied to the body.
         */
        Eigen::Vector3d mForce;

        /**
         * @brief Body torque that is currently applied to the body.
         */
        Eigen::Vector3d mTorque;

        /**
         * @brief Type of object.
         */
        EType mType;

        /**
         * @brief Actor that is associated with this rigid body.
         */
        std::shared_ptr<vislab::Actor> mActor;

        /**
         * @brief Uniform scaling of the shape.
         */
        double mScale;
    };

    /**
     * @brief The scalar multiplication operator is usually not defined for the quaternion. For convenience, we define it here.
     * @param scalar Scalar to multiply with.
     * @param q Quaterion to multiply.
     * @return Component-wise multiplication.
     */
    inline static Eigen::Quaterniond operator*(double scalar, const Eigen::Quaterniond& q)
    {
        return Eigen::Quaterniond(scalar * q.w(), scalar * q.x(), scalar * q.y(), scalar * q.z());
    }

    /**
     * @brief The quaterion addition operator is usually not defined for the quaternion. For convenience, we define it here.
     * @param p First summand.
     * @param q Second summand.
     * @return Component-wise addition.
     */
    inline static Eigen::Quaterniond operator+(const Eigen::Quaterniond& p, const Eigen::Quaterniond& q)
    {
        return Eigen::Quaterniond(p.w() + q.w(), p.x() + q.x(), p.y() + q.y(), p.z() + q.z());
    }

    /**
     * @brief Forms a skew symmetric matrix from a given coefficient vector.
     * @param a Coefficients that go on the off-diagonal elements of the skew-symmetric matrix.
     * @return Skew-symmetric matrix.
     */
    inline static Eigen::Matrix3d skew(const Eigen::Vector3d& a)
    {
        Eigen::Matrix3d s;
        s << 0, -a.z(), a.y(), a.z(), 0, -a.x(), -a.y(), a.x(), 0;
        return s;
    }
}
