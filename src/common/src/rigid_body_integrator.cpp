#include "rigid_body_integrator.hpp"

#include "rigid_body.hpp"

namespace physsim
{
    void explicitEuler(RigidBody& body, double stepSize)
    {
        // get current position and rotation of the body
        Eigen::Vector3d x    = body.position();
        Eigen::Quaterniond q = body.rotation();

        // get current linear and angular velocity of the body
        Eigen::Vector3d v = body.linearVelocity();
        Eigen::Vector3d w = body.angularVelocity();

        // update position of the body using the linear velocity and update body accordingly
        Eigen::Vector3d xnew = x + stepSize * v;
        body.setPosition(xnew);

        // quaternion-based angular velocity update of rotation and update body accordingly
        Eigen::Quaterniond wq(0, w.x(), w.y(), w.z());
        Eigen::Quaterniond qnew = (q + 0.5 * stepSize * wq * q).normalized();
        body.setRotation(qnew);

        // get current linear and angular momentum of body
        Eigen::Vector3d p = body.linearMomentum();
        Eigen::Vector3d l = body.angularMomentum();

        // get force and torque that are currently applied to body
        Eigen::Vector3d f = body.force();
        Eigen::Vector3d t = body.torque();

        // compute new linear momentum
        Eigen::Vector3d pnew = p + stepSize * f;

        // convert from linear momentum to linear velocity and update the body accordingly
        Eigen::Vector3d vnew = body.massInverse() * pnew;
        body.setLinearVelocity(vnew);

        // compute new angular momentum
        Eigen::Matrix3d I    = body.inertiaWorld();
        Eigen::Vector3d lnew = l + stepSize * (t - w.cross(I * w));

        // convert from angular momentum to angular velocity and update the body accordingly
        Eigen::Vector3d wnew = body.inertiaWorldInverse() * lnew;
        body.setAngularVelocity(wnew);
    }

    void symplecticEuler(RigidBody& body, double stepSize)
    {
        // TODO: get current position and rotation of the body

        Eigen::Vector3d x    = body.position();
        Eigen::Quaterniond q = body.rotation();

        // TODO: get current linear and angular momentum of body
        Eigen::Vector3d p = body.linearMomentum();
        Eigen::Vector3d v = body.linearVelocity();

        // TODO: get force and torque that are currently applied to body
        Eigen::Vector3d f = body.force();
        Eigen::Vector3d t = body.torque();

        // TODO: compute new linear momentum
        Eigen::Vector3d pnew = p + stepSize * f;

        // TODO: convert from linear momentum to linear velocity and update the body accordingly
        Eigen::Vector3d vnew = body.massInverse() * pnew;
        body.setLinearVelocity(vnew);

        // TODO: compute new angular momentum
        Eigen::Matrix3d I    = body.inertiaWorld();
        Eigen::Vector3d l    = body.angularMomentum();
        Eigen::Vector3d w    = body.angularVelocity();
        Eigen::Vector3d lnew = l + stepSize * (t - w.cross(I * w));

        // TODO: convert from angular momentum to angular velocity and update the body accordingly
        Eigen::Vector3d wnew = body.inertiaWorldInverse() * lnew;
        body.setAngularVelocity(wnew);


        // TODO: update position of the body using the linear velocity and update body accordingly
        Eigen::Vector3d xnew = x + stepSize * vnew;
        body.setPosition(xnew);

        // TODO: quaternion-based angular velocity update of rotation and update body accordingly
        Eigen::Quaterniond wq(0, wnew.x(), wnew.y(), wnew.z());
        Eigen::Quaterniond qnew = (q + 0.5 * stepSize * wq * q).normalized();
        body.setRotation(qnew);
    }


    void implicitEuler(RigidBody& body, double stepSize)
    {
        // Get current state
        Eigen::Vector3d x    = body.position();
        Eigen::Quaterniond q = body.rotation();
        Eigen::Vector3d p    = body.linearMomentum();
        Eigen::Vector3d v    = body.linearVelocity();
        Eigen::Vector3d f    = body.force();
        Eigen::Vector3d t    = body.torque();
    
        // Linear momentum (explicit Euler)
        Eigen::Vector3d pnew = p + stepSize * f;
        Eigen::Vector3d vnew = body.massInverse() * pnew;
        body.setLinearVelocity(vnew);
    
        // 1. Compute explicit torque update (excluding gyroscopic torque)
        Eigen::Matrix3d Ib = body.inertiaBody(); // inertia in body coords
        Eigen::Matrix3d Ib_inv = body.inertiaBodyInverse();
        Eigen::Quaterniond q_inv = q.conjugate();
    
        // Transform angular velocity to body coordinates
        Eigen::Vector3d w_world = body.angularVelocity();
        Eigen::Vector3d wb0 = q_inv * w_world;
    
        // Compute explicit part: Δωt = Δt * Ib_inv * (q_inv * t)
        Eigen::Vector3d tau_body = q_inv * t;
        Eigen::Vector3d delta_wb_explicit = stepSize * Ib_inv * tau_body;
    
        // Initial guess for ωb (body coords) after explicit update
        Eigen::Vector3d wb = wb0 + delta_wb_explicit;
    
        // 2. Newton step for implicit gyroscopic torque
        // f(wb) = Ib(wb - wb0) + dt * wb × (Ib * wb)
        Eigen::Vector3d fwb = Ib * (wb - wb0) + stepSize * wb.cross(Ib * wb);
    
        // Jacobian: J = Ib + dt * (skew(wb) * Ib - skew(Ib * wb))
        Eigen::Matrix3d skew_wb = skew(wb);
        Eigen::Matrix3d skew_Ib_wb = skew(Ib * wb);
        Eigen::Matrix3d J = Ib + stepSize * (skew_wb * Ib - skew_Ib_wb);
    
        // Solve for Newton step: J * delta_wb = -fwb
        Eigen::Vector3d delta_wb = J.colPivHouseholderQr().solve(-fwb);
    
        // Update ωb
        wb += delta_wb;
    
        // Transform back to world coordinates
        Eigen::Vector3d wnew = q * wb;
        body.setAngularVelocity(wnew);
    
        // Update angular momentum in world coordinates
        Eigen::Matrix3d Iw = body.inertiaWorld();
        Eigen::Vector3d lnew = Iw * wnew;
        body.angularMomentum() = lnew;
    
        // Update position
        Eigen::Vector3d xnew = x + stepSize * vnew;
        body.setPosition(xnew);
    
        // Quaternion-based rotation update
        Eigen::Quaterniond wq(0, wnew.x(), wnew.y(), wnew.z());
        Eigen::Quaterniond qnew = (q + 0.5 * stepSize * wq * q).normalized();
        body.setRotation(qnew);
    }

}
