#include <init_vislab.hpp>

#include "physsim_window.hpp"
#include "simulation.hpp"

#include <imgui.h>
#include <vislab/geometry/points.hpp>
#include <vislab/graphics/actor.hpp>
#include <vislab/graphics/colormap_texture.hpp>
#include <vislab/graphics/const_texture.hpp>
#include <vislab/graphics/diffuse_bsdf.hpp>
#include <vislab/graphics/perspective_camera.hpp>
#include <vislab/graphics/point_light.hpp>
#include <vislab/graphics/rectangle_geometry.hpp>
#include <vislab/graphics/scene.hpp>
#include <vislab/graphics/sphere_geometry.hpp>
#include <vislab/graphics/transform.hpp>

#include <vislab/core/array.hpp>

#include "nearest_neighbors.hpp"
#include "cuda_sph_simulation.hpp"

#include <random>
#include <iostream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace vislab;

namespace physsim
{
    /**
     * @brief Performance timer for comparing CPU vs GPU performance
     */
    class PerformanceTimer {
    public:
        void start() {
            start_time = std::chrono::high_resolution_clock::now();
        }
        
        double stop() {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            return duration.count() / 1000.0; // Return milliseconds
        }
        
    private:
        std::chrono::high_resolution_clock::time_point start_time;
    };

    /**
     * @brief Simulation mode enum
     */
    enum class SimulationMode {
        OPENMP_ONLY,
        CUDA_ONLY,
        COMPARE_BOTH
    };
    /*
     * \brief Cubic spline kernel from https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/master/SPlisHSPlasH/SPHKernels.h
     */
    class CubicKernel
    {
    protected:
        float m_radius;
        float m_k;
        float m_l;

    public:
        float getRadius() { return m_radius; }
        void setRadius(float val)
        {
            m_radius       = val;
            const float pi = static_cast<float>(EIGEN_PI);

            const float h3 = m_radius * m_radius * m_radius;
            m_k            = 8.f / (pi * h3);
            m_l            = 48.f / (pi * h3);
        }

    public:
        float W(const float r)
        {
            float res     = 0.0;
            const float q = r / m_radius;
            if (q <= 1.0)
            {
                if (q <= 0.5)
                {
                    const float q2 = q * q;
                    const float q3 = q2 * q;
                    res            = m_k * (6.f * q3 - 6.f * q2 + 1.f);
                }
                else
                {
                    res = m_k * (2.f * pow(1.f - q, 3.f));
                }
            }
            return res;
        }

        float W(const Eigen::Vector3f& r)
        {
            return W(r.norm());
        }

        Eigen::Vector3f gradW(const Eigen::Vector3f& r)
        {
            Eigen::Vector3f res;
            const float rl = r.norm();
            const float q  = rl / m_radius;
            if ((rl > 1.0e-9) && (q <= 1.0))
            {
                Eigen::Vector3f gradq = r / rl;
                gradq /= m_radius;
                if (q <= 0.5f)
                {
                    res = m_l * q * (3.f * q - 2.f) * gradq;
                }
                else
                {
                    const float factor = 1.f - q;
                    res                = m_l * (-factor * factor) * gradq;
                }
            }
            else
                res.setZero();

            return res;
        }
    };

    /**
     * @brief Implementation of weakly-coupled smooth particle hydrodynamics with Akinci boundary conditions.
     */
    class SmoothedParticleHydrodynamicsSimulation : public Simulation
    {
    public:
        /**
         * @brief Initializes the scene.
         */
        void init() override
        {
            // set extent of simulation domain
            mDomain = Eigen::AlignedBox3f(Eigen::Vector3f(-10, -5, 0), Eigen::Vector3f(10, 5, 10));

            // allocate buffers needed by the simulation
            mPositions         = std::make_shared<Array3f>("positions");
            mVelocities        = std::make_shared<Array3f>("velocities");
            mAccelerations     = std::make_shared<Array3f>("accelerations");
            mPreviousAccelerations = std::make_shared<Array3f>("previousAccelerations");
            mDensities         = std::make_shared<Array1f>("densities");
            mMasses            = std::make_shared<Array1f>("masses");
            mPressures         = std::make_shared<Array1f>("pressure");
            mBoundaryParticles = std::make_shared<Array3f>("boundaryParticles");
            mBoundaryMasses    = std::make_shared<Array1f>("boundaryMasses");

            // set simulation parameters for better performance testing
            mStepSize      = 5E-3;  // Smaller timestep for stability with more particles
            mGravity       = Eigen::Vector3f(0, 0, -9.8065f);
            mStiffness     = 1000;   // Much lower stiffness for debugging
            mViscosity     = 0.5f;   // Moderate viscosity for testing
            mExponent      = 3;      // Lower exponent for stability
            mRho0          = 1000;
            mSupportRadius = 0.5f;   // Smaller particles for finer detail
            mBoundaryDamping = 0.5f; // Damping factor for boundary collisions

            // Initialize CUDA simulation
            mSimulationMode = SimulationMode::COMPARE_BOTH; // Compare OpenMP vs CUDA
            mUseCuda = (mSimulationMode == SimulationMode::CUDA_ONLY || mSimulationMode == SimulationMode::COMPARE_BOTH);
            
            if (mUseCuda) {
                mCudaSimulation = std::make_unique<CudaSPHSimulation>();
                if (!mCudaSimulation->initialize(25000, 70000)) { // Increased limits: max particles, max boundary particles
                    std::cerr << "Failed to initialize CUDA simulation, falling back to OpenMP" << std::endl;
                    mUseCuda = false;
                    mSimulationMode = SimulationMode::OPENMP_ONLY;
                }
            }
            
            // Initialize performance tracking
            mTotalFrames = 0;
            mOpenMPTotalTime = 0.0;
            mCudaTotalTime = 0.0;

            // create ground plane
            auto rectActor = std::make_shared<Actor>("ground");
            rectActor->components.add(std::make_shared<RectangleGeometry>());
            rectActor->components.add(std::make_shared<DiffuseBSDF>(std::make_shared<ConstTexture>(Spectrum(1, 1, 1))));
            rectActor->components.add(std::make_shared<Transform>(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity(), Eigen::Vector3d(10., 5., 1.)));

            // create a point light
            auto lightActor = std::make_shared<Actor>("light");
            lightActor->components.add(std::make_shared<PointLight>(Spectrum(500.0, 500.0, 500.0)));
            lightActor->components.add(std::make_shared<Transform>(Eigen::Vector3d(5, 20, 10)));

            // create a camera
            auto cameraActor = std::make_shared<Actor>("camera");
            auto camera      = std::make_shared<PerspectiveCamera>();
            camera->setLookAt(Eigen::Vector3d(0, 0, 3));
            camera->setPosition(Eigen::Vector3d(2, 25, 6));
            camera->setUp(Eigen::Vector3d(0, 0, 1));
            camera->setNear(0.01);
            camera->setFar(100);
            cameraActor->components.add(camera);

            mSpheres                  = std::make_shared<Actor>("spheres");
            auto sphereGeometry       = std::make_shared<SphereGeometry>(0.15 * mSupportRadius); // Smaller spheres for smoother look
            sphereGeometry->positions = mPositions;
            sphereGeometry->data      = mPressures;
            mSpheres->components.add(sphereGeometry);
            auto colormapTexture                       = std::make_shared<ColormapTexture>();
            colormapTexture->transferFunction.minValue = 0;
            colormapTexture->transferFunction.maxValue = 100000;
            colormapTexture->transferFunction.values.clear();
            colormapTexture->transferFunction.values.insert(std::make_pair(0., Eigen::Vector4d(103 / 255., 169 / 255., 207 / 255., 1)));
            colormapTexture->transferFunction.values.insert(std::make_pair(0.25, Eigen::Vector4d(209 / 255., 229 / 255., 240 / 255., 1)));
            colormapTexture->transferFunction.values.insert(std::make_pair(0.5, Eigen::Vector4d(247 / 255., 247 / 255., 247 / 255., 1)));
            colormapTexture->transferFunction.values.insert(std::make_pair(0.75, Eigen::Vector4d(253 / 255., 219 / 255., 199 / 255., 1)));
            colormapTexture->transferFunction.values.insert(std::make_pair(1., Eigen::Vector4d(239 / 255., 138 / 255., 98 / 255., 1)));
            mSpheres->components.add(std::make_shared<DiffuseBSDF>(colormapTexture));
            mSpheres->components.add(std::make_shared<Transform>(Eigen::Vector3d(0, 0, 0)));

            // add elements to scene
            scene->actors.push_back(rectActor);
            scene->actors.push_back(lightActor);
            scene->actors.push_back(cameraActor);
            scene->actors.push_back(mSpheres);
        }

        /**
         * @brief Restarts the simulation.
         */
        void restart() override
        {
            // fill a portion of the domain with particles (uniform distribution with stratified sampling)
            Eigen::AlignedBox3f domainToFill(Eigen::Vector3f(-8, -4, 6), Eigen::Vector3f(-6, 4, 8)); // Smaller domain for reasonable particle count
            Eigen::Vector3i initialRes = ((domainToFill.max() - domainToFill.min()) / mSupportRadius * 1.5).cast<int>(); // Reduced density
            mPositions->setSize(initialRes.prod());
            mVelocities->setSize(initialRes.prod());
            mAccelerations->setSize(initialRes.prod());
            mPreviousAccelerations->setSize(initialRes.prod());
            mDensities->setSize(initialRes.prod());
            mPressures->setSize(initialRes.prod());
            std::default_random_engine rng;
            std::uniform_real_distribution<float> rnd(0, 1);
            for (int iz = 0; iz < initialRes.z(); ++iz)
                for (int iy = 0; iy < initialRes.y(); ++iy)
                    for (int ix = 0; ix < initialRes.x(); ++ix)
                    {
                        Eigen::Index linearIndex = (iz * initialRes.y() + iy) * initialRes.x() + ix;
                        Eigen::Vector3f relpos   = Eigen::Vector3f(
                            (ix + rnd(rng)) / (float)initialRes.x(),
                            (iy + rnd(rng)) / (float)initialRes.y(),
                            (iz + rnd(rng)) / (float)initialRes.z());
                        Eigen::Vector3f pos = domainToFill.min() + relpos.cwiseProduct((domainToFill.max() - domainToFill.min()));
                        mPositions->setValue(linearIndex, pos);
                        mVelocities->setValue(linearIndex, Eigen::Vector3f::Zero());
                        mAccelerations->setValue(linearIndex, Eigen::Vector3f::Zero());
                        mPreviousAccelerations->setValue(linearIndex, Eigen::Vector3f::Zero());
                    }

            // place boundary particles on the walls
            Eigen::Vector3i boundaryRes = ((mDomain.max() - mDomain.min()) / mSupportRadius * 2).cast<int>(); // Reduced boundary density
            mBoundaryParticles->setSize(2 * boundaryRes.x() * boundaryRes.y() + 2 * boundaryRes.y() * boundaryRes.z() + 2 * boundaryRes.x() * boundaryRes.z());
            Eigen::Index linearIndex = 0;
            for (int iy = 0; iy < boundaryRes.y(); ++iy)
                for (int ix = 0; ix < boundaryRes.x(); ++ix)
                {
                    Eigen::Vector3f relpos = Eigen::Vector3f(
                        ix / (boundaryRes.x() - 1.f),
                        iy / (boundaryRes.y() - 1.f),
                        0);
                    Eigen::Vector3f pos = mDomain.min() + relpos.cwiseProduct((mDomain.max() - mDomain.min()));
                    mBoundaryParticles->setValue(linearIndex++, pos);

                    relpos = Eigen::Vector3f(
                        ix / (boundaryRes.x() - 1.f),
                        iy / (boundaryRes.y() - 1.f),
                        1);
                    pos = mDomain.min() + relpos.cwiseProduct((mDomain.max() - mDomain.min()));
                    mBoundaryParticles->setValue(linearIndex++, pos);
                }
            for (int iz = 0; iz < boundaryRes.z(); ++iz)
                for (int iy = 0; iy < boundaryRes.y(); ++iy)
                {
                    Eigen::Vector3f relpos = Eigen::Vector3f(
                        0,
                        iy / (boundaryRes.y() - 1.f),
                        iz / (boundaryRes.z() - 1.f));
                    Eigen::Vector3f pos = mDomain.min() + relpos.cwiseProduct((mDomain.max() - mDomain.min()));
                    mBoundaryParticles->setValue(linearIndex++, pos);

                    relpos = Eigen::Vector3f(
                        1,
                        iy / (boundaryRes.y() - 1.f),
                        iz / (boundaryRes.z() - 1.f));
                    pos = mDomain.min() + relpos.cwiseProduct((mDomain.max() - mDomain.min()));
                    mBoundaryParticles->setValue(linearIndex++, pos);
                }
            for (int iz = 0; iz < boundaryRes.z(); ++iz)
                for (int ix = 0; ix < boundaryRes.x(); ++ix)
                {
                    Eigen::Vector3f relpos = Eigen::Vector3f(
                        ix / (boundaryRes.x() - 1.f),
                        0,
                        iz / (boundaryRes.z() - 1.f));
                    Eigen::Vector3f pos = mDomain.min() + relpos.cwiseProduct((mDomain.max() - mDomain.min()));
                    mBoundaryParticles->setValue(linearIndex++, pos);

                    relpos = Eigen::Vector3f(
                        ix / (boundaryRes.x() - 1.f),
                        1,
                        iz / (boundaryRes.z() - 1.f));
                    pos = mDomain.min() + relpos.cwiseProduct((mDomain.max() - mDomain.min()));
                    mBoundaryParticles->setValue(linearIndex++, pos);
                }

            // create a cubic spline kernel
            CubicKernel W;
            W.setRadius(mSupportRadius);

            // build nearest neighbor data structure for the boundary particles
            mBoundaryKNN = std::make_shared<NearestNeighbors3f>();
            mBoundaryKNN->setPoints(mBoundaryParticles);

            // compute the mass of boundary particles
            mBoundaryMasses->setSize(mBoundaryParticles->getSize());
            for (Eigen::Index i = 0; i < mBoundaryParticles->getSize(); ++i)
            {
                Eigen::Vector3f xk = mBoundaryParticles->getValue(i);
                float volk         = 0;
                NearestNeighbors3f::RadiusResult nnl;
                if (mBoundaryKNN->closestRadius(xk, mSupportRadius, nnl) != 0)
                    for (auto& nl : nnl)
                    {
                        Eigen::Vector3f xl  = mBoundaryParticles->getValue(nl.first);
                        Eigen::Vector3f xkl = xk - xl;
                        volk += W.W(xkl);
                    }
                float massk = mRho0 / volk;
                mBoundaryMasses->setValue(i, massk);
            }

            // compute the mass of domain particles
#if 0  // Disable variable mass calculation for now - use uniform masses
            mMasses->setSize(mPositions->getSize());
            auto KNN = std::make_shared<NearestNeighbors3f>();
            KNN->setPoints(mPositions);
            for (Eigen::Index i = 0; i < mPositions->getSize(); ++i)
            {
                Eigen::Vector3f xi = mPositions->getValue(i);
                float voli         = 0;
                NearestNeighbors3f::RadiusResult nn;
                if (KNN->closestRadius(xi, mSupportRadius, nn) != 0)
                    for (auto& nl : nn)
                    {
                        Eigen::Vector3f xk  = mPositions->getValue(nl.first);
                        Eigen::Vector3f xik = xi - xk;
                        voli += W.W(xik);
                    }
                NearestNeighbors3f::RadiusResult nnl;
                if (mBoundaryKNN->closestRadius(xi, mSupportRadius, nnl) != 0)
                    for (auto& nl : nnl)
                    {
                        Eigen::Vector3f xl  = mBoundaryParticles->getValue(nl.first);
                        Eigen::Vector3f xil = xi - xl;
                        voli += W.W(xil);
                    }
                float massi = mRho0 / voli;
                mMasses->setValue(i, massi);
            }
#else
            mMasses->setSize(mPositions->getSize());
            float volume = std::pow(mSupportRadius / 2, 3) * 0.8;
            for (Eigen::Index i = 0; i < mPositions->getSize(); ++i)
                mMasses->setValue(i, volume * mRho0);
#endif
        }

        /**
         * @brief Advances the simulation one time step forward.
         * @param elapsedTime Elapsed time in milliseconds during the last frame.
         * @param totalTime Total time in milliseconds since the beginning of the first frame.
         * @param timeStep Time step of the simulation. Restarts when resetting the simulation.
         */
        void advance(double elapsedTime, double totalTime, int64_t timeStep) override
        {
            double dt = mStepSize;
            
            if (mSimulationMode == SimulationMode::COMPARE_BOTH && mUseCuda && mCudaSimulation && mPositions->getSize() > 0) {
                // Performance comparison mode - run both OpenMP and GPU
                
                // Run OpenMP simulation
                mTimer.start();
                runOpenMPSimulation(dt);
                double openmpTime = mTimer.stop();
                
                // Run GPU simulation  
                mTimer.start();
                runGpuSimulation(dt);
                double cudaTime = mTimer.stop();
                
                // Update performance stats
                mTotalFrames++;
                mOpenMPTotalTime += openmpTime;
                mCudaTotalTime += cudaTime;
                
                // Print performance comparison every 100 frames
                if (mTotalFrames % 100 == 0) {
                    double avgOpenMPTime = mOpenMPTotalTime / mTotalFrames;
                    double avgCudaTime = mCudaTotalTime / mTotalFrames;
                    double speedup = avgOpenMPTime / avgCudaTime;
                    
                    std::cout << "Performance comparison (Frame " << mTotalFrames << "):" << std::endl;
                    std::cout << "  OpenMP average: " << avgOpenMPTime << " ms" << std::endl;
                    std::cout << "  CUDA average: " << avgCudaTime << " ms" << std::endl;
                    std::cout << "  CUDA speedup: " << speedup << "x" << std::endl;
                    std::cout << "  Particles: " << mPositions->getSize() << std::endl << std::endl;
                }
                
            } else if (mUseCuda && mCudaSimulation && mPositions->getSize() > 0) {
                // CUDA only mode
                runGpuSimulation(dt);
                
            } else {
                // OpenMP only mode
                runOpenMPSimulation(dt);
            }
            
            // notify that the geometry has changed
            mSpheres->components.get<Geometry>()->markChanged();
        }

    private:
        void runOpenMPSimulation(double dt) {
            // OpenMP simulation path (multi-threaded CPU)
            
            // build data structure for nearest neighbor search
            auto knn = std::make_shared<NearestNeighbors3f>();
            knn->setPoints(mPositions);
            CubicKernel W;
            W.setRadius(mSupportRadius);

            // density estimation (with OpenMP)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (Eigen::Index i = 0; i < mPositions->getSize(); ++i)
            {
                // get current position and velocity
                Eigen::Vector3f xi = mPositions->getValue(i);
                float rhoi         = 0;

                // compute density from boundary
                NearestNeighbors3f::RadiusResult nn;
                if (mBoundaryKNN->closestRadius(xi, mSupportRadius, nn) != 0)
                {
                    for (auto& n : nn)
                    {
                        float massk         = mBoundaryMasses->getValue(n.first).x();
                        Eigen::Vector3f xk  = mBoundaryParticles->getValue(n.first);
                        Eigen::Vector3f xik = xi - xk;
                        rhoi += massk * W.W(xik);
                    }
                }

                // compute density from interior particles
                if (knn->closestRadius(xi, mSupportRadius, nn) != 0)
                {
                    for (auto& n : nn)
                    {
                        float massj         = mMasses->getValue(n.first).x();
                        Eigen::Vector3f xj  = mPositions->getValue(n.first);
                        Eigen::Vector3f xij = xi - xj;
                        rhoi += massj * W.W(xij);
                    }
                }
                rhoi = std::max(rhoi, mRho0); // clamp
                mDensities->setValue(i, rhoi);
            }

            // pressure estimation based on EOS equation (WCSPH) - with stability improvements
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (Eigen::Index i = 0; i < mPositions->getSize(); ++i)
            {
                // Improved Tait equation with stability constraints
                float rhoi = mDensities->getValue(i).x();
                float density_ratio = rhoi / mRho0;
                
                // Clamp density ratio to prevent extreme pressures (key stability improvement)
                density_ratio = std::max(1.0f, std::min(density_ratio, 2.0f));
                
                float pi = mStiffness * (std::pow(density_ratio, mExponent) - 1.0f);
                pi = std::max(pi, 0.0f); // clamp to prevent negative pressure
                
                // Additional pressure clamping for stability
                float max_pressure = mStiffness * 5.0f; // Limit maximum pressure
                pi = std::min(pi, max_pressure);
                
                mPressures->setValue(i, pi);
            }

// pressure and viscosity acceleration
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (Eigen::Index i = 0; i < mPositions->getSize(); ++i)
            {
                // get current position and velocity
                Eigen::Vector3f xi = mPositions->getValue(i);
                Eigen::Vector3f vi = mVelocities->getValue(i);
                float pi           = mPressures->getValue(i).x();
                float rhoi         = mDensities->getValue(i).x();

                // find closest points
                Eigen::Vector3f ai_p(0, 0, 0);
                NearestNeighbors3f::RadiusResult nn;
                if (knn->closestRadius(xi, mSupportRadius, nn) != 0)
                {
                    for (auto& n : nn)
                    {
                        if (n.first == i) continue; // skip self
                        
                        Eigen::Vector3f xj = mPositions->getValue(n.first);
                        Eigen::Vector3f vj = mVelocities->getValue(n.first);
                        float pj           = mPressures->getValue(n.first).x();
                        float rhoj         = mDensities->getValue(n.first).x();
                        float massj        = mMasses->getValue(n.first).x();

                        Eigen::Vector3f xij = xi - xj;
                        Eigen::Vector3f vij = vi - vj;
                        
                        // Pressure acceleration (Equation 2): ai = Σj -mj * (pi/ρi² + pj/ρj²) * ∇Wij
                        Eigen::Vector3f gradW = W.gradW(xij);
                        float pressure_factor = massj * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj));
                        ai_p -= pressure_factor * gradW;

                        // Viscosity acceleration (Akinci style): ai = μ * Σj * mj * (xij · vij) / (ρj * |xij|²) * ∇Wij
                        float dot_product = xij.dot(vij);
                        float xij_norm_sq = xij.squaredNorm();
                        float eta = 0.01f * mSupportRadius * mSupportRadius; // small epsilon to avoid division by zero
                        float Pi_ij = mViscosity * dot_product / rhoj / (xij_norm_sq + eta);
                        ai_p += massj * Pi_ij * gradW;
                    }
                }

                // boundary handling
                if (mBoundaryKNN->closestRadius(xi, mSupportRadius, nn) != 0)
                {
                    for (auto& n : nn)
                    {
                        Eigen::Vector3f xk = mBoundaryParticles->getValue(n.first);
                        float massk        = mBoundaryMasses->getValue(n.first).x();
                        Eigen::Vector3f xik = xi - xk;
                        Eigen::Vector3f vij = vi;
                        
                        // Akinci boundary pressure acceleration (Equation 4): ai = Σk -mk * (pi/ρi² + pi/ρ0²) * ∇Wik
                        Eigen::Vector3f gradW = W.gradW(xik);
                        float pressure_factor = massk * (pi / (rhoi * rhoi) + pi / (mRho0 * mRho0));
                        ai_p -= pressure_factor * gradW;

                        // Akinci boundary viscosity acceleration: ai = μ * Σk * mk * (xik · vik) / (ρ0 * |xik|²) * ∇Wik
                        Eigen::Vector3f vik = vi; // boundary velocity is zero, so vi - 0 = vi
                        float dot_product = xik.dot(vik);
                        float xik_norm_sq = xik.squaredNorm();
                        float eta = 0.01f * mSupportRadius * mSupportRadius; // small epsilon to avoid division by zero
                        float Pi_ik = mViscosity * dot_product / (mRho0) / (xik_norm_sq + eta);
                        ai_p += massk * Pi_ik * gradW;
                    }
                }

                // store acceleration
                mAccelerations->setValue(i, ai_p);
            }

            // advance particles using Symplectic Euler (more stable than Verlet for SPH)
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (Eigen::Index i = 0; i < mPositions->getSize(); ++i)
            {
                // get current position, velocity, and accelerations
                Eigen::Vector3f xi    = mPositions->getValue(i);
                Eigen::Vector3f vi    = mVelocities->getValue(i);
                Eigen::Vector3f ai_p  = mAccelerations->getValue(i);
                Eigen::Vector3f ai_ext = mGravity; // External forces
                
                // Total acceleration
                Eigen::Vector3f ai_total = ai_p + ai_ext;
                
                // Apply acceleration clamping to prevent exploding particles
                float max_acceleration = 100.0f; // Reasonable limit for SPH
                if (ai_total.norm() > max_acceleration) {
                    ai_total = ai_total.normalized() * max_acceleration;
                }

                // Symplectic Euler integration (more stable than Verlet for strong forces)
                // Update velocity first
                Eigen::Vector3f vi_new = vi + ai_total * dt;
                
                // Then update position with new velocity
                Eigen::Vector3f xi_new = xi + vi_new * dt;

                // write result
                mPositions->setValue(i, xi_new);
                mVelocities->setValue(i, vi_new);
            }
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (Eigen::Index i = 0; i < mPositions->getSize(); ++i)
            {
                Eigen::Vector3f pos = mPositions->getValue(i);
                Eigen::Vector3f vel = mVelocities->getValue(i);
                bool collision = false;

                // Check and enforce domain boundaries
                const float epsilon = 0.001f;
                
                if (pos.x() < mDomain.min().x()) {
                    pos.x() = mDomain.min().x() + epsilon;
                    if (vel.x() < 0.0f) vel.x() = -vel.x() * mBoundaryDamping;
                    collision = true;
                }
                if (pos.x() > mDomain.max().x()) {
                    pos.x() = mDomain.max().x() - epsilon;
                    if (vel.x() > 0.0f) vel.x() = -vel.x() * mBoundaryDamping;
                    collision = true;
                }
                if (pos.y() < mDomain.min().y()) {
                    pos.y() = mDomain.min().y() + epsilon;
                    if (vel.y() < 0.0f) vel.y() = -vel.y() * mBoundaryDamping;
                    collision = true;
                }
                if (pos.y() > mDomain.max().y()) {
                    pos.y() = mDomain.max().y() - epsilon;
                    if (vel.y() > 0.0f) vel.y() = -vel.y() * mBoundaryDamping;
                    collision = true;
                }
                if (pos.z() < mDomain.min().z()) {
                    pos.z() = mDomain.min().z() + epsilon;
                    if (vel.z() < 0.0f) vel.z() = -vel.z() * mBoundaryDamping;
                    collision = true;
                }
                if (pos.z() > mDomain.max().z()) {
                    pos.z() = mDomain.max().z() - epsilon;
                    if (vel.z() > 0.0f) vel.z() = -vel.z() * mBoundaryDamping;
                    collision = true;
                }

                // Update position and velocity if collision occurred
                if (collision) {
                    mPositions->setValue(i, pos);
                    mVelocities->setValue(i, vel);
                }
            }
        }

        void runGpuSimulation(double dt) {
            // CUDA simulation path
            
            // Update CUDA simulation parameters
            mCudaSimulation->setParameters(mSupportRadius, mRho0, mStiffness, mExponent, mViscosity, mGravity);
            
            // Set boundary domain with current damping factor
            mCudaSimulation->setBoundaryDomain(
                mDomain.min().x(), mDomain.max().x(),
                mDomain.min().y(), mDomain.max().y(),
                mDomain.min().z(), mDomain.max().z(),
                mBoundaryDamping
            );
            
            // Upload current data to GPU
            mCudaSimulation->uploadParticleData(mPositions, mVelocities, mMasses);
            if (mBoundaryParticles->getSize() > 0) {
                mCudaSimulation->uploadBoundaryData(mBoundaryParticles, mBoundaryMasses);
            }
            
            // Run CUDA simulation step
            mCudaSimulation->step(static_cast<float>(dt));
            
            // Download results from GPU
            mCudaSimulation->downloadResults(mPositions, mVelocities, mDensities, mPressures);
        }

    public:

        /**
         * @brief Adds graphical user interface elements with imgui.
         */
        void gui() override
        {
            ImGui::PushItemWidth(100);

            double stepSizeMin = 1E-3, stepSizeMax = 1E-1;
            ImGui::SliderScalar("dt", ImGuiDataType_Double, &mStepSize, &stepSizeMin, &stepSizeMax);

            ImGui::SliderFloat("stiffness", &mStiffness, 0, 200000);
            ImGui::SliderFloat("exponent", &mExponent, 0, 10);
            ImGui::SliderFloat("viscosity", &mViscosity, 0, 50);
            
            // Boundary parameters
            ImGui::Separator();
            ImGui::Text("Boundary Parameters:");
            ImGui::SliderFloat("boundary damping", &mBoundaryDamping, 0.0f, 1.0f);
            
            // Display current particle count
            ImGui::Separator();
            ImGui::Text("Simulation Info:");
            ImGui::Text("Integration: Velocity Verlet");
            ImGui::Text("Particle Count: %d", (int)mPositions->getSize());
            ImGui::Text("Support Radius: %.3f", mSupportRadius);

            // Simulation mode selection
            ImGui::Separator();
            ImGui::Text("Simulation Mode:");
            int mode = static_cast<int>(mSimulationMode);
            const char* mode_names[] = { "OpenMP Only", "CUDA Only", "Compare Both" };
            if (ImGui::Combo("Mode", &mode, mode_names, 3)) {
                mSimulationMode = static_cast<SimulationMode>(mode);
                mUseCuda = (mSimulationMode == SimulationMode::CUDA_ONLY || mSimulationMode == SimulationMode::COMPARE_BOTH);
                
                // Reset performance counters when mode changes
                mTotalFrames = 0;
                mOpenMPTotalTime = 0.0;
                mCudaTotalTime = 0.0;
            }

            // Performance display
            if (mSimulationMode == SimulationMode::COMPARE_BOTH && mTotalFrames > 0) {
                ImGui::Separator();
                ImGui::Text("Performance Comparison:");
                double avgOpenMPTime = mOpenMPTotalTime / mTotalFrames;
                double avgCudaTime = mCudaTotalTime / mTotalFrames;
                double speedup = avgOpenMPTime / avgCudaTime;
                
                ImGui::Text("OpenMP: %.2f ms", avgOpenMPTime);
                ImGui::Text("CUDA: %.2f ms", avgCudaTime);
                ImGui::Text("Speedup: %.1fx", speedup);
                ImGui::Text("Frames: %d", mTotalFrames);
            }

            ImGui::PopItemWidth();
        }

        ///**
        // * @brief Helper function to create a diffuse sphere.
        // * @param position Position of the sphere in world space.
        // * @param scale Scale of the sphere in world space.
        // * @param color Color of the sphere.
        // * @return Reference to the sphere shape that was created.
        // */
        // std::shared_ptr<Actor> addSphere(const Eigen::Vector3d& position, const double& scale, const Spectrum& color)
        //{
        //    // create a sphere and assign the bsdf
        //    auto actor = std::make_shared<Actor>();
        //    actor->components.add(std::make_shared<SphereGeometry>(scale));
        //    actor->components.add(std::make_shared<DiffuseBSDF>(std::make_shared<ConstTexture>(color)));
        //    actor->components.add(std::make_shared<Transform>(position));
        //    scene->actors.push_back(actor);
        //    return actor;
        //}

    private:
        /**
         * @brief Fluid simulation domain. We will assume that all particles will be constrained into a box.
         */
        Eigen::AlignedBox3f mDomain;

        /**
         * @brief Positions of all particles.
         */
        std::shared_ptr<vislab::Array3f> mPositions;

        /**
         * @brief Velocities of all particles.
         */
        std::shared_ptr<vislab::Array3f> mVelocities;

        /**
         * @brief Pressure accelerations of all particles.
         */
        std::shared_ptr<vislab::Array3f> mAccelerations;

        /**
         * @brief Previous accelerations for velocity Verlet integration.
         */
        std::shared_ptr<vislab::Array3f> mPreviousAccelerations;

        /**
         * @brief Densities of all particles.
         */
        std::shared_ptr<vislab::Array1f> mDensities;

        /**
         * @brief Masses of all particles.
         */
        std::shared_ptr<vislab::Array1f> mMasses;

        /**
         * @brief Pressures of all particles.
         */
        std::shared_ptr<vislab::Array1f> mPressures;

        /**
         * @brief Nearest neighbor search data structure for the static boundary particles.
         */
        std::shared_ptr<NearestNeighbors3f> mBoundaryKNN;

        /**
         * @brief Set of static boundary particles.
         */
        std::shared_ptr<vislab::Array3f> mBoundaryParticles;

        /**
         * @brief Masses of all boundary particles.
         */
        std::shared_ptr<vislab::Array1f> mBoundaryMasses;

        /**
         * @brief Sphere geometries that are used for rendering.
         */
        std::shared_ptr<Actor> mSpheres;

        /**
         * @brief Support radius of the SPH kernels.
         */
        float mSupportRadius;

        /**
         * @brief Rest density.
         */
        float mRho0;
        /**
         * @brief Exponent of the tait equation.
         */
        float mExponent;

        /**
         * @brief Stiffness of the density to pressure conversion.
         */
        float mStiffness;

        /**
         * @brief Viscosity of the material.
         */
        float mViscosity;

        /**
         * @brief Integration step size.
         */
        double mStepSize;

        /**
         * @brief Gravity vector.
         */
        Eigen::Vector3f mGravity;

        /**
         * @brief Boundary damping factor for collisions.
         */
        float mBoundaryDamping;

        /**
         * @brief CUDA simulation instance.
         */
        std::unique_ptr<CudaSPHSimulation> mCudaSimulation;

        /**
         * @brief Flag to enable/disable CUDA simulation.
         */
        bool mUseCuda;

        /**
         * @brief Current simulation mode.
         */
        SimulationMode mSimulationMode;

        /**
         * @brief Performance timer.
         */
        PerformanceTimer mTimer;

        /**
         * @brief Performance tracking variables.
         */
        int mTotalFrames;
        double mOpenMPTotalTime;
        double mCudaTotalTime;
    };
}

int main()
{
    vislab::Init();

    physsim::PhyssimWindow window(
        1400,     // width - increased for more GUI space
        900,     // height - increased for more GUI space
        "Smoothed Particle Hydrodynamics - Velocity Verlet Integration", // title
        std::make_shared<physsim::SmoothedParticleHydrodynamicsSimulation>(),
        false // fullscreen
    );

    return window.run();
}
