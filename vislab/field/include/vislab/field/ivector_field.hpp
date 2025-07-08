#pragma once

#include "base_field.hpp"

#include "ivector_field_fwd.hpp"

// namespace vislab
//{
//     /**
//      * @brief Base class for vector fields in arbitrary dimension.
//      * @tparam VectorDimensions Number of components in the vectors.
//      * @tparam SpatialDimensions Number of spatial dimensions.
//      * @tparam DomainDimensions Total number of dimensions.
//      */
//     template <typename TVector, int64_t SpatialDimensions, int64_t DomainDimensions>
//     class IVectorField : public BaseField<TVector, SpatialDimensions, DomainDimensions>
//     {
//         VISLAB_INTERFACE(IVectorField, BaseField<TVector, SpatialDimensions, DomainDimensions>)
//
//     public:
//         /**
//          * @brief Type of the bounding box.
//          */
//         using BoundingBox = typename BaseField<TVector, SpatialDimensions, DomainDimensions>::BoundingBox;
//
//         /**
//          * @brief Type of the domain coordinates.
//          */
//         using DomainCoord = typename BaseField<TVector, SpatialDimensions, DomainDimensions>::DomainCoord;
//
//         /**
//          * @brief Type to specify the derivative degree. This type is used in the samplePartial function to specify the desired derivative of each dimension.
//          */
//         using DerivativeDegree = typename BaseField<TVector, SpatialDimensions, DomainDimensions>::DerivativeDegree;
//
//         /**
//          * @brief Type of values in the field.
//          */
//         using Value = typename BaseField<TVector, SpatialDimensions, DomainDimensions>::Value;
//
//         /**
//          * @brief Type of the vector-valued return types.
//          */
//         using Vector = Eigen::Matrix<double, SpatialDimensions, 1>;
//
//         /**
//          * @brief Type of the vector-valued return types.
//          */
//         using Matrix = Eigen::Matrix<double, SpatialDimensions, SpatialDimensions>;
//
//         /**
//          * @brief Constructor.
//          * @param domain Bounding box of the domain.
//          */
//         IVectorField(const BoundingBox& domain)
//             : BaseField<TVector, SpatialDimensions, DomainDimensions>(domain)
//         {
//         }
//
//         /**
//          * @brief Samples the magnitude of the vector field.
//          * @param coord Domain location.
//          * @return Magnitude of the vector field.
//          */
//         [[nodiscard]] double sampleMagnitude(const DomainCoord& coord) const
//         {
//             return this->sample(coord).norm();
//         }
//
//         /**
//          * @brief Samples the divergence of the vector field.
//          * @param coord Domain location.
//          * @return Divergence of the vector field.
//          */
//         [[nodiscard]] double sampleDivergence(const DomainCoord& coord) const
//         {
//             Matrix jacobian = sampleJacobian(coord);
//             return jacobian.trace();
//         }
//
//         /**
//          * @brief Samples the Jacobian matrix.
//          * @param coord Domain location.
//          * @return Jacobian matrix.
//          */
//         [[nodiscard]] Matrix sampleJacobian(const DomainCoord& coord) const
//         {
//             Matrix jacobian;
//             for (int c = 0; c < SpatialDimensions; ++c)
//             {
//                 DerivativeDegree derivative = DerivativeDegree::Zero();
//                 derivative[c]               = 1;
//                 jacobian.col(c)             = this->samplePartial(coord, derivative);
//             }
//             return jacobian;
//         }
//
//         /**
//          * @brief Samples the acceleration of the vector field.
//          * @param coord Domain location.
//          * @return Acceleration of the vector field.
//          */
//         [[nodiscard]] Vector sampleAcceleration(const DomainCoord& coord) const
//         {
//             Matrix J = sampleJacobian(coord);
//             Vector v = this->sample(coord);
//             if (this->IsSteady)
//             {
//                 return J * v;
//             }
//             else
//             {
//                 DerivativeDegree derivative   = DerivativeDegree::Zero();
//                 derivative[SpatialDimensions] = 1;
//                 return J * v + this->samplePartial(coord, derivative);
//             }
//         }
//
//         /**
//          * @brief Samples the strain rate tensor.
//          * @param coord Domain location.
//          * @return Strain rate tensor.
//          */
//         [[nodiscard]] Matrix sampleStrainRateTensor(const DomainCoord& coord) const
//         {
//             Matrix J = sampleJacobian(coord);
//             return (J + J.transpose()) / 2;
//         }
//
//         /**
//          * @brief Samples the vorticity tensor.
//          * @param coord Domain location.
//          * @return Vorticity tensor.
//          */
//         [[nodiscard]] Matrix sampleVorticityTensor(const DomainCoord& coord) const
//         {
//             Matrix J = sampleJacobian(coord);
//             return (J - J.transpose()) / 2;
//         }
//
//         /**
//          * @brief Samples the maximum imaginary part of the Jacobian, which is known as swirling strength.
//          * @param coord Domain location.
//          * @return Maximum imaginary part of the Jacobian, which is known as swirling strength.
//          */
//         [[nodiscard]] double sampleSwirlingStrength(const DomainCoord& coord) const
//         {
//             Matrix J         = sampleJacobian(coord);
//             Vector eigs_imag = J.eigenvalues().imag();
//             double imag      = 0;
//             for (int i = 0; i < SpatialDimensions; ++i)
//                 imag = std::max(imag, std::abs(eigs_imag[i]));
//             return imag;
//         }
//
//         /**
//          * @brief Computes a binary flag that determines whether the eigenvalues of the Jacobian are complex.
//          * @param coord Domain location.
//          * @return Binary flag that determines whether the eigenvalues of the Jacobian are complex.
//          */
//         [[nodiscard]] double sampleEigenvaluesAreComplex(const DomainCoord& coord) const
//         {
//             return sampleSwirlingStrength(coord) != 0 ? 1 : 0;
//         }
//
//         /**
//          * @brief Computes the real parts of the Jacobian matrix, sorted in ascending order.
//          * @param coord Domain location.
//          * @return Real parts of the Jacobian matrix, sorted in ascending order.
//          */
//         [[nodiscard]] Vector sampleEigenvaluesRealParts(const DomainCoord& coord) const
//         {
//             Matrix J    = sampleJacobian(coord);
//             Vector eigs = J.eigenvalues().real();
//             eigs.sortAscend();
//             return eigs;
//         }
//
//         /**
//          * @brief Samples the determinant of the Jacobian.
//          * @param coord Domain location.
//          * @return Determinant of the Jacobian.
//          */
//         [[nodiscard]] double sampleDeterminantJacobian(const DomainCoord& coord) const
//         {
//             return sampleJacobian(coord).determinant();
//         }
//     };
// }
