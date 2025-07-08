//#include <vislab/graphics/bsdf.hpp>
//
//namespace vislab
//{
//    BSDF::BSDF(uint32_t flags)
//        : mFlags(flags)
//    {
//    }
//
//    std::pair<Spectrum, double> BSDF::evaluate_pdf(const SurfaceInteraction& si, const Eigen::Vector3d& wo) const
//    {
//        return { evaluate(si, wo), pdf(si, wo) };
//    }
//
//    std::tuple<Spectrum, double, BSDFSample, Spectrum> BSDF::evaluate_pdf_sample(const SurfaceInteraction& si, const Eigen::Vector3d& wo, const Eigen::Vector2d& sample1) const
//    {
//        auto [e_val, pdf_val]  = evaluate_pdf(si, wo);
//        auto [bs, bsdf_weight] = sample(si, sample1);
//        return { e_val, pdf_val, bs, bsdf_weight };
//    }
//
//    Spectrum BSDF::diffuseReflectance(const SurfaceInteraction& si) const
//    {
//        Eigen::Vector3d wo = Eigen::Vector3d(0.0, 0.0, 1.0);
//        return evaluate(si, wo) * EIGEN_PI;
//    }
//
//    uint32_t BSDF::flags() const { return mFlags; }
//
//    bool BSDF::needsDifferentials() const
//    {
//        return hasFlag(EBSDFFlag::NeedsDifferentials);
//    }
//
//    bool BSDF::hasFlag(const EBSDFFlag& flag) const
//    {
//        uint32_t f = (uint32_t)flag;
//        return (mFlags & f) != 0;
//    }
//}
