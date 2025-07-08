#pragma once

#include <array>
#include <cstdint>

namespace vislab
{
    /**
     * @brief Type that is further specialized for a field, allowing for a compact specification of a derivative.
     * @tparam SpatialDimensions Number of spatial dimensions of the field.
     * @tparam Dimensions Total number of dimensions of the field (space+time).
     */
    template <int SpatialDimensions, int Dimensions>
    union BasePartial
    {
        /**
         * @brief Constructor from hash.
         * @param _hash Hash that specifies the derivative degree for all dimensions.
         */
        BasePartial(uint64_t _hash)
            : hash(_hash)
        {
        }

        /**
         * @brief Constructor from an array.
         * @param _degree Array that specifies the derivative degree for all dimensions.
         */
        BasePartial(std::array<uint8_t, Dimensions> _degree)
            : hash(0)
        {
            for (int i = 0; i < Dimensions; ++i)
                degree[i] = _degree[i];
        }

        uint64_t hash; /* Hash that can be used for comparison and switch cases. */

        /**
         * @brief Gets the derivative degree for a given dimension.
         * @param dimension Dimension to get the derivative degree for.
         * @return Derivative degree of given dimension.
         */
        uint8_t operator[](uint8_t dimension) const
        {
            return degree[dimension];
        }

        /**
         * @brief Gets the derivative degree for a given dimension.
         * @param dimension Dimension to get the derivative degree for.
         * @return Derivative degree of given dimension.
         */
        uint8_t& operator[](uint8_t dimension)
        {
            return degree[dimension];
        }

    private:
        /**
         * @brief Array of derivative degrees for up to 8 dimensions.
         */
        uint8_t degree[8];
    };

    /**
     * @brief Specialization of partial derivative specification for steady 2D fields (SpatialDimensions=2, Dimensions=2).
     */
    template <>
    union BasePartial<2, 2>
    {
        /**
         * @brief Constructor from hash.
         * @param _hash Hash that specifies the derivative degree for all dimensions.
         */
        BasePartial(uint64_t _hash)
            : hash(_hash)
        {
        }

        /**
         * @brief Constructor with individual partial derivative degrees.
         * @param _dx Degree of the x-partial derivative.
         * @param _dy Degree of the y-partial derivative.
         */
        BasePartial(uint8_t _dx, uint8_t _dy)
            : hash(0)
        {
            degreeDx = _dx;
            degreeDy = _dy;
        }

        uint64_t hash; /* Hash that can be used for comparison and switch cases. */
        struct
        {
            uint8_t degreeDx; /* Specifies the degree of the x-partial derivative. */
            uint8_t degreeDy; /* Specifies the degree of the y-partial derivative. */
        };

        static constexpr uint64_t c    = 0;               /* Hash of the function (no partial). */
        static constexpr uint64_t dx   = 1 << 0;          /* Hash of the dx partial derivative. */
        static constexpr uint64_t dy   = 1 << 8;          /* Hash of the dy partial derivative. */
        static constexpr uint64_t dxx  = 2 << 0;          /* Hash of the dxx partial derivative. */
        static constexpr uint64_t dxy  = 1 << 0 | 1 << 8; /* Hash of the dxy partial derivative. */
        static constexpr uint64_t dyy  = 2 << 8;          /* Hash of the dyy partial derivative. */
        static constexpr uint64_t dxxx = 3 << 0;          /* Hash of the dxxx partial derivative. */
        static constexpr uint64_t dxxy = 2 << 0 | 1 << 8; /* Hash of the dxxy partial derivative. */
        static constexpr uint64_t dxyy = 1 << 0 | 2 << 8; /* Hash of the dxyy partial derivative. */
        static constexpr uint64_t dyyy = 3 << 8;          /* Hash of the dyyy partial derivative. */

        /**
         * @brief Gets the derivative degree for a given dimension.
         * @param dimension Dimension to get the derivative degree for.
         * @return Derivative degree of given dimension.
         */
        uint8_t operator[](uint8_t dimension) const
        {
            return degree[dimension];
        }

        /**
         * @brief Gets the derivative degree for a given dimension.
         * @param dimension Dimension to get the derivative degree for.
         * @return Derivative degree of given dimension.
         */
        uint8_t& operator[](uint8_t dimension)
        {
            return degree[dimension];
        }

    private:
        /**
         * @brief Array of derivative degrees for up to 8 dimensions.
         */
        uint8_t degree[8];
    };

    /**
     * @brief Specialization of partial derivative specification for unsteady 2D fields (SpatialDimensions=2, Dimensions=3).
     */
    template <>
    union BasePartial<2, 3>
    {
    public:
        /**
         * @brief Constructor from hash.
         * @param _hash Hash that specifies the derivative degree for all dimensions.
         */
        BasePartial(uint64_t _hash)
            : hash(_hash)
        {
        }

        /**
         * @brief Constructor with individual partial derivative degrees.
         * @param _dx Degree of the x-partial derivative.
         * @param _dy Degree of the y-partial derivative.
         * @param _dt Degree of the t-partial derivative.
         */
        BasePartial(uint8_t _dx, uint8_t _dy, uint8_t _dt)
            : hash(0)
        {
            degreeDx = _dx;
            degreeDy = _dy;
            degreeDt = _dt;
        }

        uint64_t hash; /* Hash that can be used for comparison and switch cases. */
        struct
        {
            uint8_t degreeDx; /* Specifies the degree of the x-partial derivative. */
            uint8_t degreeDy; /* Specifies the degree of the y-partial derivative. */
            uint8_t degreeDt; /* Specifies the degree of the t-partial derivative. */
        };

        static constexpr uint64_t c    = 0;                         /* Hash of the function (no partial). */
        static constexpr uint64_t dx   = 1 << 0;                    /* Hash of the dx partial derivative. */
        static constexpr uint64_t dy   = 1 << 8;                    /* Hash of the dy partial derivative. */
        static constexpr uint64_t dt   = 1 << 16;                   /* Hash of the dt partial derivative. */
        static constexpr uint64_t dxx  = 2 << 0;                    /* Hash of the dxx partial derivative. */
        static constexpr uint64_t dxy  = 1 << 0 | 1 << 8;           /* Hash of the dxy partial derivative. */
        static constexpr uint64_t dxt  = 1 << 0 | 1 << 16;          /* Hash of the dxt partial derivative. */
        static constexpr uint64_t dyy  = 2 << 8;                    /* Hash of the dyy partial derivative. */
        static constexpr uint64_t dyt  = 1 << 8 | 1 << 16;          /* Hash of the dyt partial derivative. */
        static constexpr uint64_t dtt  = 2 << 16;                   /* Hash of the dtt partial derivative. */
        static constexpr uint64_t dxxx = 3 << 0;                    /* Hash of the dxxx partial derivative. */
        static constexpr uint64_t dxxy = 2 << 0 | 1 << 8;           /* Hash of the dxxy partial derivative. */
        static constexpr uint64_t dxxt = 2 << 0 | 1 << 16;          /* Hash of the dxxt partial derivative. */
        static constexpr uint64_t dxyy = 1 << 0 | 2 << 8;           /* Hash of the dxyy partial derivative. */
        static constexpr uint64_t dxyt = 1 << 0 | 1 << 8 | 1 << 16; /* Hash of the dxyt partial derivative. */
        static constexpr uint64_t dxtt = 1 << 0 | 2 << 16;          /* Hash of the dxtt partial derivative. */
        static constexpr uint64_t dyyy = 3 << 8;                    /* Hash of the dyyy partial derivative. */
        static constexpr uint64_t dyyt = 2 << 8 | 1 << 16;          /* Hash of the dyyt partial derivative. */
        static constexpr uint64_t dytt = 1 << 8 | 2 << 16;          /* Hash of the dytt partial derivative. */
        static constexpr uint64_t dttt = 3 << 16;                   /* Hash of the dttt partial derivative. */

        /**
         * @brief Gets the derivative degree for a given dimension.
         * @param dimension Dimension to get the derivative degree for.
         * @return Derivative degree of given dimension.
         */
        uint8_t operator[](uint8_t dimension) const
        {
            return degree[dimension];
        }

        /**
         * @brief Gets the derivative degree for a given dimension.
         * @param dimension Dimension to get the derivative degree for.
         * @return Derivative degree of given dimension.
         */
        uint8_t& operator[](uint8_t dimension)
        {
            return degree[dimension];
        }

    private:
        /**
         * @brief Array of derivative degrees for up to 8 dimensions.
         */
        uint8_t degree[8];
    };

    /**
     * @brief Specialization of partial derivative specification for steady 3D fields (SpatialDimensions=3, Dimensions=3).
     */
    template <>
    union BasePartial<3, 3>
    {
        /**
         * @brief Constructor from hash.
         * @param _hash Hash that specifies the derivative degree for all dimensions.
         */
        BasePartial(uint64_t _hash)
            : hash(_hash)
        {
        }

        /**
         * @brief Constructor with individual partial derivative degrees.
         * @param _dx Degree of the x-partial derivative.
         * @param _dy Degree of the y-partial derivative.
         * @param _dz Degree of the z-partial derivative.
         */
        BasePartial(uint8_t _dx, uint8_t _dy, uint8_t _dz)
            : hash(0)
        {
            degreeDx = _dx;
            degreeDy = _dy;
            degree_dz = _dz;
        }

        uint64_t hash; /* Hash that can be used for comparison and switch cases. */
        struct
        {
            uint8_t degreeDx; /* Specifies the degree of the x-partial derivative. */
            uint8_t degreeDy; /* Specifies the degree of the y-partial derivative. */
            uint8_t degree_dz; /* Specifies the degree of the z-partial derivative. */
        };

        static constexpr uint64_t c    = 0;                         /* Hash of the function (no partial). */
        static constexpr uint64_t dx   = 1 << 0;                    /* Hash of the dx partial derivative. */
        static constexpr uint64_t dy   = 1 << 8;                    /* Hash of the dy partial derivative. */
        static constexpr uint64_t dz   = 1 << 16;                   /* Hash of the dz partial derivative. */
        static constexpr uint64_t dxx  = 2 << 0;                    /* Hash of the dxx partial derivative. */
        static constexpr uint64_t dxy  = 1 << 0 | 1 << 8;           /* Hash of the dxy partial derivative. */
        static constexpr uint64_t dxz  = 1 << 0 | 1 << 16;          /* Hash of the dxz partial derivative. */
        static constexpr uint64_t dyy  = 2 << 8;                    /* Hash of the dyy partial derivative. */
        static constexpr uint64_t dyz  = 1 << 8 | 1 << 16;          /* Hash of the dyz partial derivative. */
        static constexpr uint64_t dzz  = 2 << 16;                   /* Hash of the dzz partial derivative. */
        static constexpr uint64_t dxxx = 3 << 0;                    /* Hash of the dxxx partial derivative. */
        static constexpr uint64_t dxxy = 2 << 0 | 1 << 8;           /* Hash of the dxxy partial derivative. */
        static constexpr uint64_t dxxz = 2 << 0 | 1 << 16;          /* Hash of the dxxz partial derivative. */
        static constexpr uint64_t dxyy = 1 << 0 | 2 << 8;           /* Hash of the dxyy partial derivative. */
        static constexpr uint64_t dxyz = 1 << 0 | 1 << 8 | 1 << 16; /* Hash of the dxyz partial derivative. */
        static constexpr uint64_t dxzz = 1 << 0 | 2 << 16;          /* Hash of the dxzz partial derivative. */
        static constexpr uint64_t dyyy = 3 << 8;                    /* Hash of the dyyy partial derivative. */
        static constexpr uint64_t dyyz = 2 << 8 | 1 << 16;          /* Hash of the dyyz partial derivative. */
        static constexpr uint64_t dyzz = 1 << 8 | 2 << 16;          /* Hash of the dyzz partial derivative. */
        static constexpr uint64_t dzzz = 3 << 16;                   /* Hash of the dzzz partial derivative. */

        /**
         * @brief Gets the derivative degree for a given dimension.
         * @param dimension Dimension to get the derivative degree for.
         * @return Derivative degree of given dimension.
         */
        uint8_t operator[](uint8_t dimension) const
        {
            return degree[dimension];
        }

        /**
         * @brief Gets the derivative degree for a given dimension.
         * @param dimension Dimension to get the derivative degree for.
         * @return Derivative degree of given dimension.
         */
        uint8_t& operator[](uint8_t dimension)
        {
            return degree[dimension];
        }

    private:
        /**
         * @brief Array of derivative degrees for up to 8 dimensions.
         */
        uint8_t degree[8];
    };

    /**
     * @brief Specialization of partial derivative specification for unsteady 3D fields (SpatialDimensions=3, Dimensions=4).
     */
    template <>
    union BasePartial<3, 4>
    {
        /**
         * @brief Constructor from hash.
         * @param _hash Hash that specifies the derivative degree for all dimensions.
         */
        BasePartial(uint64_t _hash)
            : hash(_hash)
        {
        }

        /**
         * @brief Constructor with individual partial derivative degrees.
         * @param _dx Degree of the x-partial derivative.
         * @param _dy Degree of the y-partial derivative.
         * @param _dz Degree of the z-partial derivative.
         * @param _dt Degree of the t-partial derivative.
         */
        BasePartial(uint8_t _dx, uint8_t _dy, uint8_t _dz, uint8_t _dt)
            : hash(0)
        {
            degreeDx = _dx;
            degreeDy = _dy;
            degree_dz = _dz;
            degreeDt = _dt;
        }

        uint64_t hash; /* Hash that can be used for comparison and switch cases. */
        struct
        {
            uint8_t degreeDx; /* Specifies the degree of the x-partial derivative. */
            uint8_t degreeDy; /* Specifies the degree of the y-partial derivative. */
            uint8_t degree_dz; /* Specifies the degree of the z-partial derivative. */
            uint8_t degreeDt; /* Specifies the degree of the t-partial derivative. */
        };

        static constexpr uint64_t c    = 0;                          /* Hash of the function (no partial). */
        static constexpr uint64_t dx   = 1 << 0;                     /* Hash of the dx partial derivative. */
        static constexpr uint64_t dy   = 1 << 8;                     /* Hash of the dy partial derivative. */
        static constexpr uint64_t dz   = 1 << 16;                    /* Hash of the dz partial derivative. */
        static constexpr uint64_t dt   = 1 << 24;                    /* Hash of the dt partial derivative. */
        static constexpr uint64_t dxx  = 2 << 0;                     /* Hash of the dxx partial derivative. */
        static constexpr uint64_t dxy  = 1 << 0 | 1 << 8;            /* Hash of the dxy partial derivative. */
        static constexpr uint64_t dxz  = 1 << 0 | 1 << 16;           /* Hash of the dxz partial derivative. */
        static constexpr uint64_t dxt  = 1 << 0 | 1 << 24;           /* Hash of the dxt partial derivative. */
        static constexpr uint64_t dyy  = 2 << 8;                     /* Hash of the dyy partial derivative. */
        static constexpr uint64_t dyz  = 1 << 8 | 1 << 16;           /* Hash of the dyz partial derivative. */
        static constexpr uint64_t dyt  = 1 << 8 | 1 << 24;           /* Hash of the dyt partial derivative. */
        static constexpr uint64_t dzz  = 2 << 16;                    /* Hash of the dzz partial derivative. */
        static constexpr uint64_t dzt  = 1 << 16 | 1 << 24;          /* Hash of the dzt partial derivative. */
        static constexpr uint64_t dtt  = 2 << 24;                    /* Hash of the dtt partial derivative. */
        static constexpr uint64_t dxxx = 3 << 0;                     /* Hash of the dxxx partial derivative. */
        static constexpr uint64_t dxxy = 2 << 0 | 1 << 8;            /* Hash of the dxxy partial derivative. */
        static constexpr uint64_t dxxz = 2 << 0 | 1 << 16;           /* Hash of the dxxz partial derivative. */
        static constexpr uint64_t dxxt = 2 << 0 | 1 << 24;           /* Hash of the dxxt partial derivative. */
        static constexpr uint64_t dxyy = 1 << 0 | 2 << 8;            /* Hash of the dxyy partial derivative. */
        static constexpr uint64_t dxyz = 1 << 0 | 1 << 8 | 1 << 16;  /* Hash of the dxyz partial derivative. */
        static constexpr uint64_t dxyt = 1 << 0 | 1 << 8 | 1 << 24;  /* Hash of the dxyt partial derivative. */
        static constexpr uint64_t dxzz = 1 << 0 | 2 << 16;           /* Hash of the dxzz partial derivative. */
        static constexpr uint64_t dxzt = 1 << 0 | 1 << 16 | 1 << 24; /* Hash of the dxzt partial derivative. */
        static constexpr uint64_t dxtt = 1 << 0 | 2 << 24;           /* Hash of the dxzt partial derivative. */
        static constexpr uint64_t dyyy = 3 << 8;                     /* Hash of the dyyy partial derivative. */
        static constexpr uint64_t dyyz = 2 << 8 | 1 << 16;           /* Hash of the dyyz partial derivative. */
        static constexpr uint64_t dyyt = 2 << 8 | 1 << 24;           /* Hash of the dyyt partial derivative. */
        static constexpr uint64_t dyzz = 1 << 8 | 2 << 16;           /* Hash of the dyzz partial derivative. */
        static constexpr uint64_t dyzt = 1 << 8 | 1 << 16 | 1 << 24; /* Hash of the dyzt partial derivative. */
        static constexpr uint64_t dytt = 1 << 8 | 2 << 24;           /* Hash of the dyzt partial derivative. */
        static constexpr uint64_t dzzz = 3 << 16;                    /* Hash of the dzzz partial derivative. */
        static constexpr uint64_t dzzt = 2 << 16 | 1 << 24;          /* Hash of the dzzt partial derivative. */
        static constexpr uint64_t dztt = 1 << 16 | 2 << 24;          /* Hash of the dztt partial derivative. */
        static constexpr uint64_t dttt = 3 << 24;                    /* Hash of the dttt partial derivative. */

        /**
         * @brief Gets the derivative degree for a given dimension.
         * @param dimension Dimension to get the derivative degree for.
         * @return Derivative degree of given dimension.
         */
        uint8_t operator[](uint8_t dimension) const
        {
            return degree[dimension];
        }

        /**
         * @brief Gets the derivative degree for a given dimension.
         * @param dimension Dimension to get the derivative degree for.
         * @return Derivative degree of given dimension.
         */
        uint8_t& operator[](uint8_t dimension)
        {
            return degree[dimension];
        }

    private:
        /**
         * @brief Array of derivative degrees for up to 8 dimensions.
         */
        uint8_t degree[8];
    };

    /**
     * @brief Partial derivative specification for steady 2D fields.
     */
    using PartialSteady2d = BasePartial<2, 2>;

    /**
     * @brief Partial derivative specification for unsteady 2D fields.
     */
    using PartialUnsteady2d = BasePartial<2, 3>;

    /**
     * @brief Partial derivative specification for steady 3D fields.
     */
    using PartialSteady3d = BasePartial<3, 3>;

    /**
     * @brief Partial derivative specification for unsteady 3D fields.
     */
    using PartialUnsteady3d = BasePartial<3, 4>;
}
