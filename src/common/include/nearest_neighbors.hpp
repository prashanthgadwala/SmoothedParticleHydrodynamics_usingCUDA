#pragma once

#include <vislab/core/array.hpp>

#include "include/nanoflann.hpp"

namespace physsim
{
    /**
     * @brief Class the performs nearest neighbor queries based on the nanoflann library.
     * @tparam TArrayType Type of the data array that holds the points.
     */
    template <typename TArrayType>
    class NearestNeighbors
    {
    public:
        /**
         * @brief Type of the data array that holds the points.
         */
        using ArrayType = TArrayType;

        /**
         * @brief Result type for radius queries, containing the indices of the closest points and the squared distances.
         */
        using RadiusResult = std::vector<nanoflann::ResultItem<uint32_t, typename ArrayType::Scalar>>;

        /**
         * @brief Constructor.
         */
        NearestNeighbors()
            : maxLeaf(10)
        {
        }

        /**
         * @brief Retrieves the "numPnts" closest points to the query "point".
         * @param point Point to find the closest neighbors for.
         * @param numPnts Number of closest neighbors to look for.
         * @param outputIndices Array of indices of the closest points. This array must be pre-allocated.
         * @param outputDistances Array of squared distances. This array must be pre-allocated.
         * @return Actual number of closest points that have been found.
         */
        std::size_t closestPoints(const typename ArrayType::Element& point, const Eigen::Index& numPnts, uint32_t* outputIndices, typename ArrayType::Scalar* outputDistances)
        {
            if (mArray == nullptr)
                return 0;

            return mTree->knnSearch(point.ptr(), numPnts, outputIndices, outputDistances);
        }

        /**
         * @brief Retrieves the closest points to the query "point" within a given search "radius".
         * @param point Point to find the closest neighbors for.
         * @param radius Radius to find all neighbors in.
         * @param output Vector containing the indices of the closest points and the squared distances.
         * @return Actual number of closest points that have been found.
         */
        std::size_t closestRadius(const typename ArrayType::Element& point, const typename ArrayType::Scalar& radius, RadiusResult& output)
        {
            if (mArray == nullptr)
                return 0;

            return mTree->radiusSearch(point.ptr(), radius, output);
        }

        /**
         * @brief Sets the point set and rebuilds the spatial acceleration data structure.
         * @param array New array of points to perform search queries in.
         */
        void setPoints(std::shared_ptr<ArrayType> array)
        {
            mArray = array;
            rebuildTree();
        }

        /**
         * @brief Maximum number of elements per leaf node. Changes to this variable become effective after rebuilding this tree, i.e., after setting points with "setPoints".
         */
        int maxLeaf;

    private:
        /**
         * @brief Internal point cloud data structure that provides nanoflann understandable access to the point data.
         */
        struct PointCloud
        {
            /**
             * @brief Constructor.
             * @param array Array to perform queries in.
             */
            PointCloud(const ArrayType& array)
                : mArray(array)
            {
            }

            /**
             * @brief Data array containing the data.
             */
            const ArrayType& mArray;

            /**
             * @brief Gets the number of data points.
             * @return Number of data points.
             */
            inline std::size_t kdtree_get_point_count() const { return mArray.getSize(); }

            /**
             * @brief Gets the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
             * @param p1 Coordinates of the input query point.
             * @param idx_p2 Data point to compare with.
             * @param size Number of dimensions.
             * @return L2 distance.
             */
            inline typename ArrayType::Scalar kdtree_distance(const typename ArrayType::Scalar* p1, const std::size_t idx_p2, std::size_t size) const
            {
                typename ArrayType::Scalar dist = 0;
                for (std::size_t i = 0; i < size; ++i)
                {
                    auto diff = p1[i] - mArray.getValue(idx_p2)[i];
                    dist += diff * diff;
                }
                return dist;
            }

            /**
             * @brief Returns the dim'th component of the idx'th point in the class.
             * @param idx Index of the point.
             * @param dim Dimension to get value of.
             * @return Value of point in that dimension.
             */
            inline double kdtree_get_pt(const std::size_t idx, std::size_t dim) const
            {
                return mArray.getValue(idx)[dim];
            }

            /**
             * @brief Optional bounding-box computation: return false to default to a standard bbox computation loop.
             * @tparam BBOX Bounding box type.
             * @param box Output bounding box.
             * @return True if the bounding box was provided, false if it should be computed from scratch.
             */
            template <class BBOX>
            bool kdtree_get_bbox(BBOX& box) const { return false; }
        };

        /**
         * @brief Underlying KDtree adaptor for nanoflann.
         */
        using TreeType = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<typename ArrayType::Scalar, PointCloud>,
            PointCloud,
            ArrayType::Dimensions>;

        /**
         * @brief Helper routine that rebuilds the tree data structure.
         */
        void rebuildTree()
        {
            mCloud = std::make_unique<PointCloud>(*mArray);
            mTree  = std::make_unique<TreeType>(ArrayType::Dimensions, *mCloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(maxLeaf));
            mTree->buildIndex();
        }

        /**
         * @brief Data array that contains all the points.
         */
        std::shared_ptr<ArrayType> mArray;

        /**
         * @brief Intermediate wrapper from array to nanoflann.
         */
        std::unique_ptr<PointCloud> mCloud;

        /**
         * @brief Nanoflann tree that is used for the queries.
         */
        std::unique_ptr<TreeType> mTree;
    };

    /**
     * @brief Nearest neighbor computation in two dimensions with float precision.
     */
    using NearestNeighbors2f = NearestNeighbors<vislab::Array2f>;

    /**
     * @brief Nearest neighbor computation in two dimensions with double precision.
     */
    using NearestNeighbors2d = NearestNeighbors<vislab::Array2d>;

    /**
     * @brief Nearest neighbor computation in three dimensions with float precision.
     */
    using NearestNeighbors3f = NearestNeighbors<vislab::Array3f>;

    /**
     * @brief Nearest neighbor computation in three dimensions with double precision.
     */
    using NearestNeighbors3d = NearestNeighbors<vislab::Array3d>;
}
