#pragma once

#include <vislab/core/data.hpp>

namespace vislab
{
    class IArray;
    class Attributes;

    /**
     * @brief Basic interface for point geometry.
     */
    class IPoints : public Interface<IPoints, Data>
    {
    public:
        /**
         * @brief Constructor.
         */
        IPoints();

        /**
         * @brief Copy-constructor.
         * @param other Points to copy from.
         */
        IPoints(const IPoints& other);

        /**
         * @brief Destructor.
         */
        virtual ~IPoints();

        /**
         * @brief Gets a pointer to the vertex data.
         * @return Array containing vertices.
         */
        [[nodiscard]] std::shared_ptr<IArray> getVertices();

        /**
         * @brief Gets a pointer to the vertex data.
         * @return Array containing vertices.
         */
        [[nodiscard]] std::shared_ptr<const IArray> getVertices() const;

        /**
         * @brief Removes all vertices and attributes.
         */
        virtual void clear();

        /**
         * @brief Appends a point set in the end. If the points have attributes, their types and order must be identical.
         * @param points Point data to append.
         */
        void append(const IPoints* points);

        /**
         * @brief Preprends a point set at the front.
         * @param points Point data to prepend.
         */
        void prepend(const IPoints* points);

        /**
         * @brief Removes the first n elements of this vector.
         * @param n Number of points to remove.
         */
        void removeFirst(std::size_t n = 1);

        /**
         * @brief Removes the last n elements of this vector.
         * @param n Number of points to remove.
         */
        void removeLast(std::size_t n = 1);

        /**
         * @brief Reverses the order of the points.
         */
        void reverse();

        /**
         * @brief Recomputes the bounding box from the vertex buffer.
         */
        virtual void recomputeBoundingBox() = 0;

        /**
         * @brief Tests if two point geometries are equal.
         * @param other Points to compare with.
         * @return True, if points are equal.
         */
        [[nodiscard]] virtual bool isEqual(const IPoints* other) const;

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Attributes that are stored on this geometry.
         */
        std::shared_ptr<Attributes> attributes;

    protected:
        /**
         * @brief Gets a pointer to the vertex data.
         * @return Vertex array.
         */
        [[nodiscard]] virtual std::shared_ptr<IArray> getVerticesImpl() = 0;

        /**
         * @brief Gets a pointer to the vertex data.
         * @return Vertex array.
         */
        [[nodiscard]] virtual std::shared_ptr<const IArray> getVerticesImpl() const = 0;
    };
}
