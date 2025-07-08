#pragma once

#include <vislab/core/data.hpp>

namespace vislab
{
    class IArray;
    class Attributes;

    /**
     * @brief Interface for class that stores an individual line.
     */
    class ILine : public Interface<ILine, Data>
    {
    public:
        /**
         * @brief Constructor.
         */
        ILine();

        /**
         * @brief Copy-constructor.
         * @param other Line to copy from.
         */
        ILine(const ILine& other);

        /**
         * @brief Destructor.
         */
        virtual ~ILine();

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
         * @brief Appends a line in the end. If the points have attributes, their types and order must be identical.
         * @param line Line data to append.
         */
        void append(const ILine* line);

        /**
         * @brief Preprends a line at the front.
         * @param line Line data to prepend.
         */
        void prepend(const ILine* line);

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
         * @brief Tests if two lines are equal.
         * @param other Lines to compare with.
         * @return True, if lines are equal.
         */
        [[nodiscard]] virtual bool isEqual(const ILine* other) const;

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Computes the arc length of the line.
         * @return Arc length of the line.
         */
        [[nodiscard]] virtual double arcLength() const = 0;

        /**
         * @brief Attribute that are stored on this geometry.
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
