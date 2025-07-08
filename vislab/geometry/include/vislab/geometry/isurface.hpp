#pragma once

#include <vislab/core/data.hpp>

namespace vislab
{
    class IArray;
    class Attributes;

    /**
     * @brief Interface for class that stores an individual surface.
     */
    class ISurface : public Interface<ISurface, Data>
    {
    public:
        /**
         * @brief Constructor.
         */
        ISurface();

        /**
         * @brief Copy-constructor.
         * @param other Surface to copy from.
         */
        ISurface(const ISurface& other);

        /**
         * @brief Destructor.
         */
        virtual ~ISurface();

        /**
         * @brief Gets a pointer to the vertex positions data.
         * @return Array containing vertex positions.
         */
        [[nodiscard]] std::shared_ptr<IArray> getPositions();

        /**
         * @brief Gets a pointer to the vertex positions data.
         * @return Array containing vertex positions.
         */
        [[nodiscard]] std::shared_ptr<const IArray> getPositions() const;

        /**
         * @brief Clear all vertices, attributes, and indices.
         */
        virtual void clear();

        /**
         * @brief Recomputes the bounding box from the vertex buffer.
         */
        virtual void recomputeBoundingBox() = 0;

        /**
         * @brief Tests if two surface geometries are equal.
         * @param other Surface to compare with.
         * @return True, if surfaces are equal.
         */
        [[nodiscard]] virtual bool isEqual(const ISurface* other) const;

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
        [[nodiscard]] virtual std::shared_ptr<IArray> getPositionsImpl() = 0;

        /**
         * @brief Gets a pointer to the vertex data.
         * @return Vertex array.
         */
        [[nodiscard]] virtual std::shared_ptr<const IArray> getPositionsImpl() const = 0;
    };
}
