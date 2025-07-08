#pragma once

#include "data.hpp"
#include "types.hpp"

namespace vislab
{
    /**
     * @brief Abstract base class for data arrays.
     */
    class IArray : public Interface<IArray, Data>
    {
    public:
        /**
         * @brief Constructor for an abstract data array.
         */
        IArray();

        /**
         * @brief Constructor that gives the array a name.
         * @param name Name of the array.
         */
        explicit IArray(const std::string& name);

        /**
         * @brief Gets the number of elements.
         * @return Number of elements in the array.
         */
        [[nodiscard]] virtual Eigen::Index getSize() const = 0;

        /**
         * @brief Gets the number of components of a single element.
         * @return Number of components per element.
         */
        [[nodiscard]] virtual Eigen::Index getNumComponents() const = 0;

        /**
         * @brief Get the size in bytes of a single element.
         * @return Size in bytes of each element.
         */
        [[nodiscard]] virtual Eigen::Index getElementSizeInBytes() const = 0;

        /**
         * @brief Get the size in bytes of the entire array (only counting the data, not meta information like the array name).
         * @return Total size in bytes.
         */
        [[nodiscard]] virtual Eigen::Index getSizeInBytes() const = 0;

        /**
         * @brief Sets the number of elements.
         * @param size New number of elements.
         */
        virtual void setSize(Eigen::Index newSize) = 0;

        /**
         * @brief Deletes all elements.
         */
        virtual void clear() = 0;

        /**
         * @brief Appends an entire array at the end. Must be the same type, otherwise this function does nothing.
         * @param other Array to append.
         */
        virtual void append(const IArray* other) = 0;

        /**
         * @brief Prepends an entire array at the front (linear time complexity, since the entire array is copied). Must be the same type, otherwise this function does nothing.
         * @param other Array to prepend.
         */
        virtual void prepend(const IArray* other) = 0;

        /**
         * @brief Reverses the order of the elements.
         */
        virtual void reverse() = 0;

        /**
         * @brief Removes the last n elements of this vector.
         * @param n Number of elements to remove at the end.
         */
        virtual void removeLast(std::size_t n = 1) = 0;

        /**
         * @brief Removes the first n elements of this vector.
         * @param n Number of elements to remove at the front.
         */
        virtual void removeFirst(std::size_t n = 1) = 0;

        /**
         * @brief Sort componentwise in ascending order.
         */
        virtual void sortAscending() = 0;

        /**
         * @brief Sort componentwise in descending order.
         */
        virtual void sortDescending() = 0;

        /**
         * @brief Tests if two arrays are equal.
         * @param other Array to compare with.
         * @return True if equal.
         */
        [[nodiscard]] virtual bool isEqual(const IArray* other) const = 0;

        /**
         * @brief Sets all values to zero.
         */
        virtual void setZero() = 0;

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Name of the array.
         */
        std::string name;
    };
}
