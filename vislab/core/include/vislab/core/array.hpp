#pragma once

#include "array_fwd.hpp"
#include "base_array.hpp"
#include "iarchive.hpp"
#include "traits.hpp"

namespace vislab
{
    /**
     * @brief Type-safe data array.
     * @tparam TElement Type of element in the array.
     */
    template <typename TElement>
    class Array : public Concrete<Array<TElement>, BaseArray<sizeof(TElement) / sizeof(typename TElement::Scalar)>>
    {
    public:
        /**
         * @brief Underlying scalar type.
         */
        using Scalar = typename TElement::Scalar;

        /**
         * @brief Number of components in each of the element entries.
         */
        static constexpr int Dimensions = sizeof(TElement) / sizeof(Scalar);

        /**
         * @brief Underlying Eigen Matrix type.
         */
        using Matrix = Eigen::Matrix<Scalar, Dimensions, -1>;

        /**
         * @brief Underlying element vector stored at each array index.
         */
        using Element = TElement;

        /**
         * @brief Constructor.
         */
        Array()
        {
        }

        /**
         * @brief Constructor that gives the array a name.
         * @param name Name of the array.
         */
        Array(const std::string& _name)
            : Concrete<Array<TElement>, BaseArray<sizeof(TElement) / sizeof(typename TElement::Scalar)>>()
        {
            this->name = _name;
        }

        /**
         * @brief Gets the underlying vector.
         * @return std::vector that stores the data in the array.
         */
        [[nodiscard]] inline const std::vector<TElement>& getData() const&
        {
            return mData;
        }

        /**
         * @brief Gets the underlying vector.
         * @return std::vector that stores the data in the array.
         */
        [[nodiscard]] inline std::vector<TElement>& getData() &
        {
            return mData;
        }

        /**
         * @brief Gets the underlying vector.
         * @return std::vector that stores the data in the array.
         */
        [[nodiscard]] inline std::vector<TElement>&& getData() &&
        {
            return std::move(mData);
        }

        /**
         * @brief Gets the number of elements.
         * @return Number of elements in the array.
         */
        [[nodiscard]] inline Eigen::Index getSize() const override
        {
            return mData.size();
        }

        /**
         * @brief Gets the number of components of a single element.
         * @return Number of components per element.
         */
        [[nodiscard]] inline Eigen::Index getNumComponents() const override
        {
            return Dimensions;
        }

        /**
         * @brief Get the size in bytes of a single element.
         * @return Size in bytes of each element.
         */
        [[nodiscard]] inline Eigen::Index getElementSizeInBytes() const override
        {
            return sizeof(Element);
        }

        /**
         * @brief Get the size in bytes of the entire array (only counting the data, not meta information like the array name).
         * @return Total size in bytes.
         */
        [[nodiscard]] inline Eigen::Index getSizeInBytes() const override
        {
            return getSize() * getElementSizeInBytes();
        }

        /**
         * @brief Sets the number of elements.
         * @param size New number of elements.
         */
        inline void setSize(Eigen::Index size) override
        {
            mData.resize(size);
        }

        /**
         * @brief Gets the element at a specific index in its internal format.
         * @param index Array index.
         * @return Element at a certain array index.
         */
        [[nodiscard]] inline const Element& getValue(Eigen::Index index) const
        {
            return mData[index];
        }

        /**
         * @brief Sets the element at a specific index with all its components in its internal format.
         * @param index Array index.
         * @param value New element to set at the array index.
         */
        inline void setValue(Eigen::Index index, const Element& value)
        {
            mData[index] = value;
        }

        /**
         * @brief Sets the element at a specific index if it is a scalar value.
         * @tparam D SFINAE parameter.
         * @param index Array index.
         * @param value New scalar element to set at the array index.
         * @return void
         */
        template <int D = Dimensions>
        inline typename std::enable_if_t<(D == 1)> setValue(Eigen::Index index, Scalar value)
        {
            mData[index].x() = value;
        }

        /**
         * @brief Constructs the array with an initializer list.
         * @tparam D SFINAE parameter.
         * @param list List of values.
         * @return void
         */
        template <int D = Dimensions>
        typename std::enable_if_t<(D == 1)> setValues(std::initializer_list<Scalar> list)
        {
            setSize(list.size());
            std::size_t i = 0;
            for (auto elem : list)
                setValue(i++, elem);
        }

        /**
         * @brief Constructs the array with an initializer list.
         * @param list List of elements.
         */
        void setValues(std::initializer_list<Element> list)
        {
            setSize(list.size());
            std::size_t i = 0;
            for (auto elem : list)
                setValue(i++, elem);
        }

        /**
         * @brief Sets all values to zero.
         */
        inline void setZero() override
        {
            std::memset(mData.data(), 0, getSizeInBytes());
        }

        /**
         * @brief Deletes all elements.
         */
        inline void clear() override
        {
            mData.clear();
        }

        /**
         * @brief Appends a value at the end.
         * @param value Value to append.
         */
        void append(const Element& value)
        {
            setSize(getSize() + 1);
            setValue(getSize() - 1, value);
        }

        /**
         * @brief Appends an entire array at the end.
         * @param other Array to append.
         */
        void append(const Array& other)
        {
            std::size_t oldsize = getSize();
            setSize(oldsize + other.getSize());
            for (int i = 0; i < other.getSize(); ++i)
                setValue(oldsize + i, other.getValue(i));
        }

        /**
         * @brief Appends an entire array at the end. Must be the same type, otherwise this function does nothing.
         * @param other Array to append.
         */
        void append(const IArray* other) override
        {
            if (dynamic_cast<const Array*>(other))
                append(*dynamic_cast<const Array*>(other));
        }

        /**
         * @brief Prepends an element at the front (linear time complexity, since the entire array is copied).
         * @param value Element to prepend.
         */
        void prepend(const Element& value)
        {
            std::size_t oldsize = getSize();
            setSize(getSize() + 1);
            for (int i = (int)oldsize - 1; i >= 0; --i)
                setValue(i + 1, getValue(i));
            setValue(0, value);
        }

        /**
         * @brief Prepends an entire array at the front (linear time complexity, since the entire array is copied).
         * @param other Array to prepend.
         */
        void prepend(const Array& other)
        {
            // resize the buffer to accomodate the current and the other ones data
            std::size_t oldsize   = getSize();
            std::size_t othersize = other.getSize();
            setSize(oldsize + othersize);
            // move all entries to the end
            for (int i = (int)oldsize - 1; i >= 0; --i)
                setValue(othersize + i, getValue(i));
            // insert the new values at the front
            for (std::size_t i = 0; i < othersize; ++i)
                setValue(i, other.getValue(i));
        }

        /**
         * @brief Prepends an entire array at the front (linear time complexity, since the entire array is copied). Must be the same type, otherwise this function does nothing.
         * @param other Array to prepend.
         */
        void prepend(const IArray* other) override
        {
            if (dynamic_cast<const Array*>(other))
                prepend(*dynamic_cast<const Array*>(other));
        }

        /**
         * @brief Reverses the order of the elements.
         */
        void reverse() override
        {
            const Eigen::Index n = getSize();
            for (Eigen::Index i = 0; i < n / 2; ++i)
            {
                Element temp = getValue(i);
                setValue(i, getValue(n - 1 - i));
                setValue(n - 1 - i, temp);
            }
        }

        /**
         * @brief Removes the last n elements of this vector.
         * @param n Number of elements to remove at the end.
         */
        inline void removeLast(std::size_t n = 1) override
        {
            setSize(getSize() - n);
        }

        /**
         * @brief Removes the first n elements of this vector.
         * @param n Number of elements to remove at the front.
         */
        void removeFirst(std::size_t n = 1) override
        {
            // Copy each element n places ahead
            for (std::size_t i = n; i < getSize(); ++i)
                setValue(i - n, getValue(i));
            removeLast(n);
        }

        /**
         * @brief Get the first element.
         * @return First element.
         */
        inline const Element& first() const
        {
            return getValue(0);
        }

        /**
         * @brief Get the last element.
         * @return Last element.
         */
        inline const Element& last() const
        {
            return getValue(getSize() - 1);
        }

        /**
         * @brief Gets component-wise the minimum value (computed in linear time).
         * @return Smallest value per component.
         */
        [[nodiscard]] inline Element getMin() const
        {
            Element val = getValue(0);
            for (std::size_t i = 1; i < mData.size(); ++i)
            {
                for (int d = 0; d < Dimensions; ++d)
                {
                    if (mData[i](d) < val(d))
                        val(d) = mData[i](d);
                }
            }
            return val;
        }

        /**
         * @brief Gets component-wise the maximum value (computed in linear time).
         * @return Largest value per component.
         */
        [[nodiscard]] inline Element getMax() const
        {
            Element val = getValue(0);
            for (std::size_t i = 1; i < mData.size(); ++i)
            {
                for (int d = 0; d < Dimensions; ++d)
                {
                    if (mData[i](d) > val(d))
                        val(d) = mData[i](d);
                }
            }
            return val;
        }

        /**
         * @brief Sort componentwise in ascending order.
         */
        void sortAscending() override
        {
            std::vector<Scalar> entries(getSize());
            for (int c = 0; c < getNumComponents(); ++c)
            {
                for (Eigen::Index i = 0; i < getSize(); ++i)
                    entries[i] = mData[i](c);
                std::sort(entries.begin(), entries.end());
                for (Eigen::Index i = 0; i < getSize(); ++i)
                    mData[i](c) = entries[i];
            }
        }

        /**
         * @brief Sort componentwise in descending order.
         */
        void sortDescending() override
        {
            std::vector<Scalar> entries(getSize());
            for (int c = 0; c < getNumComponents(); ++c)
            {
                for (Eigen::Index i = 0; i < getSize(); ++i)
                    entries[i] = mData[i](c);
                std::sort(entries.rbegin(), entries.rend());
                for (Eigen::Index i = 0; i < getSize(); ++i)
                    mData[i](c) = entries[i];
            }
        }

        /**
         * @brief Tests if two arrays are equal.
         * @param other Array to compare with.
         * @return True if equal.
         */
        [[nodiscard]] bool isEqual(const IArray* other) const override
        {
            const Array* otherTyped = dynamic_cast<const Array*>(other);
            if (!otherTyped)
                return false;
            if (this->getSize() != otherTyped->getSize())
                return false;
            if (this->name != other->name)
                return false;
            return mData == otherTyped->mData;
        }

        /**
         * @brief Tests if two arrays are equal.
         * @param other Array to compare with.
         * @return True if equal.
         */
        [[nodiscard]] bool isEqual(const Array& other) const
        {
            if (this->getSize() != other.getSize())
                return false;
            if (this->name != other.name)
                return false;
            return mData == other.mData;
        }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override
        {
            IArray::serialize(archive);
            archive("Data", mData);
        }

    protected:
        /**
         * @brief Vector containing the elements of the array.
         */
        std::vector<Element> mData;
    };
}
