#pragma once

#include <vislab/core/data.hpp>
#include <vislab/core/iarray.hpp>

namespace vislab
{
    /**
     * @brief Stores attribute arrays.
     */
    class Attributes : public Concrete<Attributes, Data>
    {
    public:
        /**
         * @brief Constructor.
         */
        Attributes();

        /**
         * @brief Copy-constructor.
         * @param other
         */
        Attributes(const Attributes& other);

        /**
         * @brief Destructor.
         */
        virtual ~Attributes();

        /**
         * @brief Removes all attributes.
         */
        void clear();

        /**
         * @brief Appends an attribute set in the end. Types and order must be identical.
         * @param attributes Attributes to append.
         */
        void append(const Attributes* attributes);

        /**
         * @brief Preprends an attribute set at the front.
         * @param attributes Attributes to prepend.
         */
        void prepend(const Attributes* attributes);

        /**
         * @brief Adds an array at the end to the set.
         * @param array Array to add.
         */
        void add(std::shared_ptr<IArray> array);

        /**
         * @brief Removes the first n elements of each attribute.
         * @param n Number of elements to remove.
         */
        void removeFirst(std::size_t n = 1);

        /**
         * @brief Removes the last n elements of each attribute.
         * @param n Number of elements to remove.
         */
        void removeLast(std::size_t n = 1);

        /**
         * @brief Reverses the order of the attribute entries.
         */
        void reverse();

        /**
         * @brief Tests if two attribute containers are equal.
         * @param other Attributes to compare to.
         * @return True, if they are equal.
         */
        [[nodiscard]] virtual bool isEqual(const Attributes* other) const;

        /**
         * @brief Gets the number of attribute arrays. Returns zero if no attributes have been created yet.
         * @return Number of attribute arrays.
         */
        [[nodiscard]] std::size_t getSize() const;

        /**
         * @brief sets the number of attribute arrays.
         * @return Number of attribute arrays.
         */
        void setSize(std::size_t size);

        /**
         * @brief Gets a pointer to a specific attribute. This is the base type and has to be casted into a templated type (IArray<T>) to get access to the internals of the array.
         * @param index Index of the attribute to get.
         * @return Attribute array.
         */
        [[nodiscard]] std::shared_ptr<IArray> getByIndex(std::size_t index);

        /**
         * @brief Gets a pointer to a specific attribute. This is the base type and has to be casted into a templated type (IArray<T>) to get access to the internals of the array.
         * @param index Index of the attribute to get.
         * @return Attribute array.
         */
        [[nodiscard]] std::shared_ptr<const IArray> getByIndex(std::size_t index) const;

        /**
         * @brief Gets a pointer to a specific attribute, which is dynamically casted to a specific attribute type. If the requested type does not match the internal type, the function returns null.
         * @tparam TArrayType Array type to cast into.
         * @param index Index of the attribute to get.
         * @return Attribute array.
         */
        template <typename TArrayType>
        [[nodiscard]] inline std::shared_ptr<TArrayType> getByIndex(std::size_t index) { return std::dynamic_pointer_cast<TArrayType, IArray>(mAttributes[index]); }

        /**
         * @brief Gets a pointer to a specific attribute, which is dynamically casted to a specific attribute type. If the requested type does not match the internal type, the function returns null.
         * @tparam TArrayType Array type to cast into.
         * @param index Index of the attribute to get.
         * @return Attribute array.
         */
        template <typename TArrayType>
        [[nodiscard]] inline std::shared_ptr<const TArrayType> getByIndex(std::size_t index) const { return std::dynamic_pointer_cast<const TArrayType, IArray>(mAttributes[index]); }

        /**
         * @brief Gets a pointer to a specific attribute by searching for its name. This is the base type and has to be casted into a templated type (IArray<T>) to get access to the internals of the array.
         * @param name Name of the attribute to get.
         * @return Attribute array.
         */
        [[nodiscard]] std::shared_ptr<IArray> getByName(const std::string& name);

        /**
         * @brief Gets a pointer to a specific attribute by searching for its name. This is the base type and has to be casted into a templated type (IArray<T>) to get access to the internals of the array.
         * @param name Name of the attribute to get.
         * @return Attribute array.
         */
        [[nodiscard]] std::shared_ptr<const IArray> getByName(const std::string& name) const;

        /**
         * @brief Gets a pointer to a specific attribute by searching for its name. The pointer is dynamically casted to a specific attribute type. If the requested type does not match the internal type, the function returns null.
         * @tparam TArrayType Array type to cast into.
         * @param name Name of the attribute to get.
         * @return Attribute array.
         */
        template <typename TArrayType>
        [[nodiscard]] std::shared_ptr<TArrayType> getByName(const std::string& name)
        {
            for (auto& attr : mAttributes)
                if (attr->name == name)
                    return std::dynamic_pointer_cast<TArrayType>(attr);
            return nullptr;
        }

        /**
         * @brief Gets a pointer to a specific attribute by searching for its name. the pointer is dynamically casted to a specific attribute type. if the requested type does not match the internal type, the function returns null.
         * @tparam TArrayType Array type to cast into.
         * @param name Name of the attribute to get.
         * @return Attribute array.
         */
        template <typename TArrayType>
        [[nodiscard]] std::shared_ptr<const TArrayType> getByName(const std::string& name) const
        {
            for (auto& attr : mAttributes)
                if (attr->name == name)
                    return std::dynamic_pointer_cast<const TArrayType>(attr);
            return nullptr;
        }

        /**
         * @brief Sets the pointer to a specific attribute.
         * @param index Index of the attribute to set. Must be between 0 and the number of arrays in the attributes property.
         * @param iarray Array to assign.
         */
        void setByIndex(std::size_t index, std::shared_ptr<IArray> iarray);

        /**
         * @brief Creates an attribute of a specific type (template argument) and stores the attribute with a certain name.
         * @tparam TArrayType Array type to create.
         * @param name Name of the array to create.
         * @return Attribute array that was created.
         */
        template <typename TArrayType>
        [[nodiscard]] std::shared_ptr<TArrayType> create(const std::string& name)
        {
            auto result  = std::make_shared<TArrayType>();
            result->name = name;
            mAttributes.push_back(result);
            return result;
        }

        /**
         * @brief Gets the name of an attribute.
         * @param attributeIndex Index of the attribute to get the name from.
         * @return Name of the array.
         */
        [[nodiscard]] const std::string& getName(const std::size_t& attributeIndex) const;

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Tests if the data object is fully initialized and ready for use.
         * @return True if the data object is in a valid state.
         */
        [[nodiscard]] bool isValid() const override;

    private:
        /**
         * @brief Array of attributes.
         */
        std::vector<std::shared_ptr<IArray>> mAttributes;
    };
}
