#pragma once

#include "cloneable.hpp"

#include <string>

namespace vislab
{
    /**
     * @brief Makes this an abstract vislab class, which means that it is cloneable and reflectable.
     * @tparam Type Type of the class.
     * @tparam ...Bases Optional list of base classes that also derived from Interface or Concrete.
     */
    template <typename Type, typename... Bases>
    using Interface = Cloneable<detail::MarkAbstract<Type>, Bases...>;

    /**
     * @brief Makes this a concrete vislab class, which means that it is cloneable and reflectable.
     * @tparam Type Type of the class.
     * @tparam ...Bases Optional list of base classes that also derived from Interface or Concrete.
     */
    template <typename Type, typename... Bases>
    using Concrete = Cloneable<Type, Bases...>;

    /**
     * @brief Base class for an object.
     */
    class Object : public Interface<Object>
    {
    };

    /**
     * @brief Factory for creating objects from vislab::Type.
     */
    class Factory
    {
    public:
        /**
         * @brief Allocates an object.
         * @tparam TType Optional type to cast into.
         * @param type Type of object to create.
         * @return Allocated object.
         */
        template <typename TType = vislab::Object>
        static std::shared_ptr<TType> create(vislab::Type type)
        {
            // use meta to construct the object
            meta::any anyVal = type.construct();
            // cast to the native type. note that meta keeps ownership of the object
            TType* toClone = anyVal.try_cast<TType>();
            // clone it to make our own reference counted version
            auto instance = toClone->clone();
            // return result
            return instance;
        }

        /**
         * @brief Allocates an object by name.
         * @tparam TType Optional type to cast into.
         * @param uid Unique name of the class to instantiate.
         * @return Object that was allocated
         */
        template <typename TType = vislab::Object>
        static std::shared_ptr<TType> create(const std::string& uid)
        {
            std::hash<std::string_view> hash{};
            Type t = meta::resolve(hash(uid));
            return create<TType>(t);
        }
    };
}
