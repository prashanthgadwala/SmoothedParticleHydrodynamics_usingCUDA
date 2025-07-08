#pragma once

#include "type.hpp"

#include <memory>

namespace vislab
{
    namespace detail
    {
        /**
         * @brief Helper class for marking a type T as an abstract class.
         * @tparam T Type to mark.
         */
        template <typename T>
        class MarkAbstract
        {
        };
    }

    /**
     * @brief Decorates the type Derived to become clonable and reflectable.
     * @tparam Derived Type to decorate.
     * @tparam ...Bases List of base classes.
     */
    template <typename Derived, typename... Bases>
    class Cloneable : public Bases...
    {
    public:
        /**
         * @brief Inherit constructors.
         */
        using Bases::Bases...;

        /**
         * @brief Default destructor.
         */
        virtual ~Cloneable() = default;

        /**
         * @brief Makes a deep copy of this object by using the copy-constructor of the derived class. This method is overwritten by each derived class in order to return the type of the derived class.
         * @return Deep copy of this object.
         */
        [[nodiscard]] std::unique_ptr<Derived> clone() const
        {
            return std::unique_ptr<Derived>(static_cast<Derived*>(this->cloneImpl()));
        }

        /**
         * @brief Gets type information for runtime reflection.
         * @return Type information.
         */
        [[nodiscard]] static inline Type type() noexcept
        {
            return ::meta::resolve<Derived>();
        }
        /**
         * @brief Gets type information for runtime reflection.
         * @return Type information.
         */
        [[nodiscard]] inline Type getType() const noexcept
        {
            return this->getTypeImpl();
        }

    private:
        /**
         * @brief Actual implementation of the deep cloning.
         * @return Deep copy of this object.
         */
        [[nodiscard]] virtual Cloneable* cloneImpl() const
        {
            return new Derived(static_cast<const Derived&>(*this));
        }

        /**
         * @brief Actual implementation of the type getter.
         * @return Type information.
         */
        [[nodiscard]] virtual inline Type getTypeImpl() const noexcept
        {
            return ::meta::resolve<Derived>();
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Decorates the abstract type Derived to become clonable and reflectable.
     * @tparam Derived Type to decorate.
     * @tparam ...Bases List of base classes.
     */
    template <typename Derived, typename... Bases>
    class Cloneable<detail::MarkAbstract<Derived>, Bases...> : public Bases...
    {
    public:
        /**
         * @brief Inherit constructors.
         */
        using Bases::Bases...;

        /**
         * @brief Default destructor.
         */
        virtual ~Cloneable() = default;

        /**
         * @brief Makes a deep copy of this object by using the copy-constructor of the derived class. This method is overwritten by each derived class in order to return the type of the derived class.
         * @return Deep copy of this object.
         */
        [[nodiscard]] std::unique_ptr<Derived> clone() const
        {
            return std::unique_ptr<Derived>(static_cast<Derived*>(this->cloneImpl()));
        }

        /**
         * @brief Gets type information for runtime reflection.
         * @return Type information.
         */
        [[nodiscard]] static inline Type type() noexcept
        {
            return ::meta::resolve<Derived>();
        }
        /**
         * @brief Gets type information for runtime reflection.
         * @return Type information.
         */
        [[nodiscard]] inline Type getType() const noexcept
        {
            return this->getTypeImpl();
        }

    private:
        /**
         * @brief Actual implementation of the deep cloning.
         * @return Deep copy of this object.
         */
        [[nodiscard]] virtual Cloneable* cloneImpl() const = 0;

        /**
         * @brief Actual implementation of the type getter.
         * @return Type information.
         */
        [[nodiscard]] virtual inline Type getTypeImpl() const noexcept
        {
            return ::meta::resolve<Derived>();
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Decorates the type Derived to become clonable and reflectable.
     * @tparam Derived Type to decorate.
     */
    template <typename Derived>
    class Cloneable<Derived>
    {
    public:
        /**
         * @brief Default destructor.
         */
        virtual ~Cloneable() = default;

        /**
         * @brief Makes a deep copy of this object by using the copy-constructor of the derived class. This method is overwritten by each derived class in order to return the type of the derived class.
         * @return Deep copy of this object.
         */
        [[nodiscard]] std::unique_ptr<Derived> clone() const
        {
            return std::unique_ptr<Derived>(static_cast<Derived*>(this->cloneImpl()));
        }

        /**
         * @brief Gets type information for runtime reflection.
         * @return Type information.
         */
        [[nodiscard]] static inline Type type() noexcept
        {
            return ::meta::resolve<Derived>();
        }
        /**
         * @brief Gets type information for runtime reflection.
         * @return Type information.
         */
        [[nodiscard]] inline Type getType() const noexcept
        {
            return this->getTypeImpl();
        }

    private:
        /**
         * @brief Actual implementation of the deep cloning.
         * @return Deep copy of this object.
         */
        [[nodiscard]] virtual Cloneable* cloneImpl() const
        {
            return new Derived(static_cast<const Derived&>(*this));
        }

        /**
         * @brief Actual implementation of the type getter.
         * @return Type information.
         */
        [[nodiscard]] virtual inline Type getTypeImpl() const noexcept
        {
            return ::meta::resolve<Derived>();
        }
    };

    ///////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Decorates the abstract type Derived to become clonable and reflectable.
     * @tparam Derived Type to decorate.
     */
    template <typename Derived>
    class Cloneable<detail::MarkAbstract<Derived>>
    {
    public:
        /**
         * @brief Default destructor.
         */
        virtual ~Cloneable() = default;

        /**
         * @brief Makes a deep copy of this object by using the copy-constructor of the derived class. This method is overwritten by each derived class in order to return the type of the derived class.
         * @return Deep copy of this object.
         */
        [[nodiscard]] std::unique_ptr<Derived> clone() const
        {
            return std::unique_ptr<Derived>(static_cast<Derived*>(this->cloneImpl()));
        }

        /**
         * @brief Gets type information for runtime reflection.
         * @return Type information.
         */
        [[nodiscard]] static inline Type type() noexcept
        {
            return ::meta::resolve<Derived>();
        }
        /**
         * @brief Gets type information for runtime reflection.
         * @return Type information.
         */
        [[nodiscard]] inline Type getType() const noexcept
        {
            return this->getTypeImpl();
        }

    private:
        /**
         * @brief Actual implementation of the deep cloning.
         * @return Deep copy of this object.
         */
        [[nodiscard]] virtual Cloneable* cloneImpl() const = 0;

        /**
         * @brief Actual implementation of the type getter.
         * @return Type information.
         */
        [[nodiscard]] virtual inline Type getTypeImpl() const noexcept
        {
            return ::meta::resolve<Derived>();
        }
    };
}
