#pragma once

#include "object.hpp"

namespace vislab
{

    class IArchive;

#if 1

    /**
     * @brief Interface for serializable objects.
     */
    class ISerializable : public Interface<ISerializable>
    {
    public:

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into or from.
         */
        virtual void serialize(IArchive& archive) = 0;
    };

#else

    // The current approach requires calling serialize of the base class, which might be forgotten. The pattern below avoids this developer error.

    /**
     * @brief Interface for serializable objects.
     */
    class ISerializable : public Interface<ISerializable>
    {
    protected:
        /**
         * @brief Tag for dispatching serializeImpl of the derived class.
         * @tparam Typename to make tag unique for the current class.
         */
        template <typename>
        struct SerializeTag
        {
        };

    public:
        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into or from.
         */
        void serialize(IArchive& archive) { serializeImpl(archive, SerializeTag<ISerializable>{}); }

    protected:
        /**
         * @brief Implements serialization of the object into/from an archive.
         * @param archive Archive to serialize into or from.
         */
        virtual void serializeImpl(IArchive& archive, SerializeTag<ISerializable>) {}
    };

    /**
     * @brief Example for how to use it.
     */
    class DerivedExample : public Interface<DerivedExample, Data>
    {
    public:
        int member;

    protected:
        /**
         * @brief Adds serialization code via the dispatch tag pattern.
         * @param archive Archive to serialize into/from.
         */
        void serializeImpl(IArchive& archive, ISerializable::SerializeTag<ISerializable>) override final
        {
            archive("Member", member);
            serializeImpl(archive, ISerializable::SerializeTag<DerivedExample>{});
        }

        /**
         * @brief Virtual function to override in the derived class if the derived class wants to serialize further data.
         * @param archive Archive to serialize into/from.
         */
        virtual void serializeImpl(IArchive& archive, ISerializable::SerializeTag<DerivedExample>) {}
    };

#endif
}
