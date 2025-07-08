#pragma once

#include <vislab/core/event.hpp>
#include <vislab/core/object.hpp>
#include <vislab/core/iserializable.hpp>

#include <set>

namespace vislab
{
    class Tag;

    /**
     * @brief Implements a basic collection of tags.
     */
    class Tags : public Concrete<Tags, Object, ISerializable>
    {
    public:
        /**
         * @brief Adds a tag.
         * @tparam TTag Type of the tag to add.
         * @param tag Tag to add.
         */
        template <typename TTag>
        void add()
        {
            if (!has<TTag>())
                mTags.insert(TTag::type().identifier());
        }

        /**
         * @brief Removes a tag.
         * @tparam TTag Type of the tag to remove.
         */
        template <typename TTag>
        void remove()
        {
            auto it = mTags.find(TTag::type().identifier());
            if (it != mTags.end())
            {
                mTags.erase(it);
            }
        }

        /**
         * @brief Tests if a tag is present.
         * @tparam TTag Type of tag to test.
         * @return True if the tag exists.
         */
        template <typename TTag>
        [[nodiscard]] inline bool has() const
        {
            return mTags.find(TTag::type().identifier()) != mTags.end();
        }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

    private:
        /**
         * @brief Collection of tags, indexed by their unique identifiers.
         */
        std::set<std::size_t> mTags;
    };
}
