#pragma once

#include <array>
#include <map>
#include <memory>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace vislab
{
    namespace traits
    {
#define SHORTCUTS(TraitName)                           \
    template <typename T>                              \
    using TraitName##_t = typename TraitName<T>::type; \
    template <typename T>                              \
    inline constexpr bool TraitName##_v = TraitName<T>::value;
        template <typename, typename _ = void>
        struct is_map_type : std::false_type
        {
        };

        template <typename Key, typename T, typename Compare, typename Allocator>
        struct is_map_type<std::map<Key, T, Compare, Allocator>> : std::true_type
        {
        };

        SHORTCUTS(is_map_type)

        template <typename, typename _ = void>
        struct is_unordered_map_type : std::false_type
        {
        };
        template <typename Key, typename T, typename Compare, typename Allocator>
        struct is_unordered_map_type<std::unordered_map<Key, T, Compare, Allocator>> : std::true_type
        {
        };

        SHORTCUTS(is_unordered_map_type)

        template <typename, typename _ = void>
        struct is_set_type : std::false_type
        {
        };

        template <typename Key, typename Compare, typename Allocator>
        struct is_set_type<std::set<Key, Compare, Allocator>> : std::true_type
        {
        };

        SHORTCUTS(is_set_type)

        template <typename, typename _ = void>
        struct is_vector_type : std::false_type
        {
        };

        template <typename T, typename Allocator>
        struct is_vector_type<std::vector<T, Allocator>> : std::true_type
        {
        };

        SHORTCUTS(is_vector_type)

        template <typename, typename _ = void>
        struct is_array_type : std::false_type
        {
        };

        template <typename T, std::size_t N>
        struct is_array_type<std::array<T, N>> : std::true_type
        {
        };

        SHORTCUTS(is_array_type)

        template <typename, typename _ = void>
        struct is_container_type : std::false_type
        {
        };

        template <typename T, std::size_t N>
        struct is_container_type<std::array<T, N>> : std::true_type
        {
        };

        template <typename... Args>
        struct is_container_type<std::vector<Args...>> : std::true_type
        {
        };

        template <typename... Args>
        struct is_container_type<std::set<Args...>> : std::true_type
        {
        };

        template <typename... Args>
        struct is_container_type<std::map<Args...>> : std::true_type
        {
        };

        template <typename... Args>
        struct is_container_type<std::unordered_map<Args...>> : std::true_type
        {
        };

        SHORTCUTS(is_container_type)

        template <typename, typename _ = void>
        struct is_shared_ptr : std::false_type
        {
        };

        template <typename T>
        struct is_shared_ptr<std::shared_ptr<T>> : std::true_type
        {
        };

        SHORTCUTS(is_shared_ptr)

        template <typename, typename _ = void>
        struct is_unique_ptr : std::false_type
        {
        };

        template <typename T>
        struct is_unique_ptr<std::unique_ptr<T>> : std::true_type
        {
        };

        SHORTCUTS(is_unique_ptr)

        template <typename, typename _ = void>
        struct is_weak_ptr : std::false_type
        {
        };

        template <typename T>
        struct is_weak_ptr<std::weak_ptr<T>> : std::true_type
        {
        };

        SHORTCUTS(is_weak_ptr)

        template <typename, typename _ = void>
        struct is_smart_ptr : std::false_type
        {
        };

        template <typename T>
        struct is_smart_ptr<T, std::enable_if_t<is_shared_ptr_v<T> || is_unique_ptr_v<T> || is_weak_ptr_v<T>>> : std::true_type
        {
        };

        SHORTCUTS(is_smart_ptr)

#undef SHORTCUTS
    }
}
