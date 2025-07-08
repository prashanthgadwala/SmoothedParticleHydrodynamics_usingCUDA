#pragma once

#include "types.hpp"

#include <functional>
#include <list>
#include <mutex>
#include <string>

namespace vislab
{
    /**
     * @brief An event class implementing the observer pattern for std::functions.
     *
     * Example usage:
     *		Event myEvent;
     *		myEvent += [=](void* sender) { printf("test"); };
     *		myEvent += std::bind(&global_function, ptr_to_sender);
     *		myEvent += std::bind(&Class::member_function, this, ptr_to_sender);
     *
     * @tparam TSender Type of the event sender.
     * @tparam TArgs Arguments that are send when invoking the event.
     */
    template <typename TSender, typename TArgs>
    class TEvent
    {
    public:
        /**
         * @brief Constructor.
         */
        TEvent() = default;

        /**
         * @brief Copy constructor that does not clone the observers.
         * @param other Other event to copy from.
         */
        TEvent(const TEvent& other) {}

        /**
         * @brief Signature of a callback function
         */
        using Callback = std::function<void(TSender*, const TArgs*)>;

        /**
         * @brief A handle used to identify an observer
         */
        using Handle = typename std::list<Callback>::iterator;

        /**
         * @brief Type of the sender parameter in the notify call
         */
        using SenderPtr = TSender*;

        /**
         * @brief Type of the argument parameter in the notify call
         */
        using ArgsPtr = const TArgs*;

        /**
         * @brief Sends a notification out to all observers (sending out a sender is optional)
         * @param sender The sender that raises the event.
         * @param args The arguments that are send along with the event.
         */
        void notify(SenderPtr sender = nullptr, ArgsPtr args = nullptr) const
        {
            for (auto& observer : mObservers)
                observer(sender, args);
        }

        /**
         * @brief Registers an observer.
         * @param observer Callback to call when the event is raised.
         * @return A handle that makes it possible to unregister the observer later.
         */
        Handle registerCallback(Callback observer)
        {
            std::lock_guard<std::mutex> lock(mObservablesMutex);
            return mObservers.insert(mObservers.end(), observer);
        }

        /**
         * @brief Deregisters an observer by the handle obtained at registration.
         * @param handle Handle to remove from the observer list.
         */
        void deregisterCallback(Handle& handle)
        {
            std::lock_guard<std::mutex> lock(mObservablesMutex);
            mObservers.erase(handle);
        }

        /**
         * @brief Deregisters all observers.
         */
        void clearAll()
        {
            std::lock_guard<std::mutex> lock(mObservablesMutex);
            mObservers.clear();
        }

        /**
         * @brief Operator to add an observer
         * @param fn Callback function to add to the observer list.
         * @return Handle that can be used fo remove the observer.
         */
        Handle operator+=(Callback fn) { return registerCallback(fn); }

        /**
         * @brief Operator to remove an observer by the handle obtained at registration.
         * @param handle
         */
        void operator-=(Handle& handle) { deregisterCallback(handle); }

    private:
        /**
         * @brief List of all observers.
         */
        std::list<Callback> mObservers;

        /**
         * @brief Mutex for thread safety.
         */
        std::mutex mObservablesMutex;
    };

    using Event            = TEvent<void, void>;
    using Int32Event       = TEvent<void, int32_t>;
    using Int64Event       = TEvent<void, int64_t>;
    using FloatEvent       = TEvent<void, float>;
    using DoubleEvent      = TEvent<void, double>;
    using BoolEvent        = TEvent<void, bool>;
    using StringEvent      = TEvent<void, std::string>;
    using ColorEvent       = TEvent<void, Eigen::Vector4f>;
    using Vec2fEvent       = TEvent<void, Eigen::Vector2f>;
    using Vec3fEvent       = TEvent<void, Eigen::Vector3f>;
    using Vec4fEvent       = TEvent<void, Eigen::Vector4f>;
    using Vec2dEvent       = TEvent<void, Eigen::Vector2d>;
    using Vec3dEvent       = TEvent<void, Eigen::Vector3d>;
    using Vec4dEvent       = TEvent<void, Eigen::Vector4d>;
    using Vec2iEvent       = TEvent<void, Eigen::Vector2i>;
    using Vec3iEvent       = TEvent<void, Eigen::Vector3i>;
    using Vec4iEvent       = TEvent<void, Eigen::Vector4i>;
    using StringArrayEvent = TEvent<void, std::vector<std::string>>;
}
