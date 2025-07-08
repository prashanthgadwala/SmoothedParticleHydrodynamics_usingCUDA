#pragma once

#include "iserializable.hpp"
#include "istringifiable.hpp"
#include "object.hpp"
#include "vislab_type_traits.hpp"

#include <Eigen/Eigen>
#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <unordered_map>
#include <vector>

namespace vislab
{
    /**
     * @brief Abstract base class for any kind of archive.
     */
    class IArchive : public Object
    {
    public:
        /**
         * @enum Type
         *
         * @brief Encodes the different types an archive currently serializes. Handling of
         * these is up to the implementing child class.
         */
        enum EState
        {
            /**
             * @brief The default type of object.
             */
            Group,

            /**
             * @brief An ordered list of items.
             */
            Array,

            /**
             * @brief Item in an array.
             */
            Item,

            /**
             * @brief An unordered set of key value pairs with primitive (string type) keys.
             */
            PrimitiveMap,

            /**
             * @brief Indicates a map entry in a primitive map.
             */
            PrimitiveEntry,

            /**
             * @brief An unordered set of key value pairs with complex (object type) keys.
             */
            ComplexMap,

            /**
             * @brief Indicates a map entry in a complex map.
             */
            ComplexEntry,

        };

        /**
         * @brief Initialize archive with given initial state.
         *
         * @param initialState
         */
        IArchive(EState initialState);

        /**
         * @brief Initialize archive with Object state.
         */
        IArchive();

        /**
         * @brief Indicates that a new type of object is being serialized.
         * @param state
         */
        void beginState(const char* name, EState state);

        /**
         * @brief Indicates that the current type of object is finished.
         * @param state
         */
        void endState(const char* name, EState state);

        /**
         * @brief Check if this is an input archive.
         * @return
         */
        virtual bool isInputArchive() = 0;

        /**
         * @brief Archives a bool variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, bool& value) = 0;

        /**
         * @brief Archives a char variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, char& value) = 0;

        /**
         * @brief Archives a float variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, float& value) = 0;

        /**
         * @brief Archives a double variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, double& value) = 0;

        /**
         * @brief Archives a 16-bit signed integer variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, int16_t& value) = 0;

        /**
         * @brief Archives a 16-bit unsigned integer variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, uint16_t& value) = 0;

        /**
         * @brief Archives a 32-bit signed integer variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, int32_t& value) = 0;

        /**
         * @brief Archives a 32-bit unsigned integer variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, uint32_t& value) = 0;

        /**
         * @brief Archives a 64-bit signed integer variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, int64_t& value) = 0;

        /**
         * @brief Archives a 64-bit unsigned integer variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, uint64_t& value) = 0;

        /**
         * @brief Archives a std::string variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, std::string& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<int16_t, 1> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<int16_t, 1>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<int16_t, 2> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<int16_t, 2>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<int16_t, 3> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<int16_t, 3>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<int16_t, 4> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<int16_t, 4>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<int16_t, -1> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<int16_t, -1>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector1i variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<int, 1>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector2i variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector2i& value) = 0;

        /**
         * @brief Archives an Eigen::Vector3i variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector3i& value) = 0;

        /**
         * @brief Archives an Eigen::Vector4i variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector4i& value) = 0;

        /**
         * @brief Archives an Eigen::VectorXi variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::VectorXi& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<int64_t, 1> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<int64_t, 1>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<int64_t, 2> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<int64_t, 2>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<int64_t, 3> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<int64_t, 3>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<int64_t, 4> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<int64_t, 4>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<int64_t, -1> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<int64_t, -1>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<uint16_t, 1> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<uint16_t, 1>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<uint16_t, 2> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<uint16_t, 2>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<uint16_t, 3> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<uint16_t, 3>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<uint16_t, 4> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<uint16_t, 4>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<uint16_t, -1> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<uint16_t, -1>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector1u variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<uint32_t, 1>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector2u variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector2<uint32_t>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector3u variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector3<uint32_t>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector4u variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector4<uint32_t>& value) = 0;

        /**
         * @brief Archives an Eigen::VectorXu variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::VectorX<uint32_t>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<uint64_t, 1> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<uint64_t, 1>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<uint64_t, 2> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<uint64_t, 2>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<uint64_t, 3> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<uint64_t, 3>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<uint64_t, 4> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<uint64_t, 4>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector<uint64_t, -1> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<uint64_t, -1>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector1f variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<float, 1>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector2f variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector2f& value) = 0;

        /**
         * @brief Archives an Eigen::Vector3f variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector3f& value) = 0;

        /**
         * @brief Archives an Eigen::Vector4f variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector4f& value) = 0;

        /**
         * @brief Archives an Eigen::VectorXf variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::VectorXf& value) = 0;

        /**
         * @brief Archives an Eigen::Vector1d variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector<double, 1>& value) = 0;

        /**
         * @brief Archives an Eigen::Vector2d variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector2d& value) = 0;

        /**
         * @brief Archives an Eigen::Vector3d variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector3d& value) = 0;

        /**
         * @brief Archives an Eigen::Vector4d variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Vector4d& value) = 0;

        /**
         * @brief Archives an Eigen::VectorXd variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::VectorXd& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix<int16_t, 2, 2> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<int16_t, 2, 2>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix<int16_t, 3, 3> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<int16_t, 3, 3>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix<int16_t, 4, 4> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<int16_t, 4, 4>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix2i variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix2i& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix3i variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix3i& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix4i variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix4i& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix<int64_t, 2, 2> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<int64_t, 2, 2>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix<int64_t, 3, 3> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<int64_t, 3, 3>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix<int64_t, 4, 4> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<int64_t, 4, 4>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix<uint16_t, 2, 2> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<uint16_t, 2, 2>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix<uint16_t, 3, 3> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<uint16_t, 3, 3>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix<uint16_t, 4, 4> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<uint16_t, 4, 4>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix2u variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<uint32_t, 2, 2>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix3u variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<uint32_t, 3, 3>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix4u variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<uint32_t, 4, 4>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix<uint64_t, 2, 2> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<uint64_t, 2, 2>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix<uint64_t, 3, 3> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<uint64_t, 3, 3>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix<uint64_t, 4, 4> variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<uint64_t, 4, 4>& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix2f variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix2f& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix3f variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix3f& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix4f variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix4f& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix2d variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix2d& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix3d variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix3d& value) = 0;

        /**
         * @brief Archives an Eigen::Matrix4d variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix4d& value) = 0;

        /**
         * @brief Archives a variable-size signed 16-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::MatrixX<int16_t>& value) = 0;

        /**
         * @brief Archives a variable-size signed 32-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::MatrixXi& value) = 0;

        /**
         * @brief Archives a variable-size signed 64-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::MatrixX<int64_t>& value) = 0;

        /**
         * @brief Archives a variable-size unsigned 16-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::MatrixX<uint16_t>& value) = 0;

        /**
         * @brief Archives a variable-size unsigned 32-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::MatrixX<uint32_t>& value) = 0;

        /**
         * @brief Archives a variable-size unsigned 64-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::MatrixX<uint64_t>& value) = 0;

        /**
         * @brief Archives a variable-size float Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::MatrixXf& value) = 0;

        /**
         * @brief Archives a variable-size double Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::MatrixXd& value) = 0;

        /**
         * @brief Archives a variable-column signed 16-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<int16_t, 1, Eigen::Dynamic>& value) = 0;

        /**
         * @brief Archives a variable-column signed 32-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<int32_t, 1, Eigen::Dynamic>& value) = 0;

        /**
         * @brief Archives a variable-column signed 64-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& value) = 0;

        /**
         * @brief Archives a variable-column unsigned 16-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<uint16_t, 1, Eigen::Dynamic>& value) = 0;

        /**
         * @brief Archives a variable-column unsigned 32-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<uint32_t, 1, Eigen::Dynamic>& value) = 0;

        /**
         * @brief Archives a variable-column unsigned 64-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<uint64_t, 1, Eigen::Dynamic>& value) = 0;

        /**
         * @brief Archives a variable-column float Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<float, 1, Eigen::Dynamic>& value) = 0;

        /**
         * @brief Archives a variable-column double Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix<double, 1, Eigen::Dynamic>& value) = 0;

        /**
         * @brief Archives a variable-column signed 16-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix2X<int16_t>& value) = 0;

        /**
         * @brief Archives a variable-column signed 32-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix2Xi& value) = 0;

        /**
         * @brief Archives a variable-column signed 64-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix2X<int64_t>& value) = 0;

        /**
         * @brief Archives a variable-column unsigned 16-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix2X<uint16_t>& value) = 0;

        /**
         * @brief Archives a variable-column unsigned 32-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix2X<uint32_t>& value) = 0;

        /**
         * @brief Archives a variable-column unsigned 64-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix2X<uint64_t>& value) = 0;

        /**
         * @brief Archives a variable-column float Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix2Xf& value) = 0;

        /**
         * @brief Archives a variable-column double Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix2Xd& value) = 0;

        /**
         * @brief Archives a variable-column signed 16-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix3X<int16_t>& value) = 0;

        /**
         * @brief Archives a variable-column signed 32-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix3Xi& value) = 0;

        /**
         * @brief Archives a variable-column signed 64-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix3X<int64_t>& value) = 0;

        /**
         * @brief Archives a variable-column unsigned 16-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix3X<uint16_t>& value) = 0;

        /**
         * @brief Archives a variable-column unsigned 32-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix3X<uint32_t>& value) = 0;

        /**
         * @brief Archives a variable-column unsigned 64-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix3X<uint64_t>& value) = 0;

        /**
         * @brief Archives a variable-column float Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix3Xf& value) = 0;

        /**
         * @brief Archives a variable-column double Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix3Xd& value) = 0;

        /**
         * @brief Archives a variable-column signed 16-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix4X<int16_t>& value) = 0;

        /**
         * @brief Archives a variable-column signed 32-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix4Xi& value) = 0;

        /**
         * @brief Archives a variable-column signed 64-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix4X<int64_t>& value) = 0;

        /**
         * @brief Archives a variable-column unsigned 16-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix4X<uint16_t>& value) = 0;

        /**
         * @brief Archives a variable-column unsigned 32-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix4X<uint32_t>& value) = 0;

        /**
         * @brief Archives a variable-column unsigned 64-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix4X<uint64_t>& value) = 0;

        /**
         * @brief Archives a variable-column float Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix4Xf& value) = 0;

        /**
         * @brief Archives a variable-column double Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Matrix4Xd& value) = 0;

        /**
         * @brief Archives a sparse signed 16-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::SparseMatrix<int16_t>& value) = 0;

        /**
         * @brief Archives a sparse signed 32-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::SparseMatrix<int32_t>& value) = 0;

        /**
         * @brief Archives a sparse signed 64-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::SparseMatrix<int64_t>& value) = 0;

        /**
         * @brief Archives a sparse unsigned 16-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::SparseMatrix<uint16_t>& value) = 0;

        /**
         * @brief Archives a sparse unsigned 32-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::SparseMatrix<uint32_t>& value) = 0;

        /**
         * @brief Archives a sparse unsigned 64-bit integer Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::SparseMatrix<uint64_t>& value) = 0;

        /**
         * @brief Archives a sparse float Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::SparseMatrix<float>& value) = 0;

        /**
         * @brief Archives a sparse double Eigen::Matrix variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::SparseMatrix<double>& value) = 0;

        /**
         * @brief Archives an Eigen::Quaternionf variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Quaternionf& value) = 0;

        /**
         * @brief Archives an Eigen::Quaterniond variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::Quaterniond& value) = 0;

        /**
         * @brief Archives an Eigen::AlignedBox1f variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::AlignedBox1f& value) = 0;

        /**
         * @brief Archives an Eigen::AlignedBox1d variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::AlignedBox1d& value) = 0;

        /**
         * @brief Archives an Eigen::AlignedBox2f variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::AlignedBox2f& value) = 0;

        /**
         * @brief Archives an Eigen::AlignedBox2d variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::AlignedBox2d& value) = 0;

        /**
         * @brief Archives an Eigen::AlignedBox3f variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::AlignedBox3f& value) = 0;

        /**
         * @brief Archives an Eigen::AlignedBox3d variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::AlignedBox3d& value) = 0;

        /**
         * @brief Archives an Eigen::AlignedBox4f variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::AlignedBox4f& value) = 0;

        /**
         * @brief Archives an Eigen::AlignedBox4d variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::AlignedBox4d& value) = 0;

        /**
         * @brief Archives an Eigen::AlignedBoxXf variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::AlignedBoxXf& value) = 0;

        /**
         * @brief Archives an Eigen::AlignedBoxXd variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, Eigen::AlignedBoxXd& value) = 0;

        /**
         * @brief Archives a serializable variable recursively.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        virtual void operator()(const char* name, ISerializable& value) = 0;

        /**
         * @brief Archives a StringSerializable variable.
         * @param name Unique name of the variable in the archive.
         * @param value Value of the variable.
         */
        void operator()(const char* name, IStringifiable& value)
        {
            std::string asString = value.toString();
            operator()(name, asString);
            value.fromString(asString);
        };

        /**
         * @brief Archives containers.
         * @tparam T Container type.
         * @param name Unique name of the container.
         * @param values container to serialize.
         */
        template <typename T, typename std::enable_if_t<traits::is_container_type_v<T>, int> = 0>
        void operator()(const char* name, T& container)
        {
            if constexpr (traits::is_map_type_v<T> || traits::is_unordered_map_type_v<T>)
            {
                if constexpr (std::is_base_of_v<IStringifiable, typename T::key_type> || std::is_base_of_v<std::string, typename T::key_type>)
                {
                    constexpr auto fromStringConversion = [](const std::string& str)
                    {
                        if constexpr (std::is_base_of_v<IStringifiable, typename T::key_type>)
                        {
                            typename T::key_type key;
                            key.fromString(str);
                            return key;
                        }
                        else
                        {
                            return str;
                        }
                    };

                    constexpr auto toStringConversion = [](const typename T::key_type& key)
                    {
                        if constexpr (std::is_base_of_v<IStringifiable, typename T::key_type>)
                        {
                            return key.toString();
                        }
                        else
                        {
                            return key;
                        }
                    };

                    beginState(name, EState::PrimitiveMap);
                    size_t size = container.size();
                    archiveSize(size);

                    // deserialize
                    if (isInputArchive())
                    {
                        for (size_t index = 0; index < size; index++)
                        {
                            std::string keyAsString;
                            typename T::mapped_type value;
                            beginState("MapEntry", EState::PrimitiveEntry);
                            operator()("Key", keyAsString);
                            operator()("Value", value);
                            endState("MapEntry", EState::PrimitiveEntry);
                            container[fromStringConversion(keyAsString)] = value;
                        }
                    }
                    // serialize
                    else
                    {
                        for (std::pair<const typename T::key_type, typename T::mapped_type> element : container)
                        {
                            beginState("MapEntry", EState::PrimitiveEntry);
                            std::string keyAsString = toStringConversion(element.first);
                            operator()("Key", keyAsString);
                            operator()("Value", element.second);
                            endState("MapEntry", EState::PrimitiveEntry);
                        }
                    }

                    endState(name, EState::PrimitiveMap);
                }
                else
                {
                    beginState(name, EState::ComplexMap);
                    size_t size = container.size();
                    archiveSize(size);

                    // deserialize
                    if (isInputArchive())
                    {
                        for (size_t index = 0; index < size; index++)
                        {
                            typename T::key_type key;
                            typename T::mapped_type value;
                            beginState("MapEntry", EState::ComplexEntry);
                            operator()("Key", key);
                            operator()("Value", value);
                            endState("MapEntry", EState::ComplexEntry);
                            container[key] = value;
                        }
                    }
                    // serialize
                    else
                    {
                        for (std::pair<const typename T::key_type, typename T::mapped_type> element : container)
                        {
                            beginState("MapEntry", EState::ComplexEntry);
                            typename T::key_type key      = element.first;
                            typename T::mapped_type value = element.second;
                            operator()("Key", key);
                            operator()("Value", value);
                            endState("MapEntry", EState::ComplexEntry);
                        }
                    }

                    endState(name, EState::ComplexMap);
                }
            }
            else
            {
                beginState(name, EState::Array);
                size_t size = container.size();
                archiveSize(size);
                if constexpr (traits::is_array_type_v<T>)
                {
                    //if (size != std::extent<T>::value)
                    //{
                    //    throw std::runtime_error("Size mismatch");
                    //}
                }
                else if constexpr (!traits::is_set_type_v<T>)
                {
                    container.resize(size);
                }

                size_t idx = 0;
                typename T::iterator it;
                for (it = container.begin(), idx = 0; idx < size; it++, idx++)
                {
                    beginState("Item", EState::Item);
                    archiveIndex(idx);

                    if constexpr (traits::is_set_type_v<T>)
                    {
                        auto nodeHandle = isInputArchive() ? typename T::node_type() : container.extract(it);
                        if (nodeHandle.empty())
                        {
                            typename T::node_type::value_type value = typename T::node_type::value_type();
                            operator()("Value", value);
                            container.insert(value);
                        }
                        else
                        {
                            typename T::node_type::value_type& value = nodeHandle.value();
                            operator()("Value", value);
                            container.insert(std::move(nodeHandle));
                        }
                    }
                    else
                    {
                        operator()("Value", *it);
                    }
                    endState("Item", EState::Item);
                }

                endState(name, EState::Array);
            }
        }

        /**
         * @brief Archives a Pointer to a serializable object recursively.
         * @param name Unique name of the variable.
         * @param value Pointer to object.
         */
        template <typename T, typename std::enable_if_t<std::is_pointer_v<T> || traits::is_smart_ptr_v<T>, int> = 0>
        void operator()(const char* name, T& value)
        {
            beginState(name, EState::Group);
            // before the pointer, we store whether it was nullptr
            bool isNull = !value;
            operator()("IsNull", isNull);
            std::string metaTypeName;
            if (!isInputArchive() && value)
            {
                // get the type
                Type metaType = value->getType();
                std::hash<std::string_view> hash;
                metaTypeName = std::string(metaType.prop(hash("name")).value().template cast<std::string>());
            }
            // if pointer does exist, then store its data
            if (!isNull)
            {
                operator()("MetaType", metaTypeName);
                if (isInputArchive())
                {
                    // i hope this does not break things
                    if constexpr (std::is_pointer_v<T>)
                    {
                        value = dynamic_cast<T>(allocate(metaTypeName).get()); // Not sure if that works for all pointers
                    }
                    else
                    {
                        // shared_ptr should never be copied or moved into a unique_ptr
                        static_assert(traits::is_shared_ptr_v<T> || traits::is_weak_ptr_v<T>, "If you ever find yourself in a situation where you need such thing you should set fire to your workstation - probably your whole house - and work on a new program design.");

                        value = std::move(std::dynamic_pointer_cast<typename T::element_type>(allocate(metaTypeName)));
                    }
                }
                operator()("Object", *value);
            }
            endState(name, EState::Group);
        }

    protected:
        /**
         * @brief Notify child class about the start of a new state.
         * @param name
         * @param state the state that has started.
         */
        virtual void stateStarted(const char* name, EState state) = 0;

        /**
         * @brief Notify child class about the end of the current state.
         * @param name
         * @param state the state that has ended.
         */
        virtual void stateEnded(const char* name, EState state) = 0;

        virtual void archiveIndex(size_t& index) = 0;

        virtual void archiveSize(size_t& size) = 0;

    private:
        std::stack<EState> mStates;

        std::stack<std::string> mNames;
        /**
         * @brief Allocates an object using the factory.
         * @param uid Unique type name.
         * @return Allocated object of the requested type.
         */
        std::shared_ptr<vislab::Object> allocate(const std::string& uid);
    };
}
