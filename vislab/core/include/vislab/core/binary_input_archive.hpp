#pragma once

#include "iinput_archive.hpp"

#include <fstream>

namespace vislab
{
    /**
     * @brief Archive that reads from a binary file. Note that reads and writes must be in consistent order, since everything is read/written consecutively! The "name" of variables is not used!
     */
    class BinaryInputArchive : public Concrete<BinaryInputArchive, IInputArchive>
    {
    public:
        /**
         * @brief Constructor.
         */
        BinaryInputArchive();

        /**
         * @brief Copy-Constructor. (Does not use the same file handle.)
         */
        BinaryInputArchive(const BinaryInputArchive& other);

        /**
         * @brief Destructor (closes the archive).
         */
        ~BinaryInputArchive() override;

        /**
         * @brief Opens the archive.
         * @param path Path to the file to open.
         * @return True if opening was successful.
         */
        [[nodiscard]] bool open(const char* path);

        /**
         * @brief Closes the archive.
         */
        void close();

        /**
         * @brief Archives a bool variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, bool& value) override;

        /**
         * @brief Archives a char variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, char& value) override;

        /**
         * @brief Archives a float variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, float& value) override;

        /**
         * @brief Archives a double variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, double& value) override;

        /**
         * @brief Archives a 16-bit signed integer variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, int16_t& value) override;

        /**
         * @brief Archives a 16-bit unsigned integer variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, uint16_t& value) override;

        /**
         * @brief Archives a 32-bit signed integer variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, int32_t& value) override;

        /**
         * @brief Archives a 32-bit unsigned integer variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, uint32_t& value) override;

        /**
         * @brief Archives a 64-bit signed integer variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, int64_t& value) override;

        /**
         * @brief Archives a 64-bit unsigned integer variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, uint64_t& value) override;

        /**
         * @brief Archives a std::string variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, std::string& value) override;

        /**
         * @brief Archives an Eigen::Vector<int16_t, 1> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<int16_t, 1>& value) override;

        /**
         * @brief Archives an Eigen::Vector<int16_t, 2> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<int16_t, 2>& value) override;

        /**
         * @brief Archives an Eigen::Vector<int16_t, 3> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<int16_t, 3>& value) override;

        /**
         * @brief Archives an Eigen::Vector<int16_t, 4> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<int16_t, 4>& value) override;

        /**
         * @brief Archives an Eigen::Vector<int16_t, -1> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<int16_t, -1>& value) override;

        /**
         * @brief Archives an Eigen::Vector1i variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<int, 1>& value) override;

        /**
         * @brief Archives an Eigen::Vector2i variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector2i& value) override;

        /**
         * @brief Archives an Eigen::Vector3i variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector3i& value) override;

        /**
         * @brief Archives an Eigen::Vector4i variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector4i& value) override;

        /**
         * @brief Archives an Eigen::VectorXi variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::VectorXi& value) override;

        /**
         * @brief Archives an Eigen::Vector<int64_t, 1> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<int64_t, 1>& value) override;

        /**
         * @brief Archives an Eigen::Vector<int64_t, 2> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<int64_t, 2>& value) override;

        /**
         * @brief Archives an Eigen::Vector<int64_t, 3> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<int64_t, 3>& value) override;

        /**
         * @brief Archives an Eigen::Vector<int64_t, 4> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<int64_t, 4>& value) override;

        /**
         * @brief Archives an Eigen::Vector<int64_t, -1> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<int64_t, -1>& value) override;

        /**
         * @brief Archives an Eigen::Vector<uint16_t, 1> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<uint16_t, 1>& value) override;

        /**
         * @brief Archives an Eigen::Vector<uint16_t, 2> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<uint16_t, 2>& value) override;

        /**
         * @brief Archives an Eigen::Vector<uint16_t, 3> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<uint16_t, 3>& value) override;

        /**
         * @brief Archives an Eigen::Vector<uint16_t, 4> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<uint16_t, 4>& value) override;

        /**
         * @brief Archives an Eigen::Vector<uint16_t, -1> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<uint16_t, -1>& value) override;

        /**
         * @brief Archives an Eigen::Vector1u variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<uint32_t, 1>& value) override;

        /**
         * @brief Archives an Eigen::Vector2u variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector2<uint32_t>& value) override;

        /**
         * @brief Archives an Eigen::Vector3u variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector3<uint32_t>& value) override;

        /**
         * @brief Archives an Eigen::Vector4u variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector4<uint32_t>& value) override;

        /**
         * @brief Archives an Eigen::VectorXu variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::VectorX<uint32_t>& value) override;

        /**
         * @brief Archives an Eigen::Vector<uint64_t, 1> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<uint64_t, 1>& value) override;

        /**
         * @brief Archives an Eigen::Vector<uint64_t, 2> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<uint64_t, 2>& value) override;

        /**
         * @brief Archives an Eigen::Vector<uint64_t, 3> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<uint64_t, 3>& value) override;

        /**
         * @brief Archives an Eigen::Vector<uint64_t, 4> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<uint64_t, 4>& value) override;

        /**
         * @brief Archives an Eigen::Vector<uint64_t, -1> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<uint64_t, -1>& value) override;

        /**
         * @brief Archives an Eigen::Vector1f variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<float, 1>& value) override;

        /**
         * @brief Archives an Eigen::Vector2f variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector2f& value) override;

        /**
         * @brief Archives an Eigen::Vector3f variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector3f& value) override;

        /**
         * @brief Archives an Eigen::Vector4f variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector4f& value) override;

        /**
         * @brief Archives an Eigen::VectorXf variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::VectorXf& value) override;

        /**
         * @brief Archives an Eigen::Vector1d variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector<double, 1>& value) override;

        /**
         * @brief Archives an Eigen::Vector2d variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector2d& value) override;

        /**
         * @brief Archives an Eigen::Vector3d variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector3d& value) override;

        /**
         * @brief Archives an Eigen::Vector4d variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Vector4d& value) override;

        /**
         * @brief Archives an Eigen::VectorXd variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::VectorXd& value) override;

        /**
         * @brief Archives an Eigen::Matrix<int16_t, 2, 2> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<int16_t, 2, 2>& value) override;

        /**
         * @brief Archives an Eigen::Matrix<int16_t, 3, 3> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<int16_t, 3, 3>& value) override;

        /**
         * @brief Archives an Eigen::Matrix<int16_t, 4, 4> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<int16_t, 4, 4>& value) override;

        /**
         * @brief Archives an Eigen::Matrix2i variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix2i& value) override;

        /**
         * @brief Archives an Eigen::Matrix3i variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix3i& value) override;

        /**
         * @brief Archives an Eigen::Matrix4i variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix4i& value) override;

        /**
         * @brief Archives an Eigen::Matrix<int64_t, 2, 2> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<int64_t, 2, 2>& value) override;

        /**
         * @brief Archives an Eigen::Matrix<int64_t, 3, 3> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<int64_t, 3, 3>& value) override;

        /**
         * @brief Archives an Eigen::Matrix<int64_t, 4, 4> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<int64_t, 4, 4>& value) override;

        /**
         * @brief Archives an Eigen::Matrix<uint16_t, 2, 2> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<uint16_t, 2, 2>& value) override;

        /**
         * @brief Archives an Eigen::Matrix<uint16_t, 3, 3> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<uint16_t, 3, 3>& value) override;

        /**
         * @brief Archives an Eigen::Matrix<uint16_t, 4, 4> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<uint16_t, 4, 4>& value) override;

        /**
         * @brief Archives an Eigen::Matrix2u variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix2<uint32_t>& value) override;

        /**
         * @brief Archives an Eigen::Matrix3u variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix3<uint32_t>& value) override;

        /**
         * @brief Archives an Eigen::Matrix4u variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix4<uint32_t>& value) override;

        /**
         * @brief Archives an Eigen::Matrix<uint64_t, 2, 2> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<uint64_t, 2, 2>& value) override;

        /**
         * @brief Archives an Eigen::Matrix<uint64_t, 3, 3> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<uint64_t, 3, 3>& value) override;

        /**
         * @brief Archives an Eigen::Matrix<uint64_t, 4, 4> variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<uint64_t, 4, 4>& value) override;

        /**
         * @brief Archives an Eigen::Matrix2f variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix2f& value) override;

        /**
         * @brief Archives an Eigen::Matrix3f variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix3f& value) override;

        /**
         * @brief Archives an Eigen::Matrix4f variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix4f& value) override;

        /**
         * @brief Archives an Eigen::Matrix2d variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix2d& value) override;

        /**
         * @brief Archives an Eigen::Matrix3d variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix3d& value) override;

        /**
         * @brief Archives an Eigen::Matrix4d variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix4d& value) override;

        /**
         * @brief Archives a variable-size signed 16-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::MatrixX<int16_t>& value) override;

        /**
         * @brief Archives a variable-size signed 32-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::MatrixXi& value) override;

        /**
         * @brief Archives a variable-size signed 64-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::MatrixX<int64_t>& value) override;

        /**
         * @brief Archives a variable-size unsigned 16-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::MatrixX<uint16_t>& value) override;

        /**
         * @brief Archives a variable-size unsigned 32-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::MatrixX<uint32_t>& value) override;

        /**
         * @brief Archives a variable-size unsigned 64-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::MatrixX<uint64_t>& value) override;

        /**
         * @brief Archives a variable-size float Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::MatrixXf& value) override;

        /**
         * @brief Archives a variable-size double Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::MatrixXd& value) override;

        /**
         * @brief Archives a variable-column signed 16-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<int16_t, 1, Eigen::Dynamic>& value) override;

        /**
         * @brief Archives a variable-column signed 32-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<int32_t, 1, Eigen::Dynamic>& value) override;

        /**
         * @brief Archives a variable-column signed 64-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& value) override;

        /**
         * @brief Archives a variable-column unsigned 16-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<uint16_t, 1, Eigen::Dynamic>& value) override;

        /**
         * @brief Archives a variable-column unsigned 32-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<uint32_t, 1, Eigen::Dynamic>& value) override;

        /**
         * @brief Archives a variable-column unsigned 64-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<uint64_t, 1, Eigen::Dynamic>& value) override;

        /**
         * @brief Archives a variable-column float Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<float, 1, Eigen::Dynamic>& value) override;

        /**
         * @brief Archives a variable-column double Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix<double, 1, Eigen::Dynamic>& value) override;

        /**
         * @brief Archives a variable-column signed 16-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix2X<int16_t>& value) override;

        /**
         * @brief Archives a variable-column signed 32-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix2Xi& value) override;

        /**
         * @brief Archives a variable-column signed 64-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix2X<int64_t>& value) override;

        /**
         * @brief Archives a variable-column unsigned 16-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix2X<uint16_t>& value) override;

        /**
         * @brief Archives a variable-column unsigned 32-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix2X<uint32_t>& value) override;

        /**
         * @brief Archives a variable-column unsigned 64-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix2X<uint64_t>& value) override;

        /**
         * @brief Archives a variable-column float Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix2Xf& value) override;

        /**
         * @brief Archives a variable-column double Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix2Xd& value) override;

        /**
         * @brief Archives a variable-column signed 16-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix3X<int16_t>& value) override;

        /**
         * @brief Archives a variable-column signed 32-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix3Xi& value) override;

        /**
         * @brief Archives a variable-column signed 64-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix3X<int64_t>& value) override;

        /**
         * @brief Archives a variable-column unsigned 16-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix3X<uint16_t>& value) override;

        /**
         * @brief Archives a variable-column unsigned 32-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix3X<uint32_t>& value) override;

        /**
         * @brief Archives a variable-column unsigned 64-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix3X<uint64_t>& value) override;

        /**
         * @brief Archives a variable-column float Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix3Xf& value) override;

        /**
         * @brief Archives a variable-column double Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix3Xd& value) override;

        /**
         * @brief Archives a variable-column signed 16-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix4X<int16_t>& value) override;

        /**
         * @brief Archives a variable-column signed 32-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix4Xi& value) override;

        /**
         * @brief Archives a variable-column signed 64-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix4X<int64_t>& value) override;

        /**
         * @brief Archives a variable-column unsigned 16-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix4X<uint16_t>& value) override;

        /**
         * @brief Archives a variable-column unsigned 32-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix4X<uint32_t>& value) override;

        /**
         * @brief Archives a variable-column unsigned 64-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix4X<uint64_t>& value) override;

        /**
         * @brief Archives a variable-column float Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix4Xf& value) override;

        /**
         * @brief Archives a variable-column double Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Matrix4Xd& value) override;

        /**
         * @brief Archives a sparse signed 16-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::SparseMatrix<int16_t>& value) override;

        /**
         * @brief Archives a sparse signed 32-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::SparseMatrix<int32_t>& value) override;

        /**
         * @brief Archives a sparse signed 64-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::SparseMatrix<int64_t>& value) override;

        /**
         * @brief Archives a sparse unsigned 16-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::SparseMatrix<uint16_t>& value) override;

        /**
         * @brief Archives a sparse unsigned 32-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::SparseMatrix<uint32_t>& value) override;

        /**
         * @brief Archives a sparse unsigned 64-bit integer Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::SparseMatrix<uint64_t>& value) override;

        /**
         * @brief Archives a sparse float Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::SparseMatrix<float>& value) override;

        /**
         * @brief Archives a sparse double Eigen::Matrix variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::SparseMatrix<double>& value) override;

        /**
         * @brief Archives an Eigen::AlignedBoxXf variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::AlignedBoxXf& value) override;

        /**
         * @brief Archives an Eigen::AlignedBoxXd variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::AlignedBoxXd& value) override;

        /**
         * @brief Archives an Eigen::AlignedBox1f variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::AlignedBox1f& value) override;

        /**
         * @brief Archives an Eigen::AlignedBox1d variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::AlignedBox1d& value) override;

        /**
         * @brief Archives an Eigen::AlignedBox2f variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::AlignedBox2f& value) override;

        /**
         * @brief Archives an Eigen::AlignedBox2d variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::AlignedBox2d& value) override;

        /**
         * @brief Archives an Eigen::AlignedBox3f variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::AlignedBox3f& value) override;

        /**
         * @brief Archives an Eigen::AlignedBox3d variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::AlignedBox3d& value) override;

        /**
         * @brief Archives an Eigen::AlignedBox4f variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::AlignedBox4f& value) override;

        /**
         * @brief Archives an Eigen::AlignedBox4d variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::AlignedBox4d& value) override;

        /**
         * @brief Archives an Eigen::Quaternionf variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Quaternionf& value) override;

        /**
         * @brief Archives an Eigen::Quaterniond variable.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, Eigen::Quaterniond& value) override;

        /**
         * @brief Archives a serializable variable recursively.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        void operator()(const char* name, ISerializable& value) override;

    protected:

        /**
         * @brief Notifies this archive that the given state has started.
         * @param name
         * @param state
         */
        void stateStarted(const char* name, EState state) override;

        /**
         * @brief Notifies this archive that the given state has ended.
         * @param name
         * @param state
         */
        void stateEnded(const char* name, EState state) override;

        /**
         * @brief Archive an index.
         * @param index
         */
        void archiveIndex(size_t& index) override;

        /**
         * @brief Archive a size.
         * @param size
         */
        void archiveSize(size_t& size) override;

        /**
         * @brief Check if this is an input archive.
         * @return
         */
        bool isInputArchive() override;

    private:
        /**
         * @brief Writes an Eigen::Vector type.
         * @tparam TValue Scalar type inside the Eigen::Vector.
         * @tparam Size Number of elements in the Eigen::Vector.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        template <typename TValue, int Size>
        void processVector(const char* name, Eigen::Vector<TValue, Size>& value)
        {
            if (Size == Eigen::Dynamic)
            {
                Eigen::Index size;
                mStream.read((char*)&size, sizeof(Eigen::Index));
                value.resize(size);
            }
            mStream.read((char*)value.data(), sizeof(TValue) * Size);
        }

        /**
         * @brief Reads an Eigen::Matrix type.
         * @tparam TValue Scalar type inside the Eigen::Matrix.
         * @tparam Rows Number of rows.
         * @tparam Cols Number of columns.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        template <typename TValue, int Rows, int Cols>
        void processMatrix(const char* name, Eigen::Matrix<TValue, Rows, Cols>& value)
        {
            Eigen::Index rows, cols;
            mStream.read((char*)&rows, sizeof(Eigen::Index));
            mStream.read((char*)&cols, sizeof(Eigen::Index));
            value.resize(rows, cols);
            mStream.read((char*)value.data(), sizeof(TValue) * rows * cols);
        }

        /**
         * @brief Reads an Eigen::SparseMatrix type
         * @tparam TValue Scalar type inside the Eigen::SparseMatrix.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        template <typename TValue>
        void processSparseMatrix(const char* name, Eigen::SparseMatrix<TValue>& value)
        {
            Eigen::Index rows, cols, nnz;
            mStream.read((char*)&rows, sizeof(Eigen::Index));
            mStream.read((char*)&cols, sizeof(Eigen::Index));
            mStream.read((char*)&nnz, sizeof(Eigen::Index));
            typedef Eigen::Triplet<TValue> T;
            std::vector<T> tripletList(nnz);
            mStream.read((char*)tripletList.data(), sizeof(T) * nnz);
            value = Eigen::SparseMatrix<TValue>(rows, cols);
            value.setFromTriplets(tripletList.begin(), tripletList.end());
        }

        /**
         * @brief Reads an Eigen::AlignedBox type
         * @tparam TValue Scalar type inside the Eigen::AlignedBox.
         * @tparam Size Number of dimensions.
         * @param name Name of the variable to archive.
         * @param value Value to archive.
         */
        template <typename TValue, int Size>
        void processAlignedBox(const char* name, Eigen::AlignedBox<TValue, Size>& value)
        {
            processVector("Min", value.min());
            processVector("Max", value.max());
        }

        /**
         * @brief Stream to read from.
         */
        std::ifstream mStream;
    };
}
