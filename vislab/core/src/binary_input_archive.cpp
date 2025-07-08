#include <vislab/core/binary_input_archive.hpp>

namespace vislab
{
    BinaryInputArchive::BinaryInputArchive()
    {
    }

    BinaryInputArchive::BinaryInputArchive(const BinaryInputArchive& other)
    {
    }

    BinaryInputArchive::~BinaryInputArchive()
    {
        close();
    }

    bool BinaryInputArchive::open(const char* path)
    {
        close();
        mStream.open(path, std::ios_base::in | std::ios_base::binary);
        return mStream.is_open();
    }

    void BinaryInputArchive::close()
    {
        if (mStream.is_open())
            mStream.close();
    }

    void BinaryInputArchive::operator()(const char* name, bool& value)
    {
        char boolean;
        mStream.read((char*)std::addressof(boolean), sizeof(char));
        value = (boolean == 1);
    }
    void BinaryInputArchive::operator()(const char* name, char& value) { mStream.read((char*)std::addressof(value), sizeof(char)); }
    void BinaryInputArchive::operator()(const char* name, float& value) { mStream.read((char*)std::addressof(value), sizeof(float)); }
    void BinaryInputArchive::operator()(const char* name, double& value) { mStream.read((char*)std::addressof(value), sizeof(double)); }
    void BinaryInputArchive::operator()(const char* name, int16_t& value) { mStream.read((char*)std::addressof(value), sizeof(int16_t)); }
    void BinaryInputArchive::operator()(const char* name, uint16_t& value) { mStream.read((char*)std::addressof(value), sizeof(uint16_t)); }
    void BinaryInputArchive::operator()(const char* name, int32_t& value) { mStream.read((char*)std::addressof(value), sizeof(int32_t)); }
    void BinaryInputArchive::operator()(const char* name, uint32_t& value) { mStream.read((char*)std::addressof(value), sizeof(uint32_t)); }
    void BinaryInputArchive::operator()(const char* name, int64_t& value) { mStream.read((char*)std::addressof(value), sizeof(int64_t)); }
    void BinaryInputArchive::operator()(const char* name, uint64_t& value) { mStream.read((char*)std::addressof(value), sizeof(uint64_t)); }
    void BinaryInputArchive::operator()(const char* name, std::string& value)
    {
        std::size_t size;
        mStream.read((char*)std::addressof(size), sizeof(std::size_t));
        if (size > 0)
        {
            value.resize(size);
            mStream.read((char*)value.data(), sizeof(char) * size);
        }
    }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector<int16_t, 1>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector2<int16_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector3<int16_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector4<int16_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::VectorX<int16_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector<int, 1>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector2i& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector3i& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector4i& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::VectorXi& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector<int64_t, 1>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector2<int64_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector3<int64_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector4<int64_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::VectorX<int64_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector<uint16_t, 1>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector2<uint16_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector3<uint16_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector4<uint16_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::VectorX<uint16_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector<uint32_t, 1>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector2<uint32_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector3<uint32_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector4<uint32_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::VectorX<uint32_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector<uint64_t, 1>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector2<uint64_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector3<uint64_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector4<uint64_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::VectorX<uint64_t>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector<float, 1>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector2f& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector3f& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector4f& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::VectorXf& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector<double, 1>& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector2d& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector3d& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Vector4d& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::VectorXd& value) { processVector(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2<int16_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix2<int16_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3<int16_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix3<int16_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4<int16_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix4<int16_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2i& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix2i)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3i& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix3i)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4i& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix4i)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2<int64_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix2<int64_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3<int64_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix3<int64_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4<int64_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix4<int64_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2<uint16_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix2<uint16_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3<uint16_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix3<uint16_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4<uint16_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix4<uint16_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2<uint32_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix2<uint32_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3<uint32_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix3<uint32_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4<uint32_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix4<uint32_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2<uint64_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix2<uint64_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3<uint64_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix3<uint64_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4<uint64_t>& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix4<uint64_t>)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2f& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix2f)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3f& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix3f)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4f& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix4f)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2d& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix2d)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3d& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix3d)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4d& value) { mStream.read((char*)value.data(), sizeof(Eigen::Matrix4d)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::MatrixX<int16_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::MatrixXi& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::MatrixX<int64_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::MatrixX<uint16_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::MatrixX<uint32_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::MatrixX<uint64_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::MatrixXf& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::MatrixXd& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix<int16_t, 1, Eigen::Dynamic>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix<int32_t, 1, Eigen::Dynamic>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix<uint16_t, 1, Eigen::Dynamic>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix<uint32_t, 1, Eigen::Dynamic>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix<uint64_t, 1, Eigen::Dynamic>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix<float, 1, Eigen::Dynamic>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix<double, 1, Eigen::Dynamic>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2X<int16_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2Xi& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2X<int64_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2X<uint16_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2X<uint32_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2X<uint64_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2Xf& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix2Xd& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3X<int16_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3Xi& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3X<int64_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3X<uint16_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3X<uint32_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3X<uint64_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3Xf& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix3Xd& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4X<int16_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4Xi& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4X<int64_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4X<uint16_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4X<uint32_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4X<uint64_t>& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4Xf& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Matrix4Xd& value) { processMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::SparseMatrix<int16_t>& value) { processSparseMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::SparseMatrix<int32_t>& value) { processSparseMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::SparseMatrix<int64_t>& value) { processSparseMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::SparseMatrix<uint16_t>& value) { processSparseMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::SparseMatrix<uint32_t>& value) { processSparseMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::SparseMatrix<uint64_t>& value) { processSparseMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::SparseMatrix<float>& value) { processSparseMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::SparseMatrix<double>& value) { processSparseMatrix(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::AlignedBox1f& value) { processAlignedBox(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::AlignedBox1d& value) { processAlignedBox(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::AlignedBox2f& value) { processAlignedBox(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::AlignedBox2d& value) { processAlignedBox(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::AlignedBox3f& value) { processAlignedBox(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::AlignedBox3d& value) { processAlignedBox(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::AlignedBox4f& value) { processAlignedBox(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::AlignedBox4d& value) { processAlignedBox(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::AlignedBoxXf& value) { processAlignedBox(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::AlignedBoxXd& value) { processAlignedBox(name, value); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Quaternionf& value) { mStream.read((char*)value.coeffs().data(), sizeof(Eigen::Vector4f)); }
    void BinaryInputArchive::operator()(const char* name, Eigen::Quaterniond& value) { mStream.read((char*)value.coeffs().data(), sizeof(Eigen::Vector4d)); }
    void BinaryInputArchive::operator()(const char* name, ISerializable& value)
    {
        value.serialize(*this);
    }

    void BinaryInputArchive::stateStarted(const char* name, IArchive::EState state)
    {
    }
    void BinaryInputArchive::stateEnded(const char* name, IArchive::EState state)
    {
    }
    void BinaryInputArchive::archiveIndex(size_t& index)
    {
        operator()("Index", index);
    }
    void BinaryInputArchive::archiveSize(size_t& size)
    {
        operator()("Size", size);
    }
    bool BinaryInputArchive::isInputArchive()
    {
        return true;
    }
}
