#include "vislab/core/algorithm.hpp"
#include "vislab/core/array.hpp"
#include "vislab/core/binary_input_archive.hpp"
#include "vislab/core/binary_output_archive.hpp"
#include "vislab/core/numeric_parameter.hpp"
#include "vislab/core/option_parameter.hpp"
#include "vislab/core/parameter.hpp"

#include "init_vislab.hpp"

#include "gtest/gtest.h"

namespace vislab
{
    class ContainerNative : public Concrete<ContainerNative, ISerializable>
    {
    public:
        /**
         * @brief Constructor.
         */
        ContainerNative() {}

        /**
         * @brief Destructor.
         */
        virtual ~ContainerNative() {}

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        virtual void serialize(IArchive& archive) override
        {
            archive("Bool", Bool);
            archive("Char", Char);
            archive("Float", Float);
            archive("Double", Double);
            archive("Int32", Int32);
            archive("UInt32", UInt32);
            archive("Int64", Int64);
            archive("UInt64", UInt64);
            archive("String", String);
            archive("Vector2i", Vector2i);
            archive("Vector3i", Vector3i);
            archive("Vector4i", Vector4i);
            archive("Vector2f", Vector2f);
            archive("Vector3f", Vector3f);
            archive("Vector4f", Vector4f);
            archive("Vector2d", Vector2d);
            archive("Vector3d", Vector3d);
            archive("Vector4d", Vector4d);
            archive("Matrix2i", Matrix2i);
            archive("Matrix3i", Matrix3i);
            archive("Matrix4i", Matrix4i);
            archive("Matrix2f", Matrix2f);
            archive("Matrix3f", Matrix3f);
            archive("Matrix4f", Matrix4f);
            archive("Matrix2d", Matrix2d);
            archive("Matrix3d", Matrix3d);
            archive("Matrix4d", Matrix4d);
            archive("MatrixXi", MatrixXi);
            archive("MatrixXf", MatrixXf);
            archive("MatrixXd", MatrixXd);
            archive("Quaternionf", Quaternionf);
            archive("Quaterniond", Quaterniond);
        }
        bool Bool;
        char Char;
        float Float;
        double Double;
        int32_t Int32;
        uint32_t UInt32;
        int64_t Int64;
        uint64_t UInt64;
        std::string String;
        Eigen::Vector2i Vector2i;
        Eigen::Vector3i Vector3i;
        Eigen::Vector4i Vector4i;
        Eigen::Vector2f Vector2f;
        Eigen::Vector3f Vector3f;
        Eigen::Vector4f Vector4f;
        Eigen::Vector2d Vector2d;
        Eigen::Vector3d Vector3d;
        Eigen::Vector4d Vector4d;
        Eigen::Matrix2i Matrix2i;
        Eigen::Matrix3i Matrix3i;
        Eigen::Matrix4i Matrix4i;
        Eigen::Matrix2f Matrix2f;
        Eigen::Matrix3f Matrix3f;
        Eigen::Matrix4f Matrix4f;
        Eigen::Matrix2d Matrix2d;
        Eigen::Matrix3d Matrix3d;
        Eigen::Matrix4d Matrix4d;
        Eigen::MatrixXi MatrixXi;
        Eigen::MatrixXf MatrixXf;
        Eigen::MatrixXd MatrixXd;
        Eigen::Quaternionf Quaternionf;
        Eigen::Quaterniond Quaterniond;
    };

}

namespace vislab
{
    template <typename TArchiveIn, typename TArchiveOut>
    void test_serialize_native()
    {
        ContainerNative containerIn, containerOut;
        containerOut.Bool        = false;
        containerOut.Char        = 'c';
        containerOut.Float       = 1.2f;
        containerOut.Double      = 2.3;
        containerOut.Int32       = 23;
        containerOut.UInt32      = 43;
        containerOut.Int64       = 64;
        containerOut.UInt64      = 65;
        containerOut.String      = "hello";
        containerOut.Vector2i    = Eigen::Vector2i(1, 2);
        containerOut.Vector3i    = Eigen::Vector3i(1, 2, 3);
        containerOut.Vector4i    = Eigen::Vector4i(1, 2, 3, 4);
        containerOut.Vector2f    = Eigen::Vector2f(1.1f, 2.1f);
        containerOut.Vector3f    = Eigen::Vector3f(1.1f, 2.1f, 3.1f);
        containerOut.Vector4f    = Eigen::Vector4f(1.1f, 2.1f, 3.1f, 4.1f);
        containerOut.Vector2d    = Eigen::Vector2d(1.2, 2.2);
        containerOut.Vector3d    = Eigen::Vector3d(1.2, 2.2, 3.2);
        containerOut.Vector4d    = Eigen::Vector4d(1.2, 2.2, 3.2, 4.2);
        containerOut.Matrix2i    = Eigen::Matrix2i({ { 1, 2 }, { 3, 4 } });
        containerOut.Matrix3i    = Eigen::Matrix3i({ { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        containerOut.Matrix4i    = Eigen::Matrix4i({ { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 }, { 13, 14, 15, 16 } });
        containerOut.Matrix2f    = Eigen::Matrix2f({ { 1, 2 }, { 3, 4 } });
        containerOut.Matrix3f    = Eigen::Matrix3f({ { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        containerOut.Matrix4f    = Eigen::Matrix4f({ { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 }, { 13, 14, 15, 16 } });
        containerOut.Matrix2d    = Eigen::Matrix2d({ { 1, 2 }, { 3, 4 } });
        containerOut.Matrix3d    = Eigen::Matrix3d({ { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
        containerOut.Matrix4d    = Eigen::Matrix4d({ { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 }, { 13, 14, 15, 16 } });
        containerOut.MatrixXi    = Eigen::Matrix<int, 3, 2>({ { 1, 2 }, { 3, 4 }, { 5, 6 } });
        containerOut.MatrixXf    = Eigen::Matrix<float, 3, 2>({ { 1, 2 }, { 3, 4 }, { 5, 6 } });
        containerOut.MatrixXd    = Eigen::Matrix<double, 3, 2>({ { 1, 2 }, { 3, 4 }, { 5, 6 } });
        containerOut.Quaternionf = Eigen::Quaternionf(4, 5, 6, 7);
        containerOut.Quaterniond = Eigen::Quaterniond(4, 5, 6, 7);

        TArchiveOut archiveOut;
        if (archiveOut.open("test.bin"))
            archiveOut("Container", containerOut);
        archiveOut.close();

        TArchiveIn archiveIn;
        if (archiveIn.open("test.bin"))
            archiveIn("Container", containerIn);
        archiveIn.close();

        EXPECT_EQ(containerIn.Bool, containerOut.Bool);
        EXPECT_EQ(containerIn.Char, containerOut.Char);
        EXPECT_EQ(containerIn.Float, containerOut.Float);
        EXPECT_EQ(containerIn.Double, containerOut.Double);
        EXPECT_EQ(containerIn.Int32, containerOut.Int32);
        EXPECT_EQ(containerIn.UInt32, containerOut.UInt32);
        EXPECT_EQ(containerIn.Int64, containerOut.Int64);
        EXPECT_EQ(containerIn.UInt64, containerOut.UInt64);
        EXPECT_EQ(containerIn.String, containerOut.String);
        EXPECT_EQ(containerIn.Vector2i, containerOut.Vector2i);
        EXPECT_EQ(containerIn.Vector3i, containerOut.Vector3i);
        EXPECT_EQ(containerIn.Vector4i, containerOut.Vector4i);
        EXPECT_EQ(containerIn.Vector2f, containerOut.Vector2f);
        EXPECT_EQ(containerIn.Vector3f, containerOut.Vector3f);
        EXPECT_EQ(containerIn.Vector4f, containerOut.Vector4f);
        EXPECT_EQ(containerIn.Vector2d, containerOut.Vector2d);
        EXPECT_EQ(containerIn.Vector3d, containerOut.Vector3d);
        EXPECT_EQ(containerIn.Vector4d, containerOut.Vector4d);
        EXPECT_EQ(containerIn.Matrix2i, containerOut.Matrix2i);
        EXPECT_EQ(containerIn.Matrix3i, containerOut.Matrix3i);
        EXPECT_EQ(containerIn.Matrix4i, containerOut.Matrix4i);
        EXPECT_EQ(containerIn.Matrix2f, containerOut.Matrix2f);
        EXPECT_EQ(containerIn.Matrix3f, containerOut.Matrix3f);
        EXPECT_EQ(containerIn.Matrix4f, containerOut.Matrix4f);
        EXPECT_EQ(containerIn.Matrix2d, containerOut.Matrix2d);
        EXPECT_EQ(containerIn.Matrix3d, containerOut.Matrix3d);
        EXPECT_EQ(containerIn.Matrix4d, containerOut.Matrix4d);
        EXPECT_EQ(containerIn.MatrixXi.coeff(0, 0), 1);
        EXPECT_EQ(containerIn.MatrixXi.coeff(0, 1), 2);
        EXPECT_EQ(containerIn.MatrixXi.coeff(1, 0), 3);
        EXPECT_EQ(containerIn.MatrixXi.coeff(1, 1), 4);
        EXPECT_EQ(containerIn.MatrixXi.coeff(2, 0), 5);
        EXPECT_EQ(containerIn.MatrixXi.coeff(2, 1), 6);
        EXPECT_EQ(containerIn.MatrixXf.coeff(0, 0), 1);
        EXPECT_EQ(containerIn.MatrixXf.coeff(0, 1), 2);
        EXPECT_EQ(containerIn.MatrixXf.coeff(1, 0), 3);
        EXPECT_EQ(containerIn.MatrixXf.coeff(1, 1), 4);
        EXPECT_EQ(containerIn.MatrixXf.coeff(2, 0), 5);
        EXPECT_EQ(containerIn.MatrixXf.coeff(2, 1), 6);
        EXPECT_EQ(containerIn.MatrixXd.coeff(0, 0), 1);
        EXPECT_EQ(containerIn.MatrixXd.coeff(0, 1), 2);
        EXPECT_EQ(containerIn.MatrixXd.coeff(1, 0), 3);
        EXPECT_EQ(containerIn.MatrixXd.coeff(1, 1), 4);
        EXPECT_EQ(containerIn.MatrixXd.coeff(2, 0), 5);
        EXPECT_EQ(containerIn.MatrixXd.coeff(2, 1), 6);
        EXPECT_EQ(containerIn.Quaternionf, containerOut.Quaternionf);
        EXPECT_EQ(containerIn.Quaterniond, containerOut.Quaterniond);
    }

    TEST(core, serialize_native)
    {
        Init();

        test_serialize_native<BinaryInputArchive, BinaryOutputArchive>();
    }

    class ContainerCollection : public Concrete<ContainerCollection, ISerializable>
    {
    public:
        ContainerCollection() {}
        virtual ~ContainerCollection() {}

        // Serializes the object into/from an archive.
        virtual void serialize(IArchive& archive) override
        {
            archive("Vector", Vector);
            archive("Vector2", Vector2);
            archive("Vector3", Vector3);
            archive("Map", Map);
            archive("Map2", Map2);
            archive("Map3", Map3);
            archive("UMap", UMap);
            archive("UMap2", UMap2);
            archive("UMap3", UMap3);
        }
        std::vector<int> Vector;
        std::vector<ColorParameter> Vector2;
        std::vector<std::shared_ptr<ColorParameter>> Vector3;
        std::map<int, float> Map;
        std::map<int, ColorParameter> Map2;
        std::map<int, std::shared_ptr<ColorParameter>> Map3;
        std::unordered_map<int, float> UMap;
        std::unordered_map<int, ColorParameter> UMap2;
        std::unordered_map<int, std::shared_ptr<ColorParameter>> UMap3;
    };

    template <typename TArchiveIn, typename TArchiveOut>
    void test_serialize_collection()
    {
        ContainerCollection containerIn, containerOut;
        containerOut.Vector.resize(3);
        containerOut.Vector[0] = 11;
        containerOut.Vector[1] = 12;
        containerOut.Vector[2] = 13;
        containerOut.Vector2.resize(2);
        containerOut.Vector2[0].setValue(Eigen::Vector4f(11, 11, 11, 11));
        containerOut.Vector2[0].setMinValue(Eigen::Vector4f(12, 12, 12, 12));
        containerOut.Vector2[0].setMaxValue(Eigen::Vector4f(13, 13, 13, 13));
        containerOut.Vector2[1].setValue(Eigen::Vector4f(21, 21, 21, 21));
        containerOut.Vector2[1].setMinValue(Eigen::Vector4f(22, 22, 22, 22));
        containerOut.Vector2[1].setMaxValue(Eigen::Vector4f(23, 23, 23, 23));
        containerOut.Vector3.resize(2);
        containerOut.Vector3[0] = nullptr;
        containerOut.Vector3[1] = std::make_shared<ColorParameter>();
        containerOut.Vector3[1]->setValue(Eigen::Vector4f(1.2f, 1.2f, 1.2f, 1.2f));
        containerOut.Vector3[1]->setMinValue(Eigen::Vector4f(0.2f, 0.2f, 0.2f, 0.2f));
        containerOut.Vector3[1]->setMaxValue(Eigen::Vector4f(2.2f, 2.2f, 2.2f, 2.2f));
        containerOut.Map.insert(std::make_pair(1, 1.1f));
        containerOut.Map.insert(std::make_pair(2, 1.2f));
        containerOut.Map.insert(std::make_pair(3, 1.3f));
        containerOut.Map2.insert(std::make_pair(1, containerOut.Vector2[0]));
        containerOut.Map2.insert(std::make_pair(2, containerOut.Vector2[1]));
        containerOut.Map3.insert(std::make_pair(1, containerOut.Vector3[0]));
        containerOut.Map3.insert(std::make_pair(2, containerOut.Vector3[1]));
        containerOut.UMap.insert(std::make_pair(1, 1.1f));
        containerOut.UMap.insert(std::make_pair(2, 1.2f));
        containerOut.UMap.insert(std::make_pair(3, 1.3f));
        containerOut.UMap2.insert(std::make_pair(1, containerOut.Vector2[0]));
        containerOut.UMap2.insert(std::make_pair(2, containerOut.Vector2[1]));
        containerOut.UMap3.insert(std::make_pair(1, containerOut.Vector3[0]));
        containerOut.UMap3.insert(std::make_pair(2, containerOut.Vector3[1]));

        TArchiveOut archiveOut;
        if (archiveOut.open("test.bin"))
            archiveOut("Container", containerOut);
        archiveOut.close();

        TArchiveIn archiveIn;
        if (archiveIn.open("test.bin"))
            archiveIn("Container", containerIn);
        archiveIn.close();

        EXPECT_EQ(containerIn.Vector.size(), containerOut.Vector.size());
        EXPECT_EQ(containerIn.Vector[0], containerOut.Vector[0]);
        EXPECT_EQ(containerIn.Vector[1], containerOut.Vector[1]);
        EXPECT_EQ(containerIn.Vector[2], containerOut.Vector[2]);
        EXPECT_EQ(containerIn.Vector2.size(), containerOut.Vector2.size());
        EXPECT_EQ(containerIn.Vector2[0], containerOut.Vector2[0]);
        EXPECT_EQ(containerIn.Vector2[1], containerOut.Vector2[1]);
        EXPECT_EQ(containerIn.Vector3.size(), containerOut.Vector3.size());
        EXPECT_TRUE(containerIn.Vector3[0] == nullptr);
        EXPECT_EQ(*containerIn.Vector3[1].get(), *containerOut.Vector3[1].get());
        EXPECT_EQ(containerIn.Map, containerOut.Map);
        EXPECT_EQ(containerIn.Map2.find(1)->second, containerOut.Vector2[0]);
        EXPECT_EQ(containerIn.Map2.find(2)->second, containerOut.Vector2[1]);
        EXPECT_EQ(containerIn.Map3.find(1)->second, containerOut.Vector3[0]);
        EXPECT_EQ(*containerIn.Map3.find(2)->second.get(), *containerOut.Vector3[1].get());
        EXPECT_EQ(containerIn.UMap, containerOut.UMap);
        EXPECT_EQ(containerIn.UMap2.find(1)->second, containerOut.Vector2[0]);
        EXPECT_EQ(containerIn.UMap2.find(2)->second, containerOut.Vector2[1]);
        EXPECT_EQ(containerIn.UMap3.find(1)->second, containerOut.Vector3[0]);
        EXPECT_EQ(*containerIn.UMap3.find(2)->second.get(), *containerOut.Vector3[1].get());
    }

    TEST(core, serialize_collection)
    {
        Init();

        EXPECT_TRUE(Factory::create(BinaryInputArchive::type()) != nullptr);
        EXPECT_TRUE(Factory::create(BinaryOutputArchive::type()) != nullptr);

        test_serialize_collection<BinaryInputArchive, BinaryOutputArchive>();
    }
}
