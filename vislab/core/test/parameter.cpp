#include "vislab/core/numeric_parameter.hpp"
#include "vislab/core/path_parameter.hpp"

#include "init_vislab.hpp"

#include "gtest/gtest.h"

namespace vislab
{
    TEST(core, parameter_factory)
    {
        Init();

        EXPECT_TRUE(Factory::create(Int32Parameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(FloatParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(DoubleParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Int64Parameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(StringParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(PathParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Vec2iParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Vec2fParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Vec2dParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Vec3iParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Vec3fParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Vec3dParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Vec4iParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Vec4fParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(Vec4dParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction1fParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction1dParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction2fParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction2dParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction3fParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction3dParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction4fParameter::type()) != nullptr);
        EXPECT_TRUE(Factory::create(TransferFunction4dParameter::type()) != nullptr);
    }

    TEST(core, parameter)
    {
        // create a parameter
        auto floatParameter = std::make_unique<FloatParameter>();

        // define an event listener listening to parameter changes and record if a change was reported
        float listened = 0;
        floatParameter->onChange += [=, &listened](Parameter<float>* sender, const float* value)
        {
            listened = sender->getValue();
        };

        // set a testing value
        const float testValue = 1.2f;
        floatParameter->setValue(testValue);

        // see if the event fired successfully and whether the getter gets the correct result
        EXPECT_EQ(testValue, listened);
        EXPECT_EQ(testValue, floatParameter->getValue());

        // unhook the listener
        floatParameter->onChange.clearAll();

        // set another test value
        const float testValue2 = 3.2f;
        floatParameter->setValue(testValue2);

        // see if no event was fired and whether the getter gets the correct result
        EXPECT_NE(testValue2, listened);
        EXPECT_EQ(testValue2, floatParameter->getValue());
    }
}
