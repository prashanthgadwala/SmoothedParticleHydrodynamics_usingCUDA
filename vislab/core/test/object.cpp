#include "vislab/core/object.hpp"

#include "init_vislab.hpp"

#include "gtest/gtest.h"

#include "vislab/core/iserializable.hpp"

namespace vislab
{
    class A : public Interface<A, Object, ISerializable>
    {
    public:
        int a;
    };

    class B : public Interface<B, A>
    {
    public:
        int b;
    };

    class C : public Concrete<C, B>
    {
    public:
        int c;

        void serialize(IArchive& archive) override
        {
        }
    };

    /**
     * @brief This function tests whether a type can be constructed and cloned when held in a pointer to a base class.
     * @tparam TBase Base class to hold the derived type in.
     * @tparam TDerived Derived type.
     * @param derived Instance of the derived type to do the test on.
     */
    template <typename TBase, typename TDerived>
    static void core_object_test(std::shared_ptr<TDerived> derived)
    {
        // test of the clone function
        {
            // hold in base pointer
            std::shared_ptr<TBase> base = derived;

            // clone the object
            std::shared_ptr<TBase> base2 = base->clone();

            // cast cloned back to derived type
            auto derived2 = std::dynamic_pointer_cast<C>(base2);
            EXPECT_EQ(derived->a, derived2->a);
            EXPECT_EQ(derived->b, derived2->b);
            EXPECT_EQ(derived->c, derived2->c);
        }

        // test of getType function
        {
            // hold in base pointer
            std::shared_ptr<TBase> base = derived;

            // create the object from type of base
            std::shared_ptr<Object> base2 = Factory::create(base->getType());

            // cast constructed object to derived type
            auto derived2 = std::dynamic_pointer_cast<C>(base2);
            EXPECT_TRUE(derived2 != nullptr);
        }
    }

    TEST(core, object)
    {
        Init();

        // add custom types to reflection
        meta::named_reflect<A>("A").template base<Object>();
        meta::named_reflect<B>("B").template base<A>();
        meta::named_reflect<C>("C").template base<B>().ctor();

        // allocate derived class with factory using reflection (we let the factory do the cast to "C", but we could it afterwards, too)
        auto derived = Factory::create<C>("C");
        derived->a   = 1;
        derived->b   = 2;
        derived->c   = 3;

        // test if "derived" can be cloned/created when held in a pointer of a certain base type
        core_object_test<A>(derived);
        core_object_test<ISerializable>(derived);
    }
}
