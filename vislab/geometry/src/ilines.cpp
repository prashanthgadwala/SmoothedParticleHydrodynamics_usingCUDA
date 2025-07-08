#include <vislab/geometry/ilines.hpp>

namespace vislab
{
    std::shared_ptr<ILine> ILines::getLine(std::size_t index)
    {
        return getLineImpl(index);
    }

    std::shared_ptr<const ILine> ILines::getLine(std::size_t index) const
    {
        return getLineImpl(index);
    }

    void ILines::addLine(std::shared_ptr<ILine> line) { addLineImpl(line); }
}
