#include <vislab/core/path_parameter.hpp>

#include <vislab/core/iarchive.hpp>

#include <filesystem>

namespace vislab
{
    PathParameter::PathParameter()
        : Concrete<PathParameter, Parameter<std::string>>()
        , mFilter("")
        , mFile(EFile::In)
    {
    }

    PathParameter::PathParameter(EFile fileDirection, const std::string& filter)
        : Concrete<PathParameter, Parameter<std::string>>()
        , mFilter(filter)
        , mFile(fileDirection)
    {
    }

    PathParameter::EFile PathParameter::getFileDirection() const
    {
        return mFile;
    }

    void PathParameter::setFileDirection(PathParameter::EFile file)
    {
        mFile = file;
    }

    const std::string& PathParameter::getFilter() const
    {
        return mFilter;
    }

    void PathParameter::setFilter(const std::string& filter)
    {
        mFilter = filter;
    }

    void PathParameter::serialize(IArchive& archive)
    {
        Parameter<std::string>::serialize(archive);
        archive("Filter", mFilter);
        archive("File", (int&)mFile);
    }

    bool PathParameter::operator==(const PathParameter& other) const
    {
        return Parameter<std::string>::operator==(other) && mFilter == other.mFilter && mFile == other.mFile;
    }

    bool PathParameter::isValid() const
    {
        if (mFile == EFile::In)
        {
            // if input, make sure that the file exists
            return std::filesystem::exists(this->mValue);
        }
        return true;
    }
}
