#pragma once

#include "parameter.hpp"

namespace vislab
{
    /**
     * @brief Parameter that represents the path to a file.
     */
    class PathParameter : public Concrete<PathParameter, Parameter<std::string>>
    {
    public:
        /**
         * @brief Flag that determines whether the path is to be read or written to.
         */
        enum class EFile
        {
            /**
             * @brief Read from the path.
             */
            In,

            /**
             * @brief Write to the path.
             */
            Out
        };

        /**
         * @brief Constructor.
         */
        PathParameter();

        /**
         * @brief Constructor.
         */
        PathParameter(EFile fileDirection, const std::string& filter);

        /**
         * @brief Gets the file reading direction.
         * @return File reading direction.
         */
        [[nodiscard]] EFile getFileDirection() const;

        /**
         * @brief Sets the file reading direction.
         * @param file New file reading direction to set.
         */
        void setFileDirection(EFile file);

        /**
         * @brief Gets a filter string for the UI, which restricts to certain files with a certain extension.
         * @return Filter string for the UI.
         */
        [[nodiscard]] const std::string& getFilter() const;

        /**
         * @brief Sets a filter string for the UI, which restricts to certain files with a certain extension.
         * @param filter New filter string to set.
         */
        void setFilter(const std::string& filter);

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Tests if another parameter stores the exact same values and whether they have the same visibilty.
         * @param other Other parameter to compare with.
         * @return True, if the values and visibility are the same.
         */
        [[nodiscard]] bool operator==(const PathParameter& other) const;

        /**
         * @brief Tests if the data object is fully initialized and ready for use.
         * @return True if the data object is in a valid state.
         */
        [[nodiscard]] bool isValid() const override;



    private:
        /**
         * @brief Filter string, which controls the file extensions that can be selected.
         */
        std::string mFilter;

        /**
         * @brief File reading direction, which is in or out.
         */
        EFile mFile;
    };
}
