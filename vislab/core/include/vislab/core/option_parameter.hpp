#pragma once

#include "parameter.hpp"

namespace vislab
{
    /**
     * @brief Parameter that selects an option from a list of values (strings).
     */
    class OptionParameter : public Concrete<OptionParameter, Parameter<int32_t>>
    {
    public:
        /**
         * @brief Constructor.
         */
        OptionParameter();

        /**
         * @brief Constructor.
         * @param value Initial value.
         * @param labels Labels of the options.
         */
        OptionParameter(int32_t value, const std::vector<std::string>& labels);

        /**
         * @brief Destructor.
         */
        virtual ~OptionParameter() = default;

        /**
         * @brief Copy-assignment operator. Event listeners are not copied.
         * @param other Other parameter to copy from.
         * @return Reference to self.
         */
        OptionParameter& operator=(const OptionParameter& other)
        {
            Parameter<int32_t>::operator=(other);
            mLabels   = other.mLabels;
            return *this;
        }

        /**
         * @brief Gets the list of labels.
         * @return Vector with labels for the different options.
         */
        [[nodiscard]] const std::vector<std::string>& getLabels() const;

        /**
         * @brief Sets the list of labels.
         * @param newLabels New list of labels to set.
         * @param notifyOnChange Flag that determines whether an event is raised to all listeners.
         */
        void setLabels(const std::vector<std::string>& newLabels, bool notifyOnChange = true);

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Gets the currently selected label.
         * @return The currently selected label.
         */
        std::string getCurrentLabel() const;

        /**
         * @brief Tests if another parameter stores the exact same values and whether they have the same visibilty.
         * @param other Other parameter to compare with.
         * @return True, if the values and visibility are the same.
         */
        bool operator==(const OptionParameter& other) const;

        /**
         * @brief Event to be raised when the labels change.
         */
        StringArrayEvent onLabelsChange;

    private:
        /**
         * @brief Vector containing the labels for the different options.
         */
        std::vector<std::string> mLabels;
    };
}
