#include "feature_selector.h"

FeatureSelector::FeatureSelector(Estimator& estimator)
: estimator_(estimator)
{

}

// ----------------------------------------------------------------------------

void FeatureSelector::processImage(const image_t& image, const std_msgs::Header& header)
{
  estimator_.processImage(image, header);
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------
