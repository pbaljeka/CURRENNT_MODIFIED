/****
 *
 *
 *
 ****/


#include "SkipLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"

namespace layers{

    // Construct the layer
    template <typename TDevice>
    SkipLayer<TDevice>::SkipLayer(
					const helpers::JsonValue &layerChild,
					const helpers::JsonValue &weightsSection,
					std::vector<Layer<TDevice>*> precedingLayers
					)
	// use preLayers[0] as fake preceding layers
	: TrainableLayer<TDevice>(layerChild, weightsSection, 1, 0, *(precedingLayers.back()))
    {
	m_outputErrorsFromSkipLayer = Cpu::real_vector(this->outputs().size(), (real_t)0.0);
    }	

    // Destructor
    template <typename TDevice>
    SkipLayer<TDevice>::~SkipLayer()
    {
    }
    
    template <typename TDevice>
    typename SkipLayer<TDevice>::real_vector& SkipLayer<TDevice>::outputFromGate()
    {
	printf("WARNING: Output from SkipLayer, WRONG!\n");
	return m_outputErrorsFromSkipLayer;
    }
    
    template <typename TDevice>
    typename SkipLayer<TDevice>::real_vector& SkipLayer<TDevice>::outputErrorsFromSkipLayer()
    {
        return m_outputErrorsFromSkipLayer;
    }

    template class SkipLayer<Cpu>;
    template class SkipLayer<Gpu>;
    
}
