/****
 *
 *
 *
 ****/

#ifndef LAYERS_SKIPLAYER_HPP
#define LAYERS_SKIPLAYER_HPP

#include "TrainableLayer.hpp"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace layers {
    
    /**********************************************************************
	    Definition of the Skip layer
     A base class for SkipAdd and SkipPara layers

     **********************************************************************/
    
    // class definition
    template <typename TDevice>
    class SkipLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector    real_vector;
        
    private:
	// all the preceding skipping layers
	// std::vector<Layer<TDevice>*> m_preLayers;
	// to receive the errors directly from next skip add layer
	real_vector       m_outputErrorsFromSkipLayer;
        
    public:
	
	
	// Construct the layer
	SkipLayer(
		     const helpers::JsonValue &layerChild,
		     const helpers::JsonValue &weightsSection,
		     std::vector<Layer<TDevice>*> precedingLayers
		     );

	// Destructor
	virtual ~SkipLayer();
	
	// void 
	//virtual const std::string& type() const;

	// NN forward
	//virtual void computeForwardPass();
	
	// NN backward
	//virtual void computeBackwardPass();
	
	// return all the preceding layers
	//std::vector<Layer<TDevice>*> PreLayers();
	
	virtual real_vector& outputFromGate();
	
	// return reference to the m_outputErrorsFromSkipLayer
	real_vector& outputErrorsFromSkipLayer();
    };

}


#endif 


