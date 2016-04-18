/****
 *
 *
 *
 ****/

#ifndef LAYERS_SKIPPARALAYER_HPP
#define LAYERS_SKIPPARALAYER_HPP

#include "SkipLayer.hpp"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace layers {
    
    /**********************************************************************
	    Definition of the Skip Add layer
     

     **********************************************************************/
    
    // class definition
    template <typename TDevice, typename TActFn>
    class SkipParaLayer : public SkipLayer<TDevice>
    {
	typedef typename TDevice::real_vector    real_vector;
        
    private:
	// the preceding skipping layer
	//  currently, only one previous skip layer is allowed 
	//  this is different from skipaddlayer, because I don't want to assign another
	//  vector to accumulate the input from multiple previous skip layers
	Layer<TDevice>    *m_preSkipLayer;

	// to receive the errors directly from next skip add layer
	// real_vector       m_outputErrorsFromSkipLayer;

	// the gate 
	real_vector       m_gateOutput;
        // the gate error
	real_vector       m_gateErrors; // error before the actFn of gate unit
        
    public:
	
	
	// Construct the layer
	SkipParaLayer(
		     const helpers::JsonValue &layerChild,
		     const helpers::JsonValue &weightsSection,
		     std::vector<Layer<TDevice>*> precedingLayers
		     );

	// Destructor
	virtual ~SkipParaLayer();
	
	// void 
	virtual const std::string& type() const;

	// NN forward
	virtual void computeForwardPass();
	
	// NN backward
	virtual void computeBackwardPass();
	
	// Gate output
	real_vector& outputFromGate();
	
	
	// return all the preceding layers
	Layer<TDevice>* preSkipLayer();

	// output of the gate unit
	real_vector& gateOutput();
	
	// 
	real_vector& gateErrors();
	
    };

}


#endif 


