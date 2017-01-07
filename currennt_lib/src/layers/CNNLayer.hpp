// Obsolete
/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016
 *
 * This file is part of CURRENNT. 
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifndef LAYERS_CNNLAYER_HPP
#define LAYERS_CNNLAYER_HPP

#include "TrainableLayer.hpp"
#include <boost/shared_ptr.hpp>


namespace layers {

    /******************************************************************************************//*
     * CNN unit (the component for one feature Map) 
     *
     *
     *********************************************************************************************/
    template <typename TDevice>
    class CNNUnit
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;
    protected:
	
	// CNN filter parameter
	const int m_filterW;         //
	const int m_filterH;         //
	const int m_strideW;         // stride along width
	const int m_strideH;         // stride along height
	const int m_poolW;           // pooling along width
	const int m_poolH;           // pooling along height
	const int m_paddW;           // padding length for width
	const int m_paddH;           // padding length for height
	
	int m_inputW;                //
	int m_inputH;                //
	int m_inputC;                // number of channel of input
	
	// Note: CNN, # weight != layersize * preceding_layersize 
	//            
	const int m_weightDim;      // weight of this CNN layer
	const int m_outputDim;      // size of output
	
	Layer<TDevice> &m_precedingLayer;
	real_vector    m_featOutput;       // output of this feature map
	
    public:
	
	CNNUnit(const int width,   const int height,
		const int strideW, const int strideH,
		const int poolW,   const int poolH,
		const int paddW,   const int paddH,
		const int weiDim,  const int outDim,
		Layer<TDevice> &precedingLayer);

	virtual ~CNNUnit();

	virtual void computeForwardPass();
	
	virtual void computeBackwardPass();

    };

    /******************************************************************************************//**
     * CNN layer 
     *  including the normal convolutional layer and pooling layer
     *
     *  CNNFeatureUnit: a feature unit corresponds to a feature map
     *     ^
     *     |
     *  CNNLayer
     *********************************************************************************************/
    template <typename TDevice>
    class CNNLayer : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;

    protected:
	std::vector<boost::shared_ptr<CNNUnit<TDevice> > > m_CNNUnit;
	const int m_weightDim;      // weight of this CNN layer
	const int m_outputDim;      // size of output
	
    public:
	// initializer and destructor
	CNNLayer(const helpers::JsonValue &layerChild,
		 const helpers::JsonValue &weightsSection,
		 Layer<TDevice> &precedingLayer);

	virtual ~CNNLayer();

	virtual const std::string& type() const;
	
	virtual void computeForwardPass();
	
	virtual void computeBackwardPass();

        virtual void loadSequences(const data_sets::DataSetFraction &fraction);

	virtual void mergeOutput();
	
    };
    
    
}
#endif
