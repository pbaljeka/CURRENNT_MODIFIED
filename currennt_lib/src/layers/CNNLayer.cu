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

/*
  
  Incomplete implementation
  
*/




#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif


#include "CNNLayer.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/min.cuh"
#include "../helpers/max.cuh"
#include "../helpers/safeExp.cuh"
#include "../helpers/JsonClasses.hpp"

#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"
#include "../activation_functions/Relu.cuh"

#include "../Configuration.hpp"

#include <boost/foreach.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <fstream>
#include <cmath>

#define DEBUG_LOCAL_CNN 1


#define CNNOPTIONSPLITER1 "#"
#define CNNOPTIONSPLITER2 "_"

namespace interal{
namespace{

    
} // namespace 
} // namespace internal

namespace CNNTools{
    
    // calculate the dimension of output features for each feature map (without pooling)
    int featureDim(const int inputDim, const int filterDim, 
		   const int stride, const int paddDim)
    {
	return (int)std::floor(inputDim - filterDim + 2*paddDim)+1;
    }
    
    // calculate the dimension after pooling
    int featureDimAfterPool(const int featureDim, const int pool, const int border)
    {
	if (border > 0)
	    return std::ceil(featureDim/pool);
	else
	    return std::floor(featureDim/pool);
    }
    
    // parse the option line
    std::vector<std::vector<int> > parseOption(const std::string options, 
					      const std::string layerName)
    {
	std::vector<std::string> featureOptions;
	boost::split(featureOptions, options, boost::is_any_of(CNNOPTIONSPLITER1));
	if (featureOptions.size() ==0){
	    featureOptions.push_back(options);
	}
	std::vector<std::vector<int> > parsedOption;
	std::vector<int> tmpOptions;
	std::vector<std::string> featureOpt;
	for (int i =0; i<featureOptions.size(); i++){
	    
	    boost::split(featureOpt, featureOptions[i], boost::is_any_of(CNNOPTIONSPLITER2));
	    // fixed format: filterW, filterH, strideW, strideH, poolW, poolH, padW, padH
	    if (featureOpt.size() != 10){
		printf("filterW_filterH_strideW_strideH_padW_padH_poolW_poolH_borW_borH\n");
		throw std::runtime_error(std::string("In valid configuration CNN")+layerName);
	    }
	    for (int j=0; j<featureOpt.size(); j++){
		try {
		    tmpOptions.push_back(boost::lexical_cast<int>(featureOpt[j]));
		} catch( boost::bad_lexical_cast const& ) {
		    printf("Invalid format: filterW_filterH_strideW_strideH_poolW_poolH_padW_padH");
		    throw std::runtime_error(std::string("In valid configuration CNN: not number")
					     +layerName);
	    	}
	    }
	    parsedOption.push_back(tmpOptions);
	    tmpOptions.clear();
	}
	return parsedOption;
    }
	
    int getOutFeatDim(const helpers::JsonValue &layerChild,
		     const helpers::JsonValue &weightsSection,
		     int preLayerSize)
    {
	std::string layerName = layerChild->HasMember("name") ? (*layerChild)["name"].GetString()  : "";
	std::string option    = layerChild->HasMember("size") ? (*layerChild)["size"].GetString()  : "";
	if (option.size() ==0){
	    throw std::runtime_error(std::string("Error configuration CNN: void size option"));
	}
	std::vector<std::vector<int> > options = parseOption(option, layerName);
	int featureOutDim = 0;
	for (int i = 0; i<options.size(); i++){
	    std::vector<int> tmp = options[i];
	    featureOutDim += featureDimAfterPool(featureDim(preLayerSize, tmp[0], tmp[2], tmp[4]),
					     tmp[6], tmp[8]);
	}
	return featureOutDim;
    }
    
    int getWeightDim(const helpers::JsonValue &layerChild,
		     const helpers::JsonValue &weightsSection,
		     int preLayerSize)
    {
	std::string layerName = layerChild->HasMember("name") ? (*layerChild)["name"].GetString()  : "";
	std::string option    = layerChild->HasMember("size") ? (*layerChild)["size"].GetString()  : "";
	if (option.size() ==0){
	    throw std::runtime_error(std::string("Error configuration CNN: void size option"));
	}
	std::vector<std::vector<int> > options = parseOption(option, layerName);
	int weightDim = 0;
	for (int i = 0; i<options.size(); i++){
	    // the filter size + one bias
	    weightDim += (options[i][0]*options[i][1] + 1);
	}
	return weightDim;
    }
}

namespace layers {
    

    
    /*****************************************************************************************
     * CNN feature unit
     *****************************************************************************************/
    template <typename TDevice>
    CNNUnit<TDevice>::CNNUnit(const int fWidth,  const int fHeight,
			      const int strideW, const int strideH,
			      const int poolW,   const int poolH,
			      const int paddW,   const int paddH,
			      const int weiDim,  const int outDim,
			      Layer<TDevice> &precedingLayer)
	: m_filterW   (fWidth)
	, m_filterH   (fHeight)
	, m_strideW   (strideW)
	, m_strideH   (strideH)
	, m_poolW     (poolW)
	, m_poolH     (poolH)
	, m_paddW     (paddW)
	, m_paddH     (paddH)
	, m_weightDim (weiDim)
	, m_outputDim (outDim)
	, m_precedingLayer (precedingLayer)
    {
	// initializing
	
	// the size of m_featureOutput must be determined for each fraction of data
	//  in loadSequence stage
	// const int inputW,  const int inputH, const int inputC,
	m_inputW = m_inputH = m_inputC = -1;  
	
	
    }

    template <typename TDevice>
    CNNUnit<TDevice>::~CNNUnit()
    {
    }
    
    template <typename TDevice>
    void CNNUnit<TDevice>::computeForwardPass()
    {
    }
    
    template <typename TDevice>
    void CNNUnit<TDevice>::computeBackwardPass()
    {
    }


    /*****************************************************************************************
     * CNN layer 
     *****************************************************************************************/
    template <typename TDevice>
    CNNLayer<TDevice>::CNNLayer(
        const helpers::JsonValue &layerChild, 
        const helpers::JsonValue &weightsSection,
        Layer<TDevice> &precedingLayer)
	: m_weightDim              (CNNTools::getWeightDim(layerChild, weightsSection, precedingLayer.size()))
	, m_outputDim              (CNNTools::getOutFeatDim(layerChild, weightsSection, precedingLayer.size()))
	, TrainableLayer<TDevice>  (layerChild, weightsSection, 1, 0, 
				    m_weightDim, m_outputDim,  precedingLayer)
    {
	// handling the specification

	// revise the m_outputs, m_outputErrors, m_outputErrorsCopy
	
	//
    }
    
    template <typename TDevice>
    CNNLayer<TDevice>::~CNNLayer()
    {
    }
    
    template <typename TDevice>
    void CNNLayer<TDevice>::computeForwardPass()
    {
    }
    
    template <typename TDevice>
    void CNNLayer<TDevice>::computeBackwardPass()
    {
    }

    template <typename TDevice>
    void CNNLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
	// load the sequences for TrainableLayers
	TrainableLayer<TDevice>::loadSequences(fraction);
	
	// 
    }

    template <typename TDevice>
    void CNNLayer<TDevice>::mergeOutput()
    {
    }
    
    template <typename TDevice>
    const std::string& CNNLayer<TDevice>::type() const
    {
	static const std::string m("cnn");
	return m;
    }
    
    template class CNNUnit<Cpu>;
    template class CNNUnit<Gpu>;
    template class CNNLayer<Gpu>;
    template class CNNLayer<Cpu>;
}
