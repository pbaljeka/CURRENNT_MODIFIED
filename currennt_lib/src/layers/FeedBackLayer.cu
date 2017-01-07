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

#include "FeedBackLayer.hpp"

#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/fill.h>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <stdexcept>

#define FEEDBACKLAYER_DEBUG 0

namespace internal{
namespace {

    // dustbin.txt/Block1226x02
    
    struct vectorFillForward
    {
	// Copy the output of preceding layer to the output of this layer
	// Copy the output of the target layer to  the output of this layer

	int dimInput1;      // dimension of output of preceding layer
	int dimInput2;      // dimension of output of target layer (in total)
	int dimInput2Start; // from which dimension of the target to load (may not be 0)
	int dimOutput;      // dimension of output of this layer
	int parallel;       // number of parallel sentences
	
	real_t *input1;     // preceding layer
	real_t *input2;     // target layer
	real_t *output;     // this layer
	
	// dispatched over Dim * T * Parallel
	__host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t)
	{
	    int outputEffIdx = t.get<1>();
	    int timeStep     = outputEffIdx / dimOutput;
	    int dimIdx       = outputEffIdx % dimOutput;

	    // Idx in the output of this layer
	    int outputIdx    = timeStep * dimOutput + dimIdx;
	    
	    if (dimIdx >= dimInput1){
		// copy from the target layer (feedback part)
		// here, we treat all kinds of feedback units equally
		//  i.e. softmax idx will be copied as one-dimensional data directly
		dimIdx = dimIdx - dimInput1;
		if (timeStep < parallel)      // loopback one step
		    output[outputIdx] = 0.0;
		else
		    output[outputIdx] = input2[(timeStep - parallel) * dimInput2 +
					       dimIdx + dimInput2Start];
	    }else{
		// copy from the preceding layer
		output[outputIdx] = input1[timeStep * dimInput1 + dimIdx];
	    }
	}
    };
    
    struct vectorFillBackward
    {
	int dimInput1;      // dimension of the preceding layer
	int dimOutput;      // dimension of this layer
	
	real_t *outputError;
	
	// dispatched over Dim * T * Parallel
	// Dim here is the dimension of the previous layer
	__host__ __device__ real_t operator() (const int &outputIdx) const
	{
	    int timeStep  = outputIdx / dimInput1;
	    int dimIdx    = outputIdx % dimInput1;
	    return outputError[timeStep * dimOutput + dimIdx];
	}
    };
    
}
}

namespace layers{

    // dustbin.txt/Block 1226x01
    int ParseLayerOpt(const std::string options){
	std::vector<std::string> tempArgs;
	boost::split(tempArgs, options, boost::is_any_of("_"));
	return boost::lexical_cast<int>(tempArgs[0]);
    }

    
    template <typename TDevice>
    FeedBackLayer<TDevice>::FeedBackLayer(
					  const helpers::JsonValue &layerChild,
					  const helpers::JsonValue &weightsSection,
					  Layer<TDevice>           &precedingLayer
					  )
	: TrainableLayer<TDevice>(layerChild, weightsSection, 0, 0, precedingLayer)
	, m_targetDim   (-1)
	, m_targetLayer (NULL)
    {
	m_targetBuffer.clear();
    }

    template <typename TDevice>
    FeedBackLayer<TDevice>::~FeedBackLayer()
    {
    }

    template <typename TDevice>
    void FeedBackLayer<TDevice>::linkTargetLayer(Layer<TDevice> &targetLayer)
    {
	m_targetDim      = ParseLayerOpt(targetLayer.layerAddInfor(1));
	m_targetLayer    = &targetLayer;

	// Now, use all target features for feedback
	// To be completed
	m_targetDimStart = 0;
	m_targetDimEnd   = m_targetDim;
	
	int dimExpected = (m_targetDimEnd - m_targetDimStart + this->precedingLayer().size());
	
	if (dimExpected !=this->size()){
	    printf("Feedback dim + Feedforward dim = %d\n", dimExpected);
	    throw std::runtime_error("Error in network.jsn feedback layer size");
	}
	if (m_targetDimEnd > m_targetDim || m_targetDimStart > m_targetDim ||
	    m_targetDimEnd < m_targetDimStart){
	    throw std::runtime_error("Error in configuration of targetDimStart, targetDimEnd");
	}	
	// print information
	printf("\nCreating the feedback link:\n");
	printf("\tFrom %s [%d-%d]", targetLayer.type().c_str(), m_targetDimStart, m_targetDimEnd);
	printf("\n");
    }

    template <typename TDevice>
    const std::string& FeedBackLayer<TDevice>::type() const
    {
        static std::string s;
        if (s.empty()) s = "feedback";
        return s;
    }


    // computeForward: 
    //  in training stage, target data are known
    template <typename TDevice>
    void FeedBackLayer<TDevice>::computeForwardPass()
    {
	if (m_targetLayer == NULL)
	    throw std::runtime_error("Target layer is not linked");
	
	thrust::fill(this->outputs().begin(), this->outputs().end(), 0.0);
	{{
	    // Concatenate the output of the preceding layer and the feedback layer
	    int previousSize  = this->precedingLayer().size();
	    
	    internal::vectorFillForward fn;
	    fn.dimInput1      = previousSize;   // the dimension from preceding layer
	    fn.dimInput2      = m_targetDim;    // the dimension of the output of target layer
	    fn.dimInput2Start = m_targetDimStart; // from which dimension to load from target layer
	    fn.dimOutput      = this->size();     
	    fn.parallel       = this->parallelSequences();

	    fn.input1         = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.input2         = helpers::getRawPointer(m_targetLayer->secondOutputs());
	    fn.output         = helpers::getRawPointer(this->outputs());
	    
	    int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();
	    thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(this->outputs().begin(),
							     thrust::counting_iterator<int>(0))),
		thrust::make_zip_iterator(thrust::make_tuple(this->outputs().begin()+n,
							     thrust::counting_iterator<int>(0)+n)),
		fn);
	    // dustbin.txt/Block1226x03
	    	    
	}}
	/*
	Cpu::real_vector tmp = this->outputs();
	for (int i = 0; i<this->curMaxSeqLength(); i++){
	    printf("%3d\t", i);
	    for (int j = 0; j < this->size(); j++){
		if (j<2)
		    printf("%0.2f\t", tmp[i*this->size() + j]);
		else
		    if (tmp[i*this->size() + j] * tmp[i*this->size() + j] > 0.00001)
			printf("One-hot: %3d", j);
	    }
	    printf("\n");
	}
	printf("\n");*/
    }

    // computeForwardPass
    // in synthesis stage, when the target must be predicted frame by frame
    template <typename TDevice>
    void FeedBackLayer<TDevice>::computeForwardPass(const int timeStep)
    {
	if (m_targetLayer == NULL){
	    throw std::runtime_error("Target layer is not linked");
	}	
	
	int effTimeStepS = timeStep     * this->parallelSequences();
	int effTimeStepE = (timeStep+1) * this->parallelSequences();
	
	thrust::fill(this->outputs().begin() + effTimeStepS * this->size(), 
		     this->outputs().begin() + effTimeStepE * this->size(), 0.0);
	
	{{
	    // The dimension of the concatenated feature (if no softmax exists)
	    int previousSize  = this->precedingLayer().size();
	    
	    // Concatenate the feature vector 
	    // (by treating the 1 dimensional softmax Index as a normal feature)
	    internal::vectorFillForward fn;
	    
	    fn.dimInput1      = previousSize;
	    fn.dimInput2      = m_targetDim;

	    fn.dimOutput      = this->size();
	    fn.parallel       = this->parallelSequences();
	    fn.dimInput2Start = m_targetDimStart;


	    fn.input1         = helpers::getRawPointer(this->precedingLayer().outputs());
	    fn.input2         = helpers::getRawPointer(m_targetLayer->secondOutputs());
	    fn.output         = helpers::getRawPointer(this->outputs());
	    
	    thrust::for_each(
	       thrust::make_zip_iterator(
		 thrust::make_tuple(
			this->outputs().begin()+ effTimeStepS * this->size(),
			thrust::counting_iterator<int>(0)+ effTimeStepS * this->size())),
	       thrust::make_zip_iterator(
		 thrust::make_tuple(
			this->outputs().begin()+ effTimeStepE * this->size(),
			thrust::counting_iterator<int>(0)+ effTimeStepE * this->size())),
	       fn);
	    // dustbin.txt/Block1226x04
	    
	}}
    }

    // 
    template <typename TDevice>
    void FeedBackLayer<TDevice>::computeBackwardPass()
    {
	{{
	   // Copy the gradient for the preceding layer
	   internal::vectorFillBackward fn;
	   fn.dimInput1      = this->precedingLayer().size();
	   fn.dimOutput      = this->size();
	   fn.outputError    = helpers::getRawPointer(this->outputErrors());

	   int n = (this->curMaxSeqLength() * this->parallelSequences() *
		    this->precedingLayer().size());
	   
	   thrust::transform(thrust::counting_iterator<int>(0),
			     thrust::counting_iterator<int>(0)+n,
			     this->precedingLayer().outputErrors().begin(),
			     fn);	   
	}}
    }
    
    template class FeedBackLayer<Cpu>;
    template class FeedBackLayer<Gpu>;
    
}
