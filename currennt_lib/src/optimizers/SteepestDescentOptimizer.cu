/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
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

#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "SteepestDescentOptimizer.hpp"
#include "../layers/TrainableLayer.hpp"
#include "../layers/InputLayer.hpp"
#include "../layers/Layer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../rapidjson/document.h"

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {

    struct UpdateWeightFn_withMask
    {
        real_t learningRate;
        real_t momentum;

        const real_t *weights;
        const real_t *weightUpdates;
	/* Add 0413 weight Mask*/
	const real_t *weightMask;
        real_t       *weightDeltas;

        __host__ __device__ real_t operator() (const int &weightIdx)
        {
            // calculate and store the weight delta
            real_t delta = momentum * weightDeltas[weightIdx] - learningRate * weightUpdates[weightIdx];
            weightDeltas[weightIdx] = delta;

            // calculate the new weight
	    // Modify 0413 weight Mask
            // real_t newWeight = (weights[weightIdx] + delta;
	    real_t newWeight = ((weights[weightIdx] + delta)*weightMask[weightIdx]);
            return newWeight;
        }
    };

    struct UpdateWeightFn
    {
        real_t learningRate;
        real_t momentum;

        const real_t *weights;
        const real_t *weightUpdates;
        real_t       *weightDeltas;

        __host__ __device__ real_t operator() (const int &weightIdx)
        {
            // calculate and store the weight delta
            real_t delta = momentum * weightDeltas[weightIdx] - learningRate * weightUpdates[weightIdx];
            weightDeltas[weightIdx] = delta;

            // calculate the new weight
	    // Modify 0413 weight Mask
            real_t newWeight = weights[weightIdx] + delta;
	    return newWeight;
        }
    };
    
    /* Add 16-02-22 Wang: for WE updating */
    // functor to update the parameter
    struct UpdateWeWeightFn
    {
        real_t learningRate;
        const real_t *weights;
        const real_t *weightUpdates;
        
        __host__ real_t operator() (const int &weightIdx)
        {
            // calculate and store the weight delta
            real_t delta =  -1 * learningRate * weightUpdates[weightIdx];
            // calculate the new weight
            real_t newWeight = weights[weightIdx] + delta;

            return newWeight;
        }
    };

} // anonymous namespace
} // namespace internal


namespace optimizers {
    
    /* Add 16-02-22 Wang: for WE updating */
    // add the SGD optimizer for we
    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::_updateWeInput()
    {
	if (m_weLearningRate < 0 ){
	    
	}else{
	    // The updating requires m_weBank, m_outerrors and m_weIdx, m_weDim, m_weIdx
	    // Because m_weBank is CPU::real_vector, we avoid using thrust and GPU computating
	    // currently, no momentum of we updating
	
	    // get the input layer
	    layers::InputLayer<TDevice> *layer = dynamic_cast<layers::InputLayer<TDevice>*>(this->_neuralNetwork().layers().front().get());
	
	    // because dummy error is zero, no need to know where dummy starts, just udpate using all the data
	    unsigned int inputSize  = layer->size();
	    unsigned int weIDDim    = layer->_weIDDim();
	    unsigned int weDim      = layer->_weDim();
	    // Not using assignment here
	    // Cpu::real_vector weBank = layer->_weBank();
	    Cpu::real_vector weIdx  = layer->_weIdx();
	    Cpu::real_vector err    = layer->outputErrorsCpu();

	    internal::UpdateWeWeightFn fn;
	    fn.learningRate = m_weLearningRate;
	
	    // updating now
	    for (int i=0;i<weIdx.size();i++){
	    
		if (weIdx[i]<0){
		    // note: when parallel sequences was utilized, the data buffer size is like
		    // m_parallel * m_maxLength >= m_parallel * m_curMaxLength >= sum_m_parallel(timesteps)
		    // dummy only work for the empty slots between m_parallel*m_curMaxLength and sum_m_parallel(timesteps)
		    // thus, we need to use weIdx to skip the empty slotes between m_parallel*m_maxLength and m_parallel*m_curMaxLength
		    // 
		    continue;
		}
		// locate the vector in weBank
		fn.weights       = helpers::getRawPointer(layer->_weBank())+(int)weIdx[i]*weDim;
		// locate the update vector in err (the err includes the dimension of normal input)
		fn.weightUpdates = helpers::getRawPointer(err)+i*inputSize+weIDDim;
		thrust::transform(
				  thrust::counting_iterator<int>(0),
				  thrust::counting_iterator<int>(weDim),
				  layer->_weBank().begin()+weIdx[i]*weDim,
				  fn
				  );
	    }
	    // debug
	    if(0){
		std::cout << "For debugging" << std::endl; 
	    }
	}
    }

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::_updateWeights()
    {
        /* Add 16-02-22 Wang: for WE updating */
	if (m_learningRate < 0){
	    // skip updateing the weights if learning rate is negative
	    
	}else{
	    
	    
	    // Add 0409 learning weight decay
	    if (this->_checkLRdecay()){
		m_learningRateDecay = m_learningRateDecay*m_learningRateDecayRate;
		this->_setLRdecayFalse(); // reset the flag_decay
	    }
	    
	    
	    internal::UpdateWeightFn_withMask updateWeightFn;
	    internal::UpdateWeightFn updateWeightFn2;
	    for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
        	layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
		if (!layer)
		    continue;
		
		// if mask is utilized
		if (layer->flagUseWeightMask()){
		    updateWeightFn.momentum     = m_momentum;
		    updateWeightFn.learningRate = m_learningRate*m_learningRateDecay;
		    if (layer->learningRate() >= 0.0)
			updateWeightFn.learningRate = layer->learningRate();

		    updateWeightFn.weights       = helpers::getRawPointer(layer->weights());
		    updateWeightFn.weightUpdates = helpers::getRawPointer(this->_curWeightUpdates()[i]);
		    updateWeightFn.weightDeltas  = helpers::getRawPointer(m_weightDeltas[i]);
		
		    // Add 0413 for Weight mask
		    updateWeightFn.weightMask    = helpers::getRawPointer(layer->weightMask());
		
		    thrust::transform(
				      thrust::counting_iterator<int>(0),
				      thrust::counting_iterator<int>((int)layer->weights().size()),
				      layer->weights().begin(),
				      updateWeightFn
				      );
		// if mask is not used
		}else{
		    updateWeightFn2.momentum     = m_momentum;
		    updateWeightFn2.learningRate = m_learningRate*m_learningRateDecay;
		    if (layer->learningRate() >= 0.0)
			updateWeightFn2.learningRate = layer->learningRate();
		    

		    updateWeightFn2.weights       = helpers::getRawPointer(layer->weights());
		    updateWeightFn2.weightUpdates = helpers::getRawPointer(this->_curWeightUpdates()[i]);
		    updateWeightFn2.weightDeltas  = helpers::getRawPointer(m_weightDeltas[i]);
				
		    thrust::transform(
				      thrust::counting_iterator<int>(0),
				      thrust::counting_iterator<int>((int)layer->weights().size()),
				      layer->weights().begin(),
				      updateWeightFn2
				      );
		}
	    }
	}
    }

    template <typename TDevice>
    SteepestDescentOptimizer<TDevice>::SteepestDescentOptimizer(
        NeuralNetwork<TDevice> &neuralNetwork, data_sets::DataSet &trainingSet, data_sets::DataSet &validationSet,
        data_sets::DataSet &testSet, int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery, 
        real_t learningRate, real_t momentum, real_t weLearningRate, real_t learningRateDecayRate, int decayEpochNM)
        : Optimizer<TDevice>(neuralNetwork, trainingSet, validationSet, testSet, maxEpochs, maxEpochsNoBest, validateEvery, testEvery, decayEpochNM)
        , m_learningRate    (learningRate)
        , m_learningRateFirst(learningRate)
	, m_weLearningRate(weLearningRate)
	, m_learningRateDecayRate(learningRateDecayRate)
        , m_momentum        (momentum)
    {
        // intialize the weight deltas vectors with zeros
        m_weightDeltas = this->_curWeightUpdates();
        for (size_t i = 0; i < m_weightDeltas.size(); ++i)
            thrust::fill(m_weightDeltas[i].begin(), m_weightDeltas[i].end(), 0);
	
	// Add 0409 for decaying the learning rate
	m_learningRateDecay = 1.0;
    }

    template <typename TDevice>
    SteepestDescentOptimizer<TDevice>::~SteepestDescentOptimizer()
    {
    }

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        Optimizer<TDevice>::exportState(jsonDoc);

        Optimizer<TDevice>::_exportWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", m_weightDeltas);
    }

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::importState(const helpers::JsonDocument &jsonDoc)
    {
        Optimizer<TDevice>::importState(jsonDoc);

        Optimizer<TDevice>::_importWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", &m_weightDeltas);
    }

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::importParameter(const helpers::JsonDocument &jsonDoc)
    {
        Optimizer<TDevice>::importParameter(jsonDoc);
	// currently no need for momentum
        // Optimizer<TDevice>::_importWeights(jsonDoc, "steepest_descent_optimizer_weight_deltas", &m_weightDeltas);
    }

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::adjustLR(int decayTime)
    {
	for(int i =0; i < decayTime; i++)
	    m_learningRateDecay = m_learningRateDecay*0.1;
	printf("Adjust the learning rate to %e", m_learningRate*m_learningRateDecay);
    }
    
    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::reinit()
    {
	// intialize the weight deltas vectors with zeros
        m_weightDeltas = this->_curWeightUpdates();
        for (size_t i = 0; i < m_weightDeltas.size(); ++i)
            thrust::fill(m_weightDeltas[i].begin(), m_weightDeltas[i].end(), 0);
	this->_reinit();

    }
    

    template <typename TDevice>
    void SteepestDescentOptimizer<TDevice>::setLearningRateFirst(real_t learningRateFirst)
    {
        m_learningRateFirst = learningRateFirst;
    }


    // explicit template instantiations
    template class SteepestDescentOptimizer<Cpu>;
    template class SteepestDescentOptimizer<Gpu>;

} // namespace optimizers
