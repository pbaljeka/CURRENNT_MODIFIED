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

#include "Optimizer.hpp"
#include "../layers/TrainableLayer.hpp"
#include "../layers/PostOutputLayer.hpp"
#include "../layers/BinaryClassificationLayer.hpp"
#include "../layers/MulticlassClassificationLayer.hpp"
#include "../Configuration.hpp"
#include "../helpers/JsonClasses.hpp"

#include <limits>

#include <thrust/transform.h>
#include <thrust/fill.h>

namespace optimizers {

    template <typename TDevice>
    real_t Optimizer<TDevice>::_processDataSet(data_sets::DataSet &ds, 
					       bool calcWeightUpdates, 
					       real_t *classError)
    {
        // process all data set fractions
        real_t error = 0;
        *classError = (real_t) ds.totalTimesteps();
	
	// Add 0413 Wang : for weight Mask
	m_neuralNetwork.maskWeight();
	
	// Add 0928 Wang : notify the current training epoch number
	m_neuralNetwork.notifyCurrentEpoch(m_curEpoch);

	// variances
        boost::shared_ptr<data_sets::DataSetFraction> frac;
        bool firstFraction = true;                           // first fraction
	int  uttNum        = ds.totalSequences();            // total number of sequences
	int  uttCnt        = 0;                              // utterance counter
	int  frameNum      = -1;                             // number of frames in mini-batch
	
        while ((frac = ds.getNextFraction())) {
	    
	    // get the number of frames for SGD
	    if (Configuration::instance().hybridOnlineBatch()) 
		frameNum   = frac->fracTimeLength();
	    
            // compute forward pass and calculate the error
            m_neuralNetwork.loadSequences(*frac);
            m_neuralNetwork.computeForwardPass();
            error += (m_neuralNetwork.calculateError()/ds.totalSequences());
	    
	    // check for NaN
	    if (error != error){
		printf("NaN detected. Tune the learning rate please.\n");
		this->m_blowed = true;
		break;
	    }
	    
	    // calculate the classification error if any of the layers used
            if (dynamic_cast<layers::BinaryClassificationLayer<TDevice>*>(
			&m_neuralNetwork.postOutputLayer()))
                *classError -= (real_t)static_cast<layers::BinaryClassificationLayer<TDevice>&>(
			m_neuralNetwork.postOutputLayer()).countCorrectClassifications();
            if (dynamic_cast<layers::MulticlassClassificationLayer<TDevice>*>(
			&m_neuralNetwork.postOutputLayer()))
                *classError -= (real_t)static_cast<layers::MulticlassClassificationLayer<TDevice>&>(
			m_neuralNetwork.postOutputLayer()).countCorrectClassifications();
            
	    // backward computation and parameter updateing
            if (calcWeightUpdates) {
		
                // weight noise
                std::vector<Cpu::real_vector> origWeights(m_neuralNetwork.layers().size());
                if (Configuration::instance().weightNoiseSigma() > 0) {
                    for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
                        layers::TrainableLayer<TDevice> *layer = 
			    dynamic_cast<layers::TrainableLayer<TDevice>*>(
					m_neuralNetwork.layers()[i].get());
                        if (layer) {
                            origWeights[i] = layer->weights();
                            layer->injectWeightNoise(Configuration::instance().weightNoiseSigma());
                        }
                    }
                }
		
                // compute the backward pass and accumulate the weight updates
                m_neuralNetwork.computeBackwardPass();

		// accumulate the statistics for parameter updating
                for (size_t i = 1; i < m_neuralNetwork.layers().size(); ++i) {
		    
                    layers::TrainableLayer<TDevice> *layer = 
			dynamic_cast<layers::TrainableLayer<TDevice>*>(
				m_neuralNetwork.layers()[i].get());
		    layers::MDNLayer<TDevice> *mdnlayer = 
			dynamic_cast<layers::MDNLayer<TDevice>*>(
				m_neuralNetwork.layers()[i].get());
		    
		    // Modify 0703: PostLayer can be trainable too
		    if (!layer && !(mdnlayer && mdnlayer->flagTrainable())){
			continue;
		    }
		    
		    if (layer){
			// if batch mode (not stochastic) and not the first fraction, 
			// accumulating the updates
			if (!firstFraction && !Configuration::instance().hybridOnlineBatch())
			    thrust::transform(layer->weightUpdates().begin(), 
					      layer->weightUpdates().end(), 
					      m_curWeightUpdates[i].begin(), 
					      m_curWeightUpdates[i].begin(), 
					      thrust::plus<real_t>());
			else
			    thrust::copy(layer->weightUpdates().begin(), 
					 layer->weightUpdates().end(), 
					 m_curWeightUpdates[i].begin());
						
			// restore old weights before update in case of weight noise
			if (Configuration::instance().weightNoiseSigma() > 0.0)
			    thrust::copy(origWeights[i].begin(), 
					 origWeights[i].end(), 
					 layer->weights().begin());
			
		    }else if(mdnlayer && mdnlayer->flagTrainable()){
			
			// if batch mode (not stochastic) and not the first fraction, 
			// accumulating the updates
			if (!firstFraction && !Configuration::instance().hybridOnlineBatch())
			    thrust::transform(mdnlayer->weightUpdates().begin(), 
					      mdnlayer->weightUpdates().end(), 
					      m_curWeightUpdates[i].begin(), 
					      m_curWeightUpdates[i].begin(), 
					      thrust::plus<real_t>());
			else
			    thrust::copy(mdnlayer->weightUpdates().begin(), 
					 mdnlayer->weightUpdates().end(), 
					 m_curWeightUpdates[i].begin());

			// ???
			// restore old weights before update in case of weight noise
			// if (Configuration::instance().weightNoiseSigma() > 0.0)
			//    thrust::copy(origWeights[i].begin(), origWeights[i].end(), 
			//		 mdnlayer->weights().begin());
		    }
		    
                }

                // update weights for hybrid online/batch learning
                if (Configuration::instance().hybridOnlineBatch())
                    _updateWeights(frameNum);
		
		/* Add 16-02-22 Wang: for WE updating */
		if (Configuration::instance().hybridOnlineBatch() && 
		    m_neuralNetwork.inputLayer().inputWeUpdate()){
		    _updateWeInput(frameNum);
		}
            }

            firstFraction = false;
	    //std::cerr << uttCnt << "/" << uttNum <<std::endl;
	    uttCnt += frac->numSequences();
        }

        // update weights for batch learning
        if (calcWeightUpdates && !Configuration::instance().hybridOnlineBatch())
            _updateWeights(1);

        // normalize the errors
	// strange, why totalSequences? when parallel sequences are calculated, there may be bias
        //error /= ds.totalSequences();
	//error /= ds.totalTimesteps();

        *classError /= (real_t)ds.totalTimesteps();

        return error;
    }

    template <typename TDevice>
    void Optimizer<TDevice>::_exportWeights(const helpers::JsonDocument &jsonDoc, 
					    const char *arrayName, 
					    const std::vector<real_vector> &weights)
    {
        rapidjson::Value weightsArray(rapidjson::kArrayType);
        weightsArray.Reserve((rapidjson::SizeType)weights.size(), jsonDoc->GetAllocator());

        for (size_t i = 0; i < weights.size(); ++i) {
            rapidjson::Value v(rapidjson::kArrayType);
            Cpu::real_vector w = weights[i];
            v.Reserve((rapidjson::SizeType)w.size(), jsonDoc->GetAllocator());
            for (size_t j = 0; j < w.size(); ++j)
                v.PushBack(w[j], jsonDoc->GetAllocator());
            weightsArray.PushBack(v, jsonDoc->GetAllocator());
        }

        jsonDoc->AddMember(arrayName, weightsArray, jsonDoc->GetAllocator());
    }

    template <typename TDevice>
    void Optimizer<TDevice>::_importWeights(const helpers::JsonDocument &jsonDoc, 
					    const char *arrayName, 
					    std::vector<real_vector> *weights)
    {
        if (!jsonDoc->HasMember(arrayName) || !(*jsonDoc)[arrayName].IsArray())
            throw std::runtime_error(std::string("Array '") + 
				     arrayName + "' is missing or has the wrong type");

        if ((*jsonDoc)[arrayName].Size() != (rapidjson::SizeType)weights->size())
            throw std::runtime_error(std::string("Array '") + 
				     arrayName + "' has a wrong size");

        int i = 0;
        for (rapidjson::Value::ConstValueIterator it = (*jsonDoc)[arrayName].Begin(); 
	     it != (*jsonDoc)[arrayName].End(); ++it) {
            if (!it->IsArray())
                throw std::runtime_error(std::string("Object in '") + 
					 arrayName + "' is not an array");
            if (it->Size() != (rapidjson::SizeType)(*weights)[i].size())
                throw std::runtime_error(std::string("Subarray in '") + 
					 arrayName + "' has a wrong size");

            Cpu::real_vector w;
            w.reserve(it->Size());
            for (rapidjson::Value::ConstValueIterator it2 = it->Begin(); it2 != it->End(); ++it2)
                w.push_back((real_t)it2->GetDouble());

            (*weights)[i] = w;

            ++i;
        }
    }

    template <typename TDevice>
    void Optimizer<TDevice>::_storeWeights()
    {
        for (size_t i = 1; i < m_neuralNetwork.layers().size(); ++i) {
            layers::TrainableLayer<TDevice> *layer = 
		dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
            if (layer) 
            	thrust::copy(layer->weights().begin(), 
			     layer->weights().end(), 
			     m_bestWeights[i].begin());
	    else{
		layers::MDNLayer<TDevice> *mdnlayer = 
		    dynamic_cast<layers::MDNLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
		if (mdnlayer && mdnlayer->flagTrainable())
		    thrust::copy(mdnlayer->weights().begin(), 
				 mdnlayer->weights().end(), 
				 m_bestWeights[i].begin());
	    }
        }
    }

    template <typename TDevice>
    void Optimizer<TDevice>::_restoreWeights()
    {
        for (size_t i = 1; i < m_neuralNetwork.layers().size(); ++i) {
	    layers::TrainableLayer<TDevice> *layer = 
		dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
            if (layer)
            	thrust::copy(m_bestWeights[i].begin(), 
			     m_bestWeights[i].end(), 
			     layer->weights().begin());
	    else{
		layers::MDNLayer<TDevice> *mdnlayer = 
		    dynamic_cast<layers::MDNLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
		if (mdnlayer && mdnlayer->flagTrainable())
		    thrust::copy(m_bestWeights[i].begin(), 
				 m_bestWeights[i].end(), 
				 mdnlayer->weights().begin());
	    }
        }
    }

    template <typename TDevice>
    NeuralNetwork<TDevice>& Optimizer<TDevice>::_neuralNetwork()
    {
        return m_neuralNetwork;
    }

    template <typename TDevice>
    std::vector<typename Optimizer<TDevice>::real_vector>& Optimizer<TDevice>::_curWeightUpdates()
    {
        return m_curWeightUpdates;
    }
    
    template <typename TDevice>
    std::vector<typename Optimizer<TDevice>::real_vector>& Optimizer<TDevice>::_weightStats()
    {
        return m_weightStats;
    }

    template <typename TDevice>
    Optimizer<TDevice>::Optimizer(
	NeuralNetwork<TDevice> &neuralNetwork, data_sets::DataSet &trainingSet, 
	data_sets::DataSet &validationSet, data_sets::DataSet &testSet,
	int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery, int decayEpochNM,
	unsigned optOption)
        : m_neuralNetwork             (neuralNetwork)
        , m_trainingSet               (trainingSet)
        , m_validationSet             (validationSet)
        , m_testSet                   (testSet)
        , m_maxEpochs                 (maxEpochs)
        , m_maxEpochsNoBest           (maxEpochsNoBest)
        , m_validateEvery             (validateEvery)
        , m_testEvery                 (testEvery)
        , m_finished                  (false)
        , m_curEpoch                  (0)
        , m_epochsSinceLowestError    (0)
        , m_lowestValidationError     (std::numeric_limits<real_t>::max())
        , m_curTrainingError          (std::numeric_limits<real_t>::max())
        , m_curValidationError        (std::numeric_limits<real_t>::max())
        , m_curTestError              (std::numeric_limits<real_t>::max())
        , m_curValidationClassError   (0)
        , m_curTrainingClassError     (0)
        , m_curTestClassError         (0)
	, m_decayEpochNM              (decayEpochNM)
	, m_flag_decay                (false)
	, m_waitAfterDecay            (0)
	, m_blowed                    (false)
	, m_optOption                 (optOption)
    {
        // initialize the best weights vectors
        m_bestWeights.resize(m_neuralNetwork.layers().size());
        for (size_t i = 1; i < m_neuralNetwork.layers().size(); ++i) {
            layers::TrainableLayer<TDevice> *layer = 
		dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
	    layers::MDNLayer<TDevice> *mdnlayer = 
		dynamic_cast<layers::MDNLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
            if (layer)
                m_bestWeights[i] = layer->weights();
	    else if (mdnlayer && mdnlayer->flagTrainable())
		m_bestWeights[i] = mdnlayer->weights();
        }

        // initialize the current weight updates vectors
        m_curWeightUpdates = m_bestWeights;
	
	// statistics buffer for learning
	if (m_optOption>0){
	    if (m_optOption == OPTIMIZATION_ADAGRAD || 
		m_optOption == OPTIMIZATION_STOCHASTIC_ADAGRAD){
		
		// initialize the buffer for AdaGrad
		m_weightStats = m_bestWeights;
		for (size_t i = 1; i < m_neuralNetwork.layers().size(); ++i) {
		    thrust::fill(m_weightStats[i].begin(), m_weightStats[i].end(), ADAGRADFACTOR);
		}
		if (m_optOption == OPTIMIZATION_ADAGRAD){
		    printf("\n Optimization Techinique: AdaGrad\n");
		}else{
		    printf("\n Optimization Techinique: SGD + AdaGrad (with LR %0.3f)\n", 
			   Configuration::instance().optimizerSecondLR());
		}
		
	    }else if(m_optOption == OPTIMIZATION_AVEGRAD){
		m_weightStats.clear();
		printf("\n Optimization: average gradient over each data fraction\n");
	    }else{
		m_weightStats.clear();
	    }
	}else{
	    printf("\nOptimization: plain SGD \n");
	}
	
	
	// Add 0409 to check the decayEpochNM
	if (m_decayEpochNM >= m_maxEpochsNoBest){
	    printf("WARNING: decayEpochNM should be less than m_maxEpochsNoBest.");
	}
    }

    template <typename TDevice>
    Optimizer<TDevice>::~Optimizer()
    {
    }

    template <typename TDevice>
    bool Optimizer<TDevice>::finished() const
    {
        return m_finished;
    }

    template <typename TDevice>
    int Optimizer<TDevice>::currentEpoch() const
    {
        return m_curEpoch;
    }

    template <typename TDevice>
    real_t Optimizer<TDevice>::lowestValidationError() const
    {
        return m_lowestValidationError;
    }

    template <typename TDevice>
    int Optimizer<TDevice>::epochsSinceLowestValidationError() const
    {
        return m_epochsSinceLowestError;
    }

    template <typename TDevice>
    real_t Optimizer<TDevice>::curTrainingError() const
    {
        return m_curTrainingError;
    }

    template <typename TDevice>
    real_t Optimizer<TDevice>::curValidationError() const
    {
        return m_curValidationError;
    }

    template <typename TDevice>
    real_t Optimizer<TDevice>::curTestError() const
    {
        return m_curTestError;
    }
    
    template <typename TDevice>                                        
    real_t Optimizer<TDevice>::curTrainingErrorPerFrame() const                
    {                                                                  
	return m_curTrainingErrorPerFrame;                                     
    }                                                                  
                                                                   
    template <typename TDevice>                                        
    real_t Optimizer<TDevice>::curValidationErrorPerFrame() const              
    {                                                                  
	return m_curValidationErrorPerFrame;                                   
    }                                                                  
                                                                   
    template <typename TDevice>                                        
    real_t Optimizer<TDevice>::curTestErrorPerFrame() const                    
    {                                                                  
	return m_curTestErrorPerFrame;                                         
    }                                                                  

    template <typename TDevice>
    real_t Optimizer<TDevice>::curTrainingClassError() const
    {
        return m_curTrainingClassError;
    }

    template <typename TDevice>
    real_t Optimizer<TDevice>::curValidationClassError() const
    {
        return m_curValidationClassError;
    }

    template <typename TDevice>
    real_t Optimizer<TDevice>::curTestClassError() const
    {
        return m_curTestClassError;
    }

    template <typename TDevice>
    bool Optimizer<TDevice>::train()
    {
        if (!m_finished) {
            ++m_curEpoch;

            // train one epoch and update the weights
            m_curTrainingError = _processDataSet(m_trainingSet, true, &m_curTrainingClassError);
	    
	    // Add 0511
	    if (this->m_blowed) {return m_finished;}

	    
	    // Training error
	    m_curTrainingErrorPerFrame = (m_curTrainingError * 
					  m_trainingSet.totalSequences() /
					  m_trainingSet.totalTimesteps());
	    
	    
            // calculate the validation error and store the weights if we a new lowest error
            if (!m_validationSet.empty() && m_curEpoch % m_validateEvery == 0) {
		
                m_curValidationError = _processDataSet(m_validationSet, false, 
						       &m_curValidationClassError);
                m_curValidationErrorPerFrame = (m_curValidationError * 
						m_validationSet.totalSequences() / 
						m_validationSet.totalTimesteps());

                if (m_curValidationError < m_lowestValidationError) {
                    m_lowestValidationError  = m_curValidationError;
                    m_epochsSinceLowestError = 0;
                    _storeWeights();
                }
		
                else {
                    m_epochsSinceLowestError += m_validateEvery;
                }
		
            }
            else if (m_validationSet.empty()) {
                m_epochsSinceLowestError = 0;
                _storeWeights();
            }

            // calculate the test error
            if (!m_testSet.empty() && m_curEpoch % m_testEvery == 0){
                m_curTestError = _processDataSet(m_testSet, false, &m_curTestClassError);
		m_curTestErrorPerFrame = (m_curTestError * 
					  m_testSet.totalSequences() /
					  m_testSet.totalTimesteps());
	    }
	    
	    
            // Add 0409 for decaying the learning rate
	    /*if (m_decayEpochNM > 0 && 
		m_epochsSinceLowestError >= m_decayEpochNM && 
		m_waitAfterDecay < 1){
		m_flag_decay = true;
		//m_epochsSinceLowestError = 0;
		m_waitAfterDecay = m_decayEpochNM;
	    }
	    if (m_waitAfterDecay > 0){
		m_waitAfterDecay--;
	    }
	    */
	    
	    // Check status
	    if (m_maxEpochs >= 0 && m_curEpoch >= m_maxEpochs){
		// it must be finished
		_restoreWeights();
		m_finished  = true;
		m_optStatus = "Finished";
	    }else if (m_epochsSinceLowestError >= m_maxEpochsNoBest){
		if (m_optOption == OPTIMIZATION_STOCHASTIC_ADAGRAD){
		    // no best after N epochs, swtich to AdaGrad
		    m_optOption = OPTIMIZATION_ADAGRAD;
		    // let's start from 1, to avoid save the network
		    m_epochsSinceLowestError = 1;
		    m_lowestValidationError  = std::numeric_limits<real_t>::max();
		    m_optStatus = "To ADAGRAD";
		    this->changeLR(Configuration::instance().optimizerSecondLR());
		    _restoreWeights();
		}else{
		    // no best after N epochs
		    _restoreWeights();
		    m_finished  = true;
		    m_optStatus = "Finished";
		}
	    }else{
		if (m_optOption == OPTIMIZATION_ADAGRAD){
		    m_optStatus = "ADAGRAD";
		}else if (m_optOption == OPTIMIZATION_AVEGRAD){
		    m_optStatus = "AVEGRAD";
		}else{
		    m_optStatus = "SGD";
		}
	    }
        }
	
        return m_finished;
    }

    template <typename TDevice>
    void Optimizer<TDevice>::exportState(const helpers::JsonDocument &jsonDoc) const
    {
        jsonDoc->AddMember("optimizer_finished",                   
			   m_finished,                jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_epoch",                  
			   m_curEpoch,                jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_epochs_since_lowest_error",  
			   m_epochsSinceLowestError,  jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_lowest_validation_error",    
			   m_lowestValidationError,   jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_training_error",         
			   m_curTrainingError,        jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_validation_error",       
			   m_curValidationError,      jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_test_error",             
			   m_curTestError,            jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_training_class_error",   
			   m_curTrainingClassError,   jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_validation_class_error", 
			   m_curValidationClassError, jsonDoc->GetAllocator());
        jsonDoc->AddMember("optimizer_cur_test_class_error",       
			   m_curTestClassError,       jsonDoc->GetAllocator());
	
	// Add 10-02: Add support to the status of the optimizer
	jsonDoc->AddMember("optimizer_status",       
			   m_optOption,               jsonDoc->GetAllocator());

        _exportWeights(jsonDoc, "optimizer_best_weights", m_bestWeights);
	
	if (m_optOption == OPTIMIZATION_ADAGRAD || 
	    m_optOption == OPTIMIZATION_STOCHASTIC_ADAGRAD){
	    _exportWeights(jsonDoc, "optimizer_status_vector",   m_weightStats);
	}
    }

    template <typename TDevice>
    void Optimizer<TDevice>::importState(const helpers::JsonDocument &jsonDoc)
    {
        m_finished                = 
	    helpers::checkedJsonGet<bool  >(*jsonDoc, "optimizer_finished");
        m_curEpoch                = 
	    helpers::checkedJsonGet<int   >(*jsonDoc, "optimizer_cur_epoch");
        m_epochsSinceLowestError  = 
	    helpers::checkedJsonGet<int   >(*jsonDoc, "optimizer_epochs_since_lowest_error");
        m_lowestValidationError   = 
	    helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_lowest_validation_error");
        m_curTrainingError        = 
	    helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_training_error");
        m_curValidationError      = 
	    helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_validation_error");
        m_curTestError            = 
	    helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_test_error");
        m_curTrainingClassError   = 
	    helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_training_class_error");
        m_curValidationClassError = 
	    helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_validation_class_error");
        m_curTestClassError       = 
	    helpers::checkedJsonGet<real_t>(*jsonDoc, "optimizer_cur_test_class_error");
	
	// Add 10-02: status of the optimizer
	m_optOption               =
	    helpers::checkedJsonGet<int   >(*jsonDoc, "optimizer_status");
	
        _importWeights(jsonDoc, "optimizer_best_weights", &m_bestWeights);
	
	if (m_optOption == OPTIMIZATION_ADAGRAD || 
	    m_optOption == OPTIMIZATION_STOCHASTIC_ADAGRAD){
	    _importWeights(jsonDoc, "optimizer_status_vector",   &m_weightStats);
	}
    }
    
    template <typename TDevice>
    void Optimizer<TDevice>::importParameter(const helpers::JsonDocument &jsonDoc)
    {
        _importWeights(jsonDoc, "optimizer_best_weights", &m_bestWeights);
    }

    
    /* Add 04-09 for learning rate decay */
    template <typename TDevice>
    bool Optimizer<TDevice>::_checkLRdecay()
    {
	return m_flag_decay;
    }

    template <typename TDevice>
    void Optimizer<TDevice>::_setLRdecayFalse()
    {
	m_flag_decay = false;
    }
    template <typename TDevice>
    bool Optimizer<TDevice>::checkLRdecay()
    {
	return m_flag_decay;
    }

    template <typename TDevice>
    bool Optimizer<TDevice>::blowed()
    {
	return m_blowed;
    }
    
    template <typename TDevice>
    const unsigned& Optimizer<TDevice>::_optOption() const
    {
	return m_optOption;
    }
    
    template <typename TDevice>
    const std::string& Optimizer<TDevice>::optStatus() const
    {
	return m_optStatus;
    }
    
    // reiniti optimizer
    template <typename TDevice>
    void Optimizer<TDevice>::_reinit()
    {
	m_finished = (false);
        m_curEpoch = (0);
        m_epochsSinceLowestError    =(0);
        m_lowestValidationError     =(std::numeric_limits<real_t>::max());
        m_curTrainingError          =(std::numeric_limits<real_t>::max());
        m_curValidationError        =(std::numeric_limits<real_t>::max());
        m_curTestError              =(std::numeric_limits<real_t>::max());
        m_curValidationClassError   =(0);
        m_curTrainingClassError     =(0);
        m_curTestClassError         =(0);
	m_flag_decay                =(false);
	m_waitAfterDecay            =(0);
	m_blowed                    =(false);

        m_bestWeights.resize(m_neuralNetwork.layers().size());
        for (size_t i = 1; i < m_neuralNetwork.layers().size()-1; ++i) {
	    layers::TrainableLayer<TDevice> *layer = 
		dynamic_cast<layers::TrainableLayer<TDevice>*>(m_neuralNetwork.layers()[i].get());
            if (layer)
                m_bestWeights[i] = layer->weights();
	    // ??? for MDNLayer
        }

		// statistics buffer for learning
	if (m_optOption>0){
	    if (m_optOption == OPTIMIZATION_ADAGRAD || 
		m_optOption == OPTIMIZATION_STOCHASTIC_ADAGRAD){
		// initialize the buffer for AdaGrad
		m_weightStats = m_bestWeights;
		for (size_t i = 1; i < m_neuralNetwork.layers().size(); ++i) {
		    thrust::fill(m_weightStats[i].begin(), m_weightStats[i].end(), ADAGRADFACTOR);
		}		
	    }else if(m_optOption == OPTIMIZATION_AVEGRAD){
		m_weightStats.clear();
	    }else{
		m_weightStats.clear();
	    }
	}
	
        // initialize the current weight updates vectors
        m_curWeightUpdates = m_bestWeights;
    }
    
    // explicit template instantiations
    template class Optimizer<Cpu>;
    template class Optimizer<Gpu>;

} // namespace optimizers
