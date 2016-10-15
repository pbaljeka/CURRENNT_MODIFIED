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

#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "RnnLayer.hpp"
#include "../helpers/limitedError.cuh"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Tanh.cuh"
#include "../Configuration.hpp"

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <cmath>

#define DEBUG_CLOCKRNN 0

namespace internal{
namespace {

    typedef activation_functions::Tanh     cell_act_fn_t;
    
    // Functions for computation forward 
    struct ComputeBlockOutputFn
    {
        int    effLayerSize;
        int    prevOutputDistance;
        real_t bias;
	
	const bool   *skipCRNN;     // whether this step should be skipped
        const char   *patTypes;
        const real_t *biasWeights;
        real_t *unitActs;           // W_2x_t
	real_t *unitActsBuf;        // W_1h_(t-1)


        __host__ __device__ real_t operator() (const int &outputIdx, 
					       const thrust::tuple<bool, bool> &t) const
        {
            // unpack the tuple
            bool firstCall    = t.get<0>();
            bool checkPatType = t.get<1>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set the all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    return 0;
                }
            }

            // calculate indices
            int blockIdx   = outputIdx % effLayerSize;
            // load the niag activations
            real_t unitAct = unitActs[outputIdx];
	    	    
	    if (skipCRNN  != NULL && skipCRNN[blockIdx] && !firstCall){
		// for ClockRNN skip this step, just use the output of previous step
		unitAct        = unitActs[outputIdx + prevOutputDistance];
	    }else{
		// other cases, Wx + Wh + b
		unitAct       += (bias * biasWeights[blockIdx] + unitActsBuf[outputIdx]);
		// apply the activation functions (default, use tanh)
		unitAct        = cell_act_fn_t::fn(unitAct);
	    }
	    // unitActs actually stores the output of previous steps, not activation
	    unitActs[outputIdx] = unitAct;

            return unitAct;
        }
    };

    struct ResortOutputsFn
    {
        int layerSize;
        int effLayerSize;

        const real_t *fwOutputs;
        const real_t *bwOutputs;

        __host__ __device__ real_t operator() (const int &outputIdx) const
        {
            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                return fwOutputs[offset];
            else
                return bwOutputs[offset - effLayerSize];
        }
    };
    
    struct ResortOutputErrorsFn
    {
        int layerSize;
        int effLayerSize;

        real_t *fwOutputErrors;
        real_t *bwOutputErrors;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
        {
            // unpack the tuple
            real_t outputErr = t.get<0>();
            int    outputIdx = t.get<1>();

            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                fwOutputErrors[offset] = outputErr;
            else
                bwOutputErrors[offset - effLayerSize] = outputErr;
        }
    };
    
    struct ComputeBlockErrorsFn
    {
        int effLayerSize;
        int prevOutputDistance;

        const bool   *skipCRNN;     // whether this step should be skipped
	const char   *patTypes;
        const real_t *unitActs;
        real_t       *unitDeltas;

        __host__ __device__ void operator() (
		const thrust::tuple<const real_t&, int, bool, bool, bool> &t) const
        {
            // unpack the tuple
            real_t outputErr    = t.get<0>();
            int    outputIdx    = t.get<1>();
            bool   firstCall    = t.get<2>();
            bool   lastCall     = t.get<3>();
            bool   checkPatType = t.get<4>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    unitDeltas       [outputIdx] = 0;
                    return;
                }
            }
	    
	    // calculate indices
            int blockIdx   = outputIdx % effLayerSize;
            
            // load the output of the activation (=equal to output of this unit)
	    // note: unitActs has been filled with f(unitActs)
	    //       see the second last step of ComputeBlockOutput
	    real_t unitActOutput;
	    if (skipCRNN != NULL && *(skipCRNN+blockIdx))
		unitActOutput = 1.0;
	    else
		unitActOutput = cell_act_fn_t::deriv(unitActs[outputIdx]);

            // calculate the gradient
            real_t unitDeltaData   = outputErr * unitActOutput;

            // store the gradients
            unitDeltas[outputIdx] = helpers::limitedError(unitDeltaData);
        }
    };

    struct CreateH2HClockRnn{
	int     featDim;                  // both row and col
	int     bandNum;
	int    *bandStart;        
	int    *bandEnd;
	real_t *sourceMatrix;
	real_t *targetMatrix;         
	
	__host__ __device__ void operator() (const int idx) const {
	    int rows = idx % featDim;
	    int cols = idx / featDim;
	    bool flag = false;
	    for (int band = 0; band<bandNum; band++){
		if ((cols >= bandStart[band])  && 
		    (cols <=(bandEnd[band]-1)) &&
		    (rows >= bandStart[band])){
		    *(targetMatrix+idx) = *(sourceMatrix+idx);
		    flag = true;
		    break;
		}
	    }
	    if (flag == false){
		*(targetMatrix+idx) = (rows==cols)?(1.0):0.0;
	    }
	}
    };
    
    struct CleanUnitDeltasClockRnn{
	const bool   *skipCRNN;     // whether this step should be skipped
        real_t       *unitDeltas;   // 
	__host__ __device__ void operator() (const int idx) const {
	    if ((*(skipCRNN+idx)))
		*(unitDeltas+idx) = 0.0;
	}
    };
       
    struct SetH2HMatrix{
	
	int    *bandConfig;
	int     bandNum;
	real_t *sourceW;
	real_t *targetW;
	int     featDim;
	int     matrixSize;
	
	__host__ __device__ void operator() (const int idx) const{
	    int bandIdx     = idx / matrixSize + 1;
	    int sourcePos   = idx % matrixSize; 
	    int rows        = sourcePos % featDim;
	    int cols        = sourcePos / featDim;
	    
	    int colStart = 0;
	    int colEnd   = 0;
	    int tmp = 0b01;
	    bool flag = false;
	    
	    for (int band = 0; band<bandNum; band++){
		if (bandIdx  & (tmp << band)){
		    colStart = (band > 0)?(bandConfig[2*band-1]):(colStart);
		    colEnd   = bandConfig[2*band+1];
		    if (cols >= colStart && cols < colEnd && rows >= colStart){
			*(targetW + idx) = *(sourceW + sourcePos);
			flag = true;
			break;
		    }
		}
	    }
	    if (flag == false){
		*(targetW + idx) = (rows==cols) ? (1.0):(0.0);
	    }
	}
    };
}
}


namespace layers {
    
    // Parse the option of ClockRNN
    // input: options, layer size
    // change: m_crStep
    // return: number of possible Hidden2Hidden Matrix
    int ReadClockRNNOptions(const std::string options, Cpu::int_vector &m_crStep, const int size)
    {
	// read in the option
	// 
	std::vector<std::string> tempArgs;
	boost::split(tempArgs, options, boost::is_any_of("_"));
	if ((tempArgs.size() % 2) != 0){
	    printf("ClockRNN option should be TimeReso1_Dim1_TimeReso2_Dim2");
	    throw std::runtime_error("Error in RNNLayer");
	}
	m_crStep.resize(tempArgs.size(),-1);
	for (int i=0; i < tempArgs.size(); i++){
	    m_crStep[i] = boost::lexical_cast<int>(tempArgs[i]);
	}
	if (m_crStep[tempArgs.size()-1]!=size){
	    printf("ClockRNN options has unequal layer size: %d VS %d\n.Please check network.jsn", 
		   m_crStep[tempArgs.size()-1], size);
	    throw std::runtime_error("Error in RNNLayer");
	}
	return std::pow(2, tempArgs.size()/2)-1;
    }

    // Parse m_crStep
    // input:  m_crStep (from the function above) and time step
    // change: tmpSkipFlagCR (which dimension should be skipped in forward propagation)
    // return: which Hidden2Hidden matrix should be used ?
    int DimSkipFlagCR(Cpu::bool_vector &tmpSkipFlagCR, Cpu::int_vector &m_crStep, 
		      int timestep, int parallelSent)
    {
	int timeResolution;
	int featDim = tmpSkipFlagCR.size() / parallelSent;
	for (int idx = 0; idx < tmpSkipFlagCR.size(); idx++){
	    int dim = idx % featDim;
	    for (int block = 0; block < m_crStep.size()/2; block++){
		if (dim < m_crStep[block*2+1]){
		    timeResolution = m_crStep[block*2];
		    if ((timestep % timeResolution)!=0)
			tmpSkipFlagCR[dim] = true;	
		    break;
		}
	    }
	}
	int tmpNumber = 0b1;
	int matrixIdx = 0;
	for (int block = 0; block < m_crStep.size()/2; block++){
	    timeResolution = m_crStep[block*2];
	    if ((timestep % timeResolution)==0)
		matrixIdx += (tmpNumber << block);
	}
	return matrixIdx;
    }
    
    /*int StartEndPos(Cpu::int_vector &crStep, int timestep, 
      Cpu::int_vector &s, Cpu::int_vector &e)
    {
	int band = 0;
	int timeResolution;
	for (int block = 0; block < crStep.size()/2; block++){
	    timeResolution = crStep[block*2];
	    if ((timestep % timeResolution) ==0){
		s[band] = (block==0)?(0):(crStep[2*block-1]);
		e[band] = crStep[2*block+1];
		band++;
	    }
	}
	return band;
    }*/
    
    template <typename TDevice>
    RnnLayer<TDevice>::RnnLayer(const helpers::JsonValue &layerChild, 
				const helpers::JsonValue &weightsSection,
				Layer<TDevice>           &precedingLayer,
				bool                      bidirectional)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 
				  1, 
				  helpers::safeJsonGetInt(layerChild, "size")/(bidirectional?2:1),
				  precedingLayer)
        , m_isBidirectional      (bidirectional)
    {
        if (m_isBidirectional && this->size() % 2 != 0)
            throw std::runtime_error("Cannot create a bidirectional layer with an odd layer size");
	
        int ls  = this->size();
        int pls = this->precedingLayer().size();
	int els = this->size() / (m_isBidirectional ? 2 : 1);
	
	int numInputWeights      = ls * pls;
	int numInternalWeights   = ls * els;
	
	// get ClockRNN state
	m_crStepStr = ((layerChild->HasMember("clock")) ? 
		       ((*layerChild)["clock"].GetString()) : (""));
	
	if (m_crStepStr.size()>0){
	    m_clockRNN     = true;
	    Cpu::int_vector temp;
	    // how many possible Hidden2Hidden matrices will be 
	    m_numH2Hmat    = ReadClockRNNOptions(m_crStepStr, m_crStep, els);
	    
	    // initialize the ClockRNN Hidden2Hidden matrices
	    m_h2hClockRNN.resize(els * els * (m_isBidirectional ? 2: 1) * (m_numH2Hmat), 0.0);
	    
	    m_crStepDevice = m_crStep;
	}else{
	    m_clockRNN     = false;
	    m_crStep.clear();
	    m_h2hClockRNN.clear();
	    m_crStepDevice.clear();
	}
	
	// Prepare the wrappers and pointers to weight and data
	// pointers to the bias weight [PL+0, PL+L-1]
	_rawBiasWeights = helpers::getRawPointer(this->weights()) + numInputWeights;

	// prepare a one vector for updating bias (parallel * timeLength)
	m_onesVec.resize(this->outputs().size()/this->size(),1.0);
	
	// the forward and backward operator
	if (m_isBidirectional){
	    // bidirectional case
	    
	    // temporary buffer tmp [els, timeLength * parallel]
	    Cpu::real_vector tmp(this->outputs().size()/2, 0);
	    m_fw.tmpOutputs        = tmp;    // initialize the Device vector using host vector
	    m_bw.tmpOutputs        = tmp;    
	    m_fw.tmpOutputErrors   = tmp;
	    m_bw.tmpOutputErrors   = tmp;
	    m_fw.unitActs          = tmp;
	    m_bw.unitActs          = tmp;
	    m_fw.unitDeltas        = tmp;
	    m_bw.unitDeltas        = tmp;
	    m_fw.unitActsBuf       = tmp;
	    m_bw.unitActsBuf       = tmp;
	    
	    if (m_clockRNN){
		// temporary buffer tmp2 [els, timeLength * parallel]
		Cpu::bool_vector tmp2(this->outputs().size()/2, false);
		m_fw.skipCR            = tmp2;
		m_bw.skipCR            = tmp2;
	    }
	    
	    // wrap the InputToHidden
	    m_fw.weightMatrices.InputToHiddenWrap = 
		helpers::Matrix<TDevice>(&this->weights(),        pls, els, 0);
	    m_fw.weightUpdateMatrices.InputToHiddenWrap = 
		helpers::Matrix<TDevice>(&this->_weightUpdates(), pls, els, 0);
	    m_bw.weightMatrices.InputToHiddenWrap = 
		helpers::Matrix<TDevice>(&this->weights(),        pls, els, numInputWeights/2);
	    m_bw.weightUpdateMatrices.InputToHiddenWrap = 
		helpers::Matrix<TDevice>(&this->_weightUpdates(), pls, els, numInputWeights/2);
	    
	    // wrap the HiddenToHidden
	    int numInputAndBiasF = ls * (pls + 1);
	    int numInputAndBiasB = ls * (pls + 1) + numInternalWeights/2;
	    m_fw.weightMatrices.HiddenToHiddenWrap = 
		helpers::Matrix<TDevice>(&this->weights(),        els, els, numInputAndBiasF);
	    m_fw.weightUpdateMatrices.HiddenToHiddenWrap = 
		helpers::Matrix<TDevice>(&this->_weightUpdates(), els, els, numInputAndBiasF);
	    m_bw.weightMatrices.HiddenToHiddenWrap = 
		helpers::Matrix<TDevice>(&this->weights(),        els, els, numInputAndBiasB);
	    m_bw.weightUpdateMatrices.HiddenToHiddenWrap = 
		helpers::Matrix<TDevice>(&this->_weightUpdates(), els, els, numInputAndBiasB);
	    
	    // wrap the matrix for bias
	    numInputAndBiasF = ls * pls;
	    numInputAndBiasB = ls * pls + ls/2;
	    m_fw.weightMatrices.BiasWrap = 
		helpers::Matrix<TDevice>(&this->weights(),        els, 1, numInputAndBiasF);
	    m_fw.weightUpdateMatrices.BiasWrap = 
		helpers::Matrix<TDevice>(&this->_weightUpdates(), els, 1, numInputAndBiasF);
	    m_bw.weightMatrices.BiasWrap = 
		helpers::Matrix<TDevice>(&this->weights(),        els, 1, numInputAndBiasB);
	    m_bw.weightUpdateMatrices.BiasWrap = 
		helpers::Matrix<TDevice>(&this->_weightUpdates(), els, 1, numInputAndBiasB);
	    	    
	    // wrap the weights for each time step
	    for (int timestep = 0; timestep < this->maxSeqLength(); ++timestep) {
                int rows      = this->size() / (m_isBidirectional ? 2 : 1);
                int cols      = this->parallelSequences();
                int offset    = timestep * rows * cols;
		int paralNum  = this->parallelSequences();
		
		timestep_matrices_t fm;
		timestep_matrices_t bm;

                fm.tmpOutputsWrapT = 
		    helpers::Matrix<TDevice>(&m_fw.tmpOutputs,      rows, cols, offset);
		fm.tmpOutputErrorsWrapT = 
		    helpers::Matrix<TDevice>(&m_fw.tmpOutputErrors, rows, cols, offset);
		fm.unitActsWrapT = 
		    helpers::Matrix<TDevice>(&m_fw.unitActs,        rows, cols, offset);
		fm.unitDeltasWrapT = 
		    helpers::Matrix<TDevice>(&m_fw.unitDeltas,      rows, cols, offset);
		fm.unitActsBufWrapT = 
		    helpers::Matrix<TDevice>(&m_fw.unitActsBuf,     rows, cols, offset);
		
		fm.unitDeltaP    = helpers::getRawPointer(m_fw.unitDeltas) + offset;

		bm.tmpOutputsWrapT = 
		    helpers::Matrix<TDevice>(&m_bw.tmpOutputs,      rows, cols, offset);
		bm.tmpOutputErrorsWrapT = 
		    helpers::Matrix<TDevice>(&m_bw.tmpOutputErrors, rows, cols, offset);
		bm.unitActsWrapT = 
		    helpers::Matrix<TDevice>(&m_bw.unitActs,        rows, cols, offset);
		bm.unitDeltasWrapT = 
		    helpers::Matrix<TDevice>(&m_bw.unitDeltas,      rows, cols, offset);
		bm.unitActsBufWrapT = 
		    helpers::Matrix<TDevice>(&m_bw.unitActsBuf,     rows, cols, offset);
		
		bm.unitDeltaP    = helpers::getRawPointer(m_bw.unitDeltas) + offset;

		// for ClockRNN
		if (m_clockRNN){
		    // tmpFlagCR:    a skip flag for each dimenison in a parallel block
		    // h2hMatrixIdx: in each time step, which hidden2hidden matrix should be used
		    //               h2hMatrix \in [0, MaxNumberofH2HMatrix]
		    Cpu::bool_vector tmpFlagCR(rows * paralNum, false);
		    int h2hMatrixIdx = DimSkipFlagCR(tmpFlagCR, m_crStep, timestep, paralNum)-1;
		    
		    if (h2hMatrixIdx<0){
			printf("Zero input at time %d", timestep);
			throw std::runtime_error("Error in timeresolution configuration");
		    }
		    
		    fm.skipCRPos = timestep * rows * paralNum;
		    bm.skipCRPos = timestep * rows * paralNum;

		    if (DEBUG_CLOCKRNN){
			printf("%d:\n", timestep);
			for(int i = 0; i < tmpFlagCR.size(); i++){
			    printf("%d ", tmpFlagCR[i]);
			}
			printf("\n");
		    }
		    
		    thrust::copy(tmpFlagCR.begin(), tmpFlagCR.end(), 
				 m_fw.skipCR.begin() + fm.skipCRPos);
		    thrust::copy(tmpFlagCR.begin(), tmpFlagCR.end(), 
				 m_bw.skipCR.begin() + bm.skipCRPos);
		    
		    fm.h2hIdx    = h2hMatrixIdx;
		    bm.h2hIdx    = h2hMatrixIdx;
		    //fm.skipCR = tmpSkipFlagCR;
		    //bm.skipCR = tmpSkipFlagCR;
		    
		    /*Cpu::int_vector  tmpCrStart; 
		    Cpu::int_vector  tmpCrEnd;
		    tmpCrStart.resize(m_crStep.size()/2,-1);
		    tmpCrEnd.resize(m_crStep.size()/2,-1);
		    int band = StartEndPos(m_crStep, timestep, tmpCrStart, tmpCrEnd);
		    if (band<1){
			printf("Zero input at time %d.", timestep);
			throw std::runtime_error("Error in Timeresolution");
		    }
		    fm.m_crS.resize(band,-1);
		    bm.m_crS.resize(band,-1);
		    fm.m_crE.resize(band,-1);
		    bm.m_crE.resize(band,-1);
		    thrust::copy(tmpCrStart.begin(), tmpCrStart.begin()+band, fm.m_crS.begin());
		    thrust::copy(tmpCrStart.begin(), tmpCrStart.begin()+band, bm.m_crS.begin());
		    thrust::copy(tmpCrEnd.begin(),   tmpCrEnd.begin()+band, fm.m_crE.begin());
		    thrust::copy(tmpCrEnd.begin(),   tmpCrEnd.begin()+band, bm.m_crE.begin());
		    */
		    
		    // wrap the temporary hidden to hidden matrix for ClockRNN
		    h2hMatrixIdx = h2hMatrixIdx * els * els;
		    fm.h2hWrap = helpers::Matrix<TDevice>(&m_h2hClockRNN, els, els, h2hMatrixIdx);
		    h2hMatrixIdx = h2hMatrixIdx + m_h2hClockRNN.size()/2;
		    bm.h2hWrap = helpers::Matrix<TDevice>(&m_h2hClockRNN, els, els, h2hMatrixIdx);

		}else{
		    //fm.skipCR.clear();
		    //bm.skipCR.clear();
		}
		m_fw.timestepMatrices.push_back(fm);
		m_bw.timestepMatrices.push_back(bm);
	    }
	    
	}else{
	    // unidirectional case, directly use the forward operator
	    Cpu::real_vector tmp(this->outputs().size(), 0);
	    m_fw.tmpOutputs        .swap(this->_outputs());     // just swap (directly use it)
	    m_fw.tmpOutputErrors   .swap(this->outputErrors()); // 
	    m_fw.unitActs          = tmp;
	    m_fw.unitDeltas        = tmp;
	    m_fw.unitActsBuf       = tmp;
	    
	    if (m_clockRNN){
		Cpu::bool_vector tmp2(this->outputs().size()/2, false);
		m_fw.skipCR        = tmp2;
	    }

	    // wrap the InputToHidden
	    m_fw.weightMatrices.InputToHiddenWrap = 
		helpers::Matrix<TDevice>(&this->weights(),        pls, els, 0);
	    m_fw.weightUpdateMatrices.InputToHiddenWrap = 
		helpers::Matrix<TDevice>(&this->_weightUpdates(), pls, els, 0);
	    
	    // wrap the HiddenToHidden
	    int numInputAndBiasF = ls * (pls + 1);
	    m_fw.weightMatrices.HiddenToHiddenWrap = 
		helpers::Matrix<TDevice>(&this->weights(),        els, els, numInputAndBiasF);
	    m_fw.weightUpdateMatrices.HiddenToHiddenWrap = 
		helpers::Matrix<TDevice>(&this->_weightUpdates(), els, els, numInputAndBiasF);
	    
	    // wrap the matrix for bias
	    numInputAndBiasF     = ls * pls;
	    m_fw.weightMatrices.BiasWrap = 
		helpers::Matrix<TDevice>(&this->weights(),        els, 1, numInputAndBiasF);
	    m_fw.weightUpdateMatrices.BiasWrap = 
		helpers::Matrix<TDevice>(&this->_weightUpdates(), els, 1, numInputAndBiasF);
	    
	    // wrap the weights for each time step
	    for (int timestep = 0; timestep < this->maxSeqLength(); ++timestep) {
                int rows      = this->size() / (m_isBidirectional ? 2 : 1);
                int cols      = this->parallelSequences();
                int offset    = timestep * rows * cols;
		int paralNum  = this->parallelSequences();

		timestep_matrices_t fm;

                fm.tmpOutputsWrapT = 
		    helpers::Matrix<TDevice>(&m_fw.tmpOutputs,      rows, cols, offset);
		fm.tmpOutputErrorsWrapT = 
		    helpers::Matrix<TDevice>(&m_fw.tmpOutputErrors, rows, cols, offset);
		fm.unitActsWrapT = 
		    helpers::Matrix<TDevice>(&m_fw.unitActs,        rows, cols, offset);
		fm.unitDeltasWrapT = 
		    helpers::Matrix<TDevice>(&m_fw.unitDeltas,      rows, cols, offset);
		fm.unitActsBufWrapT = 
		    helpers::Matrix<TDevice>(&m_fw.unitActsBuf,     rows, cols, offset);
		
		fm.unitDeltaP    = helpers::getRawPointer(m_fw.unitDeltas) + offset;

		// for ClockRNN
		if (m_clockRNN){		    
		    // tmpFlagCR:    a skip flag for each dimenison in a parallel block
		    // h2hMatrixIdx: in each time step, which hidden2hidden matrix should be used
		    //               h2hMatrix \in [0, MaxNumberofH2HMatrix]
		    Cpu::bool_vector tmpFlagCR(rows * paralNum, false);
		    int h2hMatrixIdx = DimSkipFlagCR(tmpFlagCR, m_crStep, timestep, paralNum)-1;

		    if (h2hMatrixIdx<0){
			printf("Zero input at time %d", timestep);
			throw std::runtime_error("Error in timeresolution configuration");
		    }

		    // fm.skipCR = tmpSkipFlagCR;
		    fm.skipCRPos = timestep * rows * paralNum;
		    thrust::copy(tmpFlagCR.begin(),    tmpFlagCR.end(), 
				 m_fw.skipCR.begin() + fm.skipCRPos);
		    fm.h2hIdx    = h2hMatrixIdx;
		    
		    /*
		    Cpu::int_vector  tmpCrStart; 
		    Cpu::int_vector  tmpCrEnd;
		    tmpCrStart.resize(m_crStep.size()/2,-1);
		    tmpCrEnd.resize(m_crStep.size()/2,-1);
		    int band = StartEndPos(m_crStep, timestep, tmpCrStart, tmpCrEnd);
		    if (band<1){
			printf("Zero input at time %d", timestep);
			throw std::runtime_error("Error in timeresolution configuration");
		    }
		    fm.m_crS.resize(band,-1);
		    fm.m_crE.resize(band,-1);
		    thrust::copy(tmpCrStart.begin(), tmpCrStart.begin()+band, fm.m_crS.begin());
		    thrust::copy(tmpCrEnd.begin(),   tmpCrEnd.begin()+band, fm.m_crE.begin());
		    */
		    
		    // wrap the temporary hidden to hidden matrix for ClockRNN
		    h2hMatrixIdx = els * els * h2hMatrixIdx;
		    fm.h2hWrap = helpers::Matrix<TDevice>(&m_h2hClockRNN, els, els, h2hMatrixIdx);
		    
		}else{
		    //fm.skipCR.clear();
		}
                m_fw.timestepMatrices.push_back(fm);
	    }
	}
	// swap it back
	if (!m_isBidirectional) {
            m_fw.tmpOutputs     .swap(this->_outputs());
            m_fw.tmpOutputErrors.swap(this->outputErrors());
        }
    }


    template <typename TDevice>
    RnnLayer<TDevice>::~RnnLayer()
    {
    }

    template <typename TDevice>
    const std::string& RnnLayer<TDevice>::type() const
    {
        static const std::string su("rnn");
        static const std::string sb("brnn");
        return (m_isBidirectional ? sb : su);
    }

    template <typename TDevice>
    bool RnnLayer<TDevice>::isBidirectional() const
    {
        return m_isBidirectional;
    }

    template <typename TDevice>
    void RnnLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
	TrainableLayer<TDevice>::loadSequences(fraction);

	// update the input data wrap
	// because the duration of the input sequence varies,
	// the wrappers must be created for each training epoch
	int rows = this->size() / (m_isBidirectional ? 2 : 1);
	int cols = this->curMaxSeqLength() * this->parallelSequences();
	
        m_precLayerOutputsWrapA = 
	    helpers::Matrix<TDevice>(&this->precedingLayer().outputs(), 
				     this->precedingLayer().size(), 
				     cols);

	m_onesVecWrap            = helpers::Matrix<TDevice>(&m_onesVec, 1, cols);
	if (m_isBidirectional){
	    m_fw.unitActsWrapA   = helpers::Matrix<TDevice>(&m_fw.unitActs,   rows, cols);
	    m_fw.unitDeltasWrapA = helpers::Matrix<TDevice>(&m_fw.unitDeltas, rows, cols);
	    m_bw.unitActsWrapA   = helpers::Matrix<TDevice>(&m_bw.unitActs,   rows, cols);
	    m_bw.unitDeltasWrapA = helpers::Matrix<TDevice>(&m_bw.unitDeltas, rows, cols);
	}else{
	    m_fw.unitActsWrapA   = helpers::Matrix<TDevice>(&m_fw.unitActs,   rows, cols);
	    m_fw.unitDeltasWrapA = helpers::Matrix<TDevice>(&m_fw.unitDeltas, rows, cols);
	}
	
	// Copy the Hidden2Hidden Matrix to each possible Hidden2Hidden Matrix format
	if (m_clockRNN){
	    int ls      = this->size();
	    int pls     = this->precedingLayer().size();
	    int h2hsize = rows * rows;
	    internal::SetH2HMatrix fn;
	    if (m_isBidirectional){
		fn.bandConfig = helpers::getRawPointer(m_crStepDevice);
		fn.bandNum    = m_crStepDevice.size()/2;
		fn.featDim    = rows;
		fn.matrixSize = h2hsize;
		
		fn.sourceW    = helpers::getRawPointer(this->weights()) + ls * (pls + 1);
		fn.targetW    = helpers::getRawPointer(m_h2hClockRNN);
		thrust::for_each(thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0)      + h2hsize * m_numH2Hmat,
				 fn);
		

		fn.sourceW    = helpers::getRawPointer(this->weights()) + ls * (pls + 1) + h2hsize;
		fn.targetW    = helpers::getRawPointer(m_h2hClockRNN)   + h2hsize * m_numH2Hmat;
		thrust::for_each(thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0)      + h2hsize * m_numH2Hmat,
				 fn);
	    }else{
		fn.bandConfig = helpers::getRawPointer(m_crStepDevice);
		fn.bandNum    = m_crStepDevice.size()/2;
		fn.featDim    = rows;
		fn.matrixSize = h2hsize;
		fn.sourceW = helpers::getRawPointer(this->weights())    + ls * (pls + 1);
		fn.targetW = helpers::getRawPointer(m_h2hClockRNN);
		thrust::for_each(thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0)      + h2hsize * m_numH2Hmat,
				 fn);
		
	    }
	    // For debug
	    // Show all the H2H matrices
	    if (DEBUG_CLOCKRNN){
	    Cpu::real_vector h2hMatrix_debug = m_h2hClockRNN;
	    int biasPos_debug = 0;
	    for (int i = 0; i < m_numH2Hmat; i++){
		printf("Forward: Matrix %d\n", i);
		for (int x_row = 0; x_row < rows; x_row++){
		    for (int y_col = 0; y_col < rows; y_col++){
			printf("%f ", h2hMatrix_debug[biasPos_debug + x_row + y_col*rows]);
		    }
		    printf("\n");
		}
		biasPos_debug  += h2hsize;
	    }
	    if (m_isBidirectional){
		for (int i = 0; i < m_numH2Hmat; i++){
		    printf("Forward: Matrix %d\n", i);
		    for (int x_row = 0; x_row < rows; x_row++){
			for (int y_col = 0; y_col < rows; y_col++){
			    printf("%f ", h2hMatrix_debug[biasPos_debug + x_row + y_col*rows]);
			}
			printf("\n");
		    }
		    biasPos_debug  += h2hsize;
		}
	    }
	    // Show matrix index
	    printf("Time-MatrixIdx\n");
	    for (int t = 0; t < this->curMaxSeqLength(); t++){
		printf("%5d-%3d ", t, m_fw.timestepMatrices[t].h2hIdx);
		if (t % 10 == 9) printf("\n");
	    }
	    }
	}
	
    }

    template <typename TDevice>
    void RnnLayer<TDevice>::computeForwardPass()
    {
	// for unidirectional LSTM, we can write the outputs directly in the layer output vector
        if (!m_isBidirectional) {
            m_fw.tmpOutputs.swap(this->_outputs());
        }

	// step1. from precedingLayer to this layer
	//        matrix multiplication, save to the m_fw and m_bw buffers
	{{
	     m_fw.unitActsWrapA.assignProduct(m_fw.weightMatrices.InputToHiddenWrap, true, 
					      m_precLayerOutputsWrapA, false);
	     if (m_isBidirectional){
		 m_bw.unitActsWrapA.assignProduct(m_bw.weightMatrices.InputToHiddenWrap, true, 
						  m_precLayerOutputsWrapA, false);
	     }
	}}

	// step2. from time 0 to T-1, compute and transform
	{{
	    // effective layer size (bi-directional half)
	    int els = this->size() / (m_isBidirectional ? 2 : 1);
	    // shift to the data of the next time step
	    // (one time step may contain multiple parallel utterances)
            int n   = this->parallelSequences() * els;             
	    
	    // forward states
            internal::ComputeBlockOutputFn fn;
            fn.effLayerSize       = els;
            fn.prevOutputDistance = -n;
            fn.bias               = this->bias();
            fn.patTypes           = helpers::getRawPointer(this->patTypes());
            fn.biasWeights        = _rawBiasWeights;
            fn.unitActs           = helpers::getRawPointer(m_fw.unitActs);
	    fn.unitActsBuf        = helpers::getRawPointer(m_fw.unitActsBuf);
	    
	    for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
		
		
		if (timestep != 0) {
		    // Add W*H_t-1 to output
		    if (m_clockRNN){
			m_fw.timestepMatrices[timestep].unitActsBufWrapT.assignProduct(
			      m_fw.timestepMatrices[timestep].h2hWrap,           true, 
			      m_fw.timestepMatrices[timestep-1].tmpOutputsWrapT, false);
		    }else{
			m_fw.timestepMatrices[timestep].unitActsBufWrapT.assignProduct(
			      m_fw.weightMatrices.HiddenToHiddenWrap,            true, 
			      m_fw.timestepMatrices[timestep-1].tmpOutputsWrapT, false);
		    }
		}

		// for ClockRNN
		if (m_clockRNN)
		    fn.skipCRNN  = (helpers::getRawPointer(m_fw.skipCR) + 
				    m_fw.timestepMatrices[timestep].skipCRPos);
		else
		    fn.skipCRNN  = NULL;
		
		thrust::transform(
		  thrust::counting_iterator<int>(n*timestep),
		  thrust::counting_iterator<int>(n*timestep) + n,
		  thrust::make_zip_iterator(
		    thrust::make_tuple(
		      thrust::constant_iterator<bool>(!timestep), 
		      thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
		  m_fw.tmpOutputs.begin() + n*timestep,
		  fn
		);
	    }
	    
	    if (m_isBidirectional) {
		fn.prevOutputDistance = +n;
                fn.biasWeights       += els;
                fn.unitActs           = helpers::getRawPointer(m_bw.unitActs);
		fn.unitActsBuf        = helpers::getRawPointer(m_bw.unitActsBuf);

		for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
		    
		    if (timestep != this->curMaxSeqLength()-1) {
			// Add W*H_t+1 to output
			if (m_clockRNN){
			    m_bw.timestepMatrices[timestep].unitActsBufWrapT.assignProduct(
				m_bw.timestepMatrices[timestep].h2hWrap, true, 
				m_bw.timestepMatrices[timestep+1].tmpOutputsWrapT, false);
			}else{
			    m_bw.timestepMatrices[timestep].unitActsBufWrapT.assignProduct(
				m_bw.weightMatrices.HiddenToHiddenWrap, true, 
				m_bw.timestepMatrices[timestep+1].tmpOutputsWrapT, false);
			}
		    }

		    // for ClockRNN
		    if (m_clockRNN)
			fn.skipCRNN  = (helpers::getRawPointer(m_bw.skipCR) + 
					m_bw.timestepMatrices[timestep].skipCRPos);
		    else
			fn.skipCRNN  = NULL;
		    
		    thrust::transform(
		      thrust::counting_iterator<int>(n*timestep),
		      thrust::counting_iterator<int>(n*timestep) + n,
		      thrust::make_zip_iterator(
		        thrust::make_tuple(
			  thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1), 
			  thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
		      m_bw.tmpOutputs.begin() + n*timestep,
		      fn
		      );
		}
	    }

	}}

	// step3. get results from m_fw, m_bw to this->outputs()

        // resort outputs
        if (m_isBidirectional) {
            internal::ResortOutputsFn fn;
            fn.layerSize    = this->size();
            fn.effLayerSize = this->size() / 2;
            fn.fwOutputs    = helpers::getRawPointer(m_fw.tmpOutputs);
            fn.bwOutputs    = helpers::getRawPointer(m_bw.tmpOutputs);
	    
            thrust::transform(
	      thrust::counting_iterator<int>(0),
	      thrust::counting_iterator<int>(0) + 
	      this->curMaxSeqLength() * this->parallelSequences() * this->size(),
	      this->_outputs().begin(),
	      fn
	    );
        }
        else {
            this->_outputs().swap(m_fw.tmpOutputs);
        }

    }


    template <typename TDevice>
    void RnnLayer<TDevice>::computeBackwardPass()
    {
	// step0. put the gradients back to the buffer
	//        gradients have been transformed by the next layer
        if (m_isBidirectional) {
            internal::ResortOutputErrorsFn fn;
            fn.layerSize      = this->size();
            fn.effLayerSize   = this->size() / 2;
            fn.fwOutputErrors = helpers::getRawPointer(m_fw.tmpOutputErrors);
            fn.bwOutputErrors = helpers::getRawPointer(m_bw.tmpOutputErrors);

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            thrust::for_each(
                thrust::make_zip_iterator(
			thrust::make_tuple(this->outputErrors().begin(),   
					   thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(
			thrust::make_tuple(this->outputErrors().begin()+n, 
					   thrust::counting_iterator<int>(0)+n)),
                fn
                );
        }
        else {
            m_fw.tmpOutputs     .swap(this->outputs());
            m_fw.tmpOutputErrors.swap(this->outputErrors());
        }
	
	// step1. compute the errors in each time step
	{{
	    int els = this->size() / (m_isBidirectional ? 2 : 1);
            int ls  = this->size();
	    int pls = this->precedingLayer().size();
	    int n   = this->parallelSequences() * els;

            // forward states
            internal::ComputeBlockErrorsFn fn;
            fn.effLayerSize       = els;
            fn.prevOutputDistance = -n;
            fn.patTypes           = helpers::getRawPointer(this->patTypes());
            fn.unitActs           = helpers::getRawPointer(m_fw.unitActs);
            fn.unitDeltas         = helpers::getRawPointer(m_fw.unitDeltas);

            for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
		
                // collect errors from previous timestep
                if (timestep != this->curMaxSeqLength()-1) {
		    		    
		    if (m_clockRNN){
			
			// for ClockRNN
			// step1. create the time-dependent matrix from hidden to hidden
			/*{{
			    internal::CreateH2HClockRnn fn;
			    fn.featDim      = els;
			    fn.bandNum      = 
				m_fw.timestepMatrices[timestep+1].m_crS.size();
			    fn.bandStart    = 
			       helpers::getRawPointer(m_fw.timestepMatrices[timestep+1].m_crS);
			    fn.bandEnd      = 
			       helpers::getRawPointer(m_fw.timestepMatrices[timestep+1].m_crE);
			    
			    fn.targetMatrix = helpers::getRawPointer(m_h2hClockRNN);
			    fn.sourceMatrix = helpers::getRawPointer(this->weights())+(ls*(pls+1));
			    thrust::for_each(thrust::counting_iterator<int>(0),
					     thrust::counting_iterator<int>(0) + els * els,
					     fn);
			    
					     }}*/
			// step2. get the errors
			m_fw.timestepMatrices[timestep].tmpOutputErrorsWrapT.addProduct(
				m_fw.timestepMatrices[timestep+1].h2hWrap, false, 
				m_fw.timestepMatrices[timestep+1].unitDeltasWrapT, false);

			// step3. set the gradient of the next step to zero
			/*{{
			    internal::CleanUnitDeltasClockRnn fn;
			    fn.skipCRNN    = 
			      helpers::getRawPointer(m_fw.timestepMatrices[timestep+1].skipCR);
			    fn.unitDeltas  = 
			      (m_fw.timestepMatrices[timestep+1].unitDeltaP);
			    thrust::for_each(thrust::counting_iterator<int>(0),
					     thrust::counting_iterator<int>(0) + els,
					     fn);
					     }}*/
		    }
		    else{
			// normal case
			m_fw.timestepMatrices[timestep].tmpOutputErrorsWrapT.addProduct(
				m_fw.weightMatrices.HiddenToHiddenWrap, false, 
				m_fw.timestepMatrices[timestep+1].unitDeltasWrapT, false);
		    }
                }
		
		if (m_clockRNN){
		    fn.skipCRNN  = helpers::getRawPointer(m_fw.skipCR) + 
			m_fw.timestepMatrices[timestep].skipCRPos;
		}else{
		    fn.skipCRNN = NULL;
		}

                // compute errors
                thrust::for_each(
                  thrust::make_zip_iterator(
		    thrust::make_tuple(
		      m_fw.tmpOutputErrors.begin() + n*timestep,   
		      thrust::counting_iterator<int>(n*timestep),   
		      thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1),   
		      thrust::constant_iterator<bool>(!timestep),   
		      thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
		  thrust::make_zip_iterator(
		    thrust::make_tuple(
		      m_fw.tmpOutputErrors.begin() + n*timestep +n, 
		      thrust::counting_iterator<int>(n*timestep)+n, 
		      thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1) + n, 
		      thrust::constant_iterator<bool>(!timestep)+n, 
		      thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)),
		  fn
		);
            }
	    
	    // backward states
            if (m_isBidirectional) {
                fn.prevOutputDistance = +n;
                fn.unitActs           = helpers::getRawPointer(m_bw.unitActs);
                fn.unitDeltas         = helpers::getRawPointer(m_bw.unitDeltas);

                for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
                    // collect errors from previous timestep
                    if (timestep != 0) {
			
			if (m_clockRNN){
			    // for ClockRNN
			    // step1. create the time-dependent matrix from hidden to hidden
			    /*{{
			    internal::CreateH2HClockRnn fn;
			    fn.featDim      = els;
			    fn.bandNum      = 
				m_bw.timestepMatrices[timestep-1].m_crS.size();
			    fn.bandStart    = 
			      helpers::getRawPointer(m_bw.timestepMatrices[timestep-1].m_crS);
			    fn.bandEnd      = 
			      helpers::getRawPointer(m_bw.timestepMatrices[timestep-1].m_crE);
			    fn.targetMatrix = helpers::getRawPointer(m_h2hClockRNN);
			    fn.sourceMatrix = (helpers::getRawPointer(this->weights()) 
					       + (ls*(pls+1)) + els * els);
			    thrust::for_each(thrust::counting_iterator<int>(0),
					     thrust::counting_iterator<int>(0) + els * els,
					     fn);
			    
			    }}*/
			    // step2. get the errors
			    m_bw.timestepMatrices[timestep].tmpOutputErrorsWrapT.addProduct(
				m_bw.timestepMatrices[timestep-1].h2hWrap, false, 
				m_bw.timestepMatrices[timestep-1].unitDeltasWrapT, false);

			    // step3. set the gradient of the next step to zero
			    /*{{
			    internal::CleanUnitDeltasClockRnn fn;
			    fn.skipCRNN    = 
			      helpers::getRawPointer(m_bw.timestepMatrices[timestep-1].skipCR);
			    fn.unitDeltas  = 
			      (m_bw.timestepMatrices[timestep-1].unitDeltaP);
			    thrust::for_each(thrust::counting_iterator<int>(0),
					     thrust::counting_iterator<int>(0) + els,
					     fn);
					     }}*/
			}
			else{
			    // normal case
			    m_bw.timestepMatrices[timestep].tmpOutputErrorsWrapT.addProduct(
				m_bw.weightMatrices.HiddenToHiddenWrap, false, 
				m_bw.timestepMatrices[timestep-1].unitDeltasWrapT, false);
			}
                    }
		    
		    if (m_clockRNN){
			fn.skipCRNN  = helpers::getRawPointer(m_bw.skipCR) + 
			    m_bw.timestepMatrices[timestep].skipCRPos;
		    }else{
			fn.skipCRNN = NULL;
		    }
		    
                    // compute errors
                    thrust::for_each(
                      thrust::make_zip_iterator(
			thrust::make_tuple(
			  m_bw.tmpOutputErrors.begin() + n*timestep,   
			  thrust::counting_iterator<int>(n*timestep),   
			  thrust::constant_iterator<bool>(!timestep),   
			  thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1),
			  thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
		      thrust::make_zip_iterator(
			thrust::make_tuple(
			  m_bw.tmpOutputErrors.begin() + n*timestep +n, 
			  thrust::counting_iterator<int>(n*timestep)+n, 
			  thrust::constant_iterator<bool>(!timestep)+n, 
			  thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1)+n,
			  thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)),
                        fn
                    );
                }
	    }	    
	}}
	
	// Move the step3 above here, we can set the unitDeltas to zero for all dimensions
	// that have been skipped by clockRNN
	if (m_clockRNN){
	    {{
		internal::CleanUnitDeltasClockRnn fn;
		fn.skipCRNN    = helpers::getRawPointer(m_fw.skipCR);
		fn.unitDeltas  = helpers::getRawPointer(m_fw.unitDeltas);
		thrust::for_each(thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0) + m_fw.skipCR.size(),
				 fn);
		if (m_isBidirectional){
		    fn.skipCRNN    = helpers::getRawPointer(m_bw.skipCR);
		    fn.unitDeltas  = helpers::getRawPointer(m_bw.unitDeltas);
		    thrust::for_each(thrust::counting_iterator<int>(0),
				     thrust::counting_iterator<int>(0) + m_bw.skipCR.size(),
				     fn);
		}
	    }}
	}
	
	// step2. back-propagate the error to the preceding layer
        {{
	    TrainableLayer<TDevice> *pl = 
		dynamic_cast<TrainableLayer<TDevice>*>(&this->precedingLayer());
            if (pl) {
                helpers::Matrix<TDevice> plErrorsMatrix(
			&pl->outputErrors(), pl->size(), 
			this->curMaxSeqLength()*this->parallelSequences());
                // forward states
                plErrorsMatrix.assignProduct(m_fw.weightMatrices.InputToHiddenWrap, false, 
					     m_fw.unitDeltasWrapA,                  false);
                // backward states
                if (m_isBidirectional) {
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.InputToHiddenWrap, false, 
					      m_bw.unitDeltasWrapA,                  false);
                }
            }
        }}

	// step3. update the weight and bias of this layer
	{{

	    int rows = this->size() / (m_isBidirectional ? 2 : 1);
	    int cols = (this->curMaxSeqLength()-1) * this->parallelSequences();
		
	    // we can also use matrix format
	    if (m_isBidirectional){
		// InputToHidden
		m_fw.weightUpdateMatrices.InputToHiddenWrap.assignProduct(
			m_precLayerOutputsWrapA, false,
			m_fw.unitDeltasWrapA,    true);
		m_bw.weightUpdateMatrices.InputToHiddenWrap.assignProduct(
			m_precLayerOutputsWrapA, false,
			m_bw.unitDeltasWrapA,    true);

		// HiddenToHidden
		// points to shift version of data
		int oneStepDataNum  = this->size()/2 * this->parallelSequences();
		helpers::Matrix<TDevice> shiftPreviousDataFw(
			&m_fw.tmpOutputs, rows, cols);
		helpers::Matrix<TDevice> unitDeltasShiftWrapAFw(
			&m_fw.unitDeltas, rows, cols, oneStepDataNum);
		m_fw.weightUpdateMatrices.HiddenToHiddenWrap.assignProduct(
			shiftPreviousDataFw,        false,
			unitDeltasShiftWrapAFw,     true);

		helpers::Matrix<TDevice> shiftPreviousDataBw(
			&m_bw.tmpOutputs, rows, cols, oneStepDataNum);
		helpers::Matrix<TDevice> unitDeltasShiftWrapABw(
			&m_bw.unitDeltas, rows, cols);
		m_bw.weightUpdateMatrices.HiddenToHiddenWrap.assignProduct(
			shiftPreviousDataBw,        false,
			unitDeltasShiftWrapABw,     true);

	    }else{
		m_fw.weightUpdateMatrices.InputToHiddenWrap.assignProduct(
			m_precLayerOutputsWrapA, false,
			m_fw.unitDeltasWrapA,    true);

		// HiddenToHidden
		// points to shift version of data
		int oneStepDataNum  = this->size() * this->parallelSequences();
		helpers::Matrix<TDevice> shiftPreviousDataFw(
			&m_fw.tmpOutputs, rows, cols);
		helpers::Matrix<TDevice> unitDeltasShiftWrapAFw(
			&m_fw.unitDeltas, rows, cols, oneStepDataNum);
		m_fw.weightUpdateMatrices.HiddenToHiddenWrap.assignProduct(
			shiftPreviousDataFw,        false,
			unitDeltasShiftWrapAFw,     true);

	    }
	    
	    // 
	    // compute the bias weight updates
	    {{
		if (m_isBidirectional){
		    m_fw.weightUpdateMatrices.BiasWrap.assignProduct(
			m_fw.unitDeltasWrapA, false,
			m_onesVecWrap,        true);
		    m_bw.weightUpdateMatrices.BiasWrap.assignProduct(
			m_bw.unitDeltasWrapA, false,
			m_onesVecWrap,        true);
		}else{
		    m_fw.weightUpdateMatrices.BiasWrap.assignProduct(
			m_fw.unitDeltasWrapA, false,
			m_onesVecWrap,        true);
		}
	    }}

	}}
	
	// re-swap the output errors and the tmp output errors of the forward pass
        if (!m_isBidirectional) {
            this->outputErrors().swap(m_fw.tmpOutputErrors);
            this->_outputs()    .swap(m_fw.tmpOutputs);
        }
    }

    template <typename TDevice>
    void RnnLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray, 
					const helpers::JsonAllocator &allocator) const
    {
        TrainableLayer<TDevice>::exportLayer(layersArray, allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("clock", m_crStepStr.c_str(), allocator);
    }
    
    // explicit template instantiations
    template class RnnLayer<Cpu>;
    template class RnnLayer<Gpu>;

} // namespace layers
