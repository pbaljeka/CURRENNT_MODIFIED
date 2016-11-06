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
 *****************************************************************************//*

*/


#ifdef _MSC_VER
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "MDNUnit.hpp"
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

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <fstream>
#include <cmath>

#define SKIP_MARKER helpers::NumericLimits<real_t>::max()
#define MDN_MIXTURE_VARIANCE_INI 1.0
#define PI_DEFINITION 3.141215


// VARADJUST: parameter for initializing the mean of mixture Gaussian
// The meean of differerent mixtures will be initialized with equal interval
// between [-VARADJUST VARADJUST] * var + data_mean
// EX, for 4 mixtures, [-2, 2] => [-1.5, -0.5, 0.5, 1.5]*var
// Note: 2.0 may be too large
//       change it to 0.8
// #define VARADJUST 0.8 // obsolete, replaced by config->m_varInitPara

//#define DEBUG_LOCAL 1     // I turn on DEBUG_LOCAL for debugging
//#define ALTER_TIEVAR 1    // Whether tied variance for all mixtures (not useful anymore)

#define DEBUG_LOCAL_LOCAL 1     // I turn on DEBUG_LOCAL for debugging
#define MIXTUREDYNDIAGONAL 1// Whether the transformation matrix in mixture_dyn should diagonal?


namespace internal {
namespace {
    
    /******************************************
     Utilities
    ******************************************/
    
    // copy the data from t.get<0> to the memory block started from Output 
    struct CopySimple
    {
	real_t *Output;           // output address to store data
	const char *patTypes;     // sentence termination check

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            int outputIdx = t.get<1>();
	    if (patTypes[outputIdx] == PATTYPE_NONE)
                return;

            // store the result
            *(Output+outputIdx) = t.get<0>();
        }
    };
    
    struct CopySimple2
    {
	real_t *Output;           // output address to store data
	real_t *in;
        __host__ __device__ void operator() (const int &t) const
        {
            *(Output+t) = *(in + t);
        }
    };
    
    // Covert the NN output x to the sigmoided f(x)
    struct ComputeSigmoid
    {
	int NNOutputSize;
	int startD;
	int endD;
	const char *patTypes;
	const real_t *NNoutput;

	__host__ __device__ real_t operator() (const int &outputIdx) const
	{
	    // layer size
	    // The data is not stored in continuous block, 
	    // thus, we have to calculate the position of the data
	    
	    // timeStep: which frame is it?
	    // dimStep:  which dimension in the NN output side this frame ?
	    const int timeStep = outputIdx / (endD - startD);
	    const int dimStep  = (outputIdx % (endD - startD)) + startD;
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return 0;

	    const real_t *data = NNoutput + (NNOutputSize * timeStep ) + dimStep;
	    real_t b = activation_functions::Logistic::fn(*data);
	    return b;
	}
    };

    struct CalculateOffsetFn
    {
	// Calculate the offset ( mean of [max, min]) of one frame
	// This offset is used in softmax and mixture model: 
	/// exp(x_i_t - offset) / sum_k (exp(x_k_t - offset))

        int NNOutputSize;        // layersize of the NN output
	int startD;              // which dim of the NN output is the first dim of mixture weight
	int endD;                // which dim of the NN output is the last dim of mixture weight
        const real_t *NNoutputs; // pointer to the NN output
        const char *patTypes;    // patterns to skip void time step

        __host__ __device__ real_t operator() (const int &patIdx) const
        {
            
	    // check if the pattern belongs to a sequence;
            // if not we return a certain number to avoid 
            // looking up patTypes for future calculations
            if (patTypes[patIdx] == PATTYPE_NONE)
                return SKIP_MARKER;

            // search for the min and max output
            real_t max = helpers::NumericLimits<real_t>::min();
            real_t min = helpers::NumericLimits<real_t>::max();

	    // point to the start of one frame
            const real_t *offOutputs = &NNoutputs[patIdx * NNOutputSize + startD];

	    // loop over the dimensions of this frame
            for (int i = 0; i < (endD - startD); ++i) {
                real_t x = offOutputs[i];
                min = helpers::min(min, x);
                max = helpers::max(max, x);
            }

            // calculate the offset
            real_t offset = (real_t)0.5 * (min + max);

            return offset;
        }
    };    
    
    struct CalculateExpFn
    {
	// Calculate the exponential exp()
	// Note: specifically for softmax and mixture weight
	//       exp(x - offset) will be calulated given offset
	int NNOutputSize;         // see CalculateOffsetFn above 
	int startD;
	int endD;
	const real_t *offset;     // the average of the [max, min]
	const char *patTypes;
	const real_t *NNOutput;

	__host__ __device__ real_t operator() (const int &outputIdx) const
	{
	    // timeStep: which frame is it?
	    // dimStep:  which dimension in the NN output side this frame ?
	    const int timeStep = outputIdx / (endD - startD);
	    const int dimStep  = (outputIdx % (endD - startD)) + startD;
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return SKIP_MARKER;

	    const real_t *data = NNOutput + (NNOutputSize * timeStep ) + dimStep;
	    real_t b = helpers::safeExp(*data - offset[timeStep]);
	    return b;
	}
    };

    struct CalculateExpSimpleFnForVar
    {
	// Calculate the expoential exp()
	// Note: specifically used for calculating the variance in Mixture model
	//       the output value will be floored by varFloor
	//       no offset is used here
	int NNOutputSize;
	int startD;
	int endD;
	real_t varFloor;           // the variance floor !!!
	const char   *patTypes;
	const real_t *NNOutput;

	__host__ __device__ real_t operator() (const int &outputIdx) const
	{

	    // timeStep: which frame is it?
	    // dimStep:  which dimension in the NN output side this frame ?
	    const int timeStep = outputIdx / (endD - startD);
	    const int dimStep  = (outputIdx % (endD - startD)) + startD;
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return SKIP_MARKER;

	    const real_t *data = NNOutput + (NNOutputSize * timeStep ) + dimStep;
	    real_t b = helpers::safeExp(*data);
	    b = (b < varFloor)?(varFloor):b;
	    return b;
	}
    };
	
    
    struct SumUpOutputsFn
    {
	// SumUp the value over the data of in a continuous memory space
	// For example, used in sum_k exp(w_k) for mixture weight normalization
	// Results, will be saved to the first argument (real_t& )
	
        int dimSize;            // sum over this number of dimension
        const real_t *outputs;  // data
        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            int patIdx = t.get<1>();

            // check if the pattern belongs to a sequence
            if (t.get<0>() == SKIP_MARKER)
                return;
	    
	    // point to the start of the continuous memory space
            const real_t *offOutputs = &outputs[patIdx * dimSize];

	    // sum up the outputs
            real_t sum = 0;
            for (int i = 0; i < dimSize; ++i)
                sum += offOutputs[i];

            // store the result
            t.get<0>() = sum;
        }
    };

    struct NormalizeOutputsFn
    {
	// Normalize the output using the sum of data
	// Note: for the last step of softmax
	//       normfacts points to the results cauclated bu SumUpOutputsFn

        int layerSize;
        const real_t *normFacts; // the data for normalization

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            int outputIdx = t.get<1>();

            // calculate the pattern index
            int patIdx = outputIdx / layerSize;

            // check if we can stop the calculation
            real_t normFact = normFacts[patIdx];
            if (normFact == SKIP_MARKER)
                return;

            // calculate the normalized value
            real_t x = t.get<0>() / normFact;

            // store the result
            t.get<0>() = x;
        }
    };
    

    struct CopyMean
    {
	// Copy the mean value from output of NN to MDN unit
	// Note: for mixture MDNUnit

        int NNOutputSize;       // layer size of NN output layer
	int featureDim;         // feature dimension of this MDN unit 
	int startD;             // which dim of the NN output is the first dim of the feature
	const real_t *NNOutput; // pointer to the output or NN
	const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            int outputIdx = t.get<1>();

            // calculate the pattern index
	    // timeStep: which frame is it?
	    // dimStep:  which dimension in the NN output side this frame ?
            const int timeStep = outputIdx / featureDim;
	    const int dimStep  = (outputIdx % featureDim) + startD;

	    if (patTypes[timeStep] == PATTYPE_NONE)
                return;
            // store the result
            t.get<0>() = *(NNOutput + timeStep * NNOutputSize + dimStep);
        }
    };
    


    /******************************************
     Lilelihood calculation
     and 
     Backpropagation functions for
     Sigmoid, softmax, mixture MDNUnit
     ******************************************/

    // Calculate errors (- log likelihood) for sigmoid
    struct ComputeSigmoidError
    {
	int startD;             // start position of data in NN output layer
	int endD;               // end position of data in NN output layer
	int startDOut;          // start position of data in observed data
	int layerSizeOut;       // dimension of the observed data
	const char *patTypes;
	const real_t *targets;  // targets data
	
	// from 1 to timesteps * para_dim
	__host__ __device__ real_t operator() (const thrust::tuple<real_t, int> &t) const
	{
	    
	    int outputIdx = t.get<1>();  // index
	    real_t prob   = t.get<0>();  // output of NN (probability)

	    // timeStep: which frame is it?
	    // dimStep:  which dimension in the NN output side this frame ?
	    const int timeStep = outputIdx / (endD - startD);
	    const int dimStep  = (outputIdx % (endD - startD)) + startDOut;
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return 0;
	    
	    // target data
	    const real_t *data = targets + (layerSizeOut * timeStep) + dimStep;
	    
	    // p(x>0)Delta(x>0) + (1-p(x<0))Delta(x<0)
	    real_t b = ((*data)>0)?(prob):(1-prob);
	    
	    real_t targetProb = helpers::max(helpers::NumericLimits<real_t>::min(), b);
	    return -1*log(targetProb);
	}
    };
    
    // Definition the back-propagation for sigmoid units
    struct ComputeSigmoidBP
    {
	int startD;
	int endD;
	int startDOut;
	int layerSizeOut;
	int layerSizeIn;
	const char *patTypes;
	const real_t *targets;  // targets data
	real_t *errors;         // errors of previous layer

	// from 1 to timesteps * para_dim
	__host__ __device__ void operator() (const thrust::tuple<real_t, int> &t) const
	{
	    
	    int outputIdx = t.get<1>();  // index
	    real_t prob   = t.get<0>();  // output of NN (probability)

	    // timeStep: which frame is it?
	    // dimStep:  which dimension in the NN output side this frame ?
	    const int timeStep = outputIdx / (endD - startD);
	    const int dimStep  = (outputIdx % (endD - startD));
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;
	    
	    // target data
	    const real_t *data = targets + (layerSizeOut * timeStep) + dimStep + startDOut;
	    
	    // position of the gradient data in the NN output layer side
	    const int pos_error= layerSizeIn * timeStep + dimStep + startD;

	    // calculate the gradient
	    // note: we assume the training data will be normalized with zero mean.
	    //       thus, data \in {-a, a}, where the value of a is determined by
	    //       the data corpus, usually, a positive number
	    *(errors+pos_error) = (*(data)>0)?(-1+prob):(prob);

	}
    };

    // Definition for the softmax Errors
    struct ComputeCrossEntropyErrorFn
    {
        int layerSize;
	int layerSizeOut;
	int startDOut;
	real_t *output;       // targets data
	const real_t *prob;   // mdn parameter (softmax)

	// from 1 to timesteps
        __host__ __device__ real_t operator() (const int outputIdx) const
        {

	    real_t *targetClass = (output + (outputIdx*layerSizeOut+startDOut));
	    // calculate the CEE
            if (targetClass < 0)
                return 0;
            else {
                int Idx    = outputIdx * layerSize + (int)(*targetClass);
                real_t targetProb = helpers::max(helpers::NumericLimits<real_t>::min(), 
						 prob[Idx]);
                return -1*log(targetProb);
            }
        }
    };
    
    // Definition for the back-propagation of softmax
    struct ComputeSoftMaxBP
    {
	int layerSizeIn;        //
	int layerSizeOut;     // 
	int startD;
	int startDOut;
	int paraDim;
	
	real_t *errors;       // the error buffer of the previous layer
	real_t *targetData;   // the ground truth value of target
	real_t *prob;         // the output probablity computed by this softmax unit
	const char *patTypes; 

	// from 1 to timesteps * softmax class number
	__host__ __device__ void operator() (const int outputIdx) const
	{

	    // timeStep: which frame is it?
	    // dimStep:  which dimension in the NN output side this frame ?
	    const int timeStep = outputIdx / paraDim;
	    const int dimStep  = outputIdx % paraDim;
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;

	    // target data (1-dimensional data)
	    const real_t *data = targetData + (layerSizeOut * timeStep) + startDOut;
	    bool hitflag = ((((*data) - dimStep)*((*data) - dimStep)) < 0.0001);

	    // position of the gradient data in the NN output layer side
	    const int pos_error= layerSizeIn * timeStep + dimStep + startD;
	    
	    real_t *probptr    = prob + outputIdx;
	    
	    // calculate the gradient
	    // note: we assume the target data is a real number that has not been normalized
	    *(errors+pos_error) = (hitflag)?(-1+(*probptr)):((*probptr));

	}
    };
    

    struct ComputeMixtureDistance
    {
	// Calculate the mixture distance \sum_d (x_d-\mu_d)^2/(2*std^2) for mixture model
	// This function is used in EM-style generation
	//  and forward anc backward propagation of mixture unit

	int  startDOut;
	int  layerSizeOut;
	int  mixture_num;
	int  featureDim;
	int  totaltime;
	bool tieVar;

	const char   *patTypes;
	const real_t *output;    // targets data
	const real_t *mdnPara;   // mean value of the mixture

	// from 1 to timesteps * num_mixture
	__host__ __device__ real_t operator() (const int idx) const
	{
	    
	    int timeStep = idx / mixture_num; //t.get<0>();
	    int mixIndex = idx % mixture_num; //t.get<1>(); 
	    
	    // point to the targets data x
	    int pos_data = (layerSizeOut * timeStep)+startDOut;
		
	    const real_t *data, *mean, *var;

	    if (patTypes[timeStep] == PATTYPE_NONE)
		return 0;
	    
	    // point to the mixture data (mean and variance)
	    // skip the mixture weight part
	    int pos =      totaltime * mixture_num;  
	    // point to the mean
	    int pos_mean = pos+timeStep*featureDim*mixture_num+mixIndex*featureDim;
	    
	    // skip mean && weight
	    pos =      totaltime * (mixture_num + mixture_num * featureDim); 
            #ifdef ALTER_TIEVAR
	    int pos_var  = pos+timeStep*mixture_num;
            #else
	    int pos_var  = pos+ (tieVar?
				 (timeStep * mixture_num + mixIndex) :
				 (timeStep * mixture_num * featureDim + mixIndex*featureDim));
            #endif
	    var  = mdnPara + pos_var;
	    
	    // accumulate the distance over dimension
	    real_t tmp = 0.0;
	    for (int i = 0; i<featureDim; i++){
		data = output  + pos_data + i;
		mean = mdnPara + pos_mean + i;
		var  = mdnPara + pos_var  + (tieVar?0:i);
		tmp += (*data-*mean)*(*data-*mean)/((*var)*(*var))/2;
		
	    }
	    return tmp;
	}
    };

    // Calculate the mixture distance \sum_d (x_d-\mu_d)^2/(2*std^2) for mixture model
    // This function is used in EM-style generation
    //  and forward anc backward propagation of mixture unit
    struct ComputeMixtureDistanceWithTransForm
    {
	int startDOut;
	int layerSizeOut;
	int mixture_num;
	int featureDim;
	int totaltime;
	bool tieVar;

	const char *patTypes;
	const real_t *output;    // targets data
	const real_t *mdnPara;   // mean value of the mixture
	const real_t *tranData; 
	// from 1 to timesteps * num_mixture
	__host__ __device__ real_t operator() (const int idx) const
	{
	    
	    int timeStep = idx / mixture_num; //t.get<0>();
	    int mixIndex = idx % mixture_num; //t.get<1>(); 
	    
	    // point to the targets data x
	    int pos_data = (layerSizeOut * timeStep)+startDOut;
		
	    const real_t *data, *mean, *var, *trans;

	    if (patTypes[timeStep] == PATTYPE_NONE)
		return 0;
	    
	    // point to the mixture data (mean and variance)
	    int pos =  totaltime * mixture_num;
	    int pos_mean = pos+timeStep*featureDim*mixture_num+mixIndex*featureDim;
	    pos     =  totaltime * (mixture_num + mixture_num * featureDim);

            #ifdef ALTER_TIEVAR
	    int pos_var  = pos+timeStep*mixture_num;
            #else
	    int pos_var  = pos+ (tieVar?
				 (timeStep * mixture_num + mixIndex) :
				 (timeStep * mixture_num * featureDim + mixIndex*featureDim));
            #endif
	    var  = mdnPara + pos_var;
	    
	    // pointer to the transformation part 
	    int pos_trans = idx;
	    
	    // accumulate the distance over dimension
	    real_t tmp = 0.0;
	    for (int i = 0; i<featureDim; i++){
		data = output    + pos_data + i;
		mean = mdnPara   + pos_mean + i;
		var  = mdnPara   + pos_var  + (tieVar?0:i);
		trans= tranData  + pos_trans+ i;
		tmp += (*data-*mean-*trans)*(*data-*mean-*trans)/((*var)*(*var))/2;
		
	    }
	    return tmp;
	}
    };

    // Copy the data from the output buffer to the target unit (for mixture_dyn unit)
    struct CopyTargetData
    {
	int startDOut;
	int layerSizeOut;
	int featureDim;

	const char *patTypes;
	const real_t *output;   // targets data
	real_t *target;   // 

	// from 1 to timesteps * num_mixture
	__host__ __device__ void operator() (const int idx) const
	{
	    
	    int timeStep  = idx / featureDim; //t.get<0>();
	    int featIndex = idx % featureDim; //t.get<1>(); 
	    
	    // point to the targets data x
	    int pos_data = (layerSizeOut * timeStep)+startDOut+featIndex;
		
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;
	    *(target+idx) = *(output + pos_data);
	    
	}
    };

    // Shift the mean value u => u + w^To + b
    struct ShiftBiasStep1
    {
	int featureDim;
	int mixNum;
	int totalTime;

	real_t   *linearPart;   // Wx'
	real_t   *biasPart;     // b
	real_t   *mdnPara;      // 

	// from 1 to timesteps * num_mixture
	__host__ __device__ void operator() (const int idx) const
	{
	    
	    int temp      = idx % (featureDim * mixNum); 
	    int featIndex = temp % featureDim; 
	    int timeStep  = idx / (featureDim * mixNum);
	    int mixIndex  = temp / featureDim;

	    int pos_mean;
	    // Add to the mean value
	    pos_mean = (totalTime * mixNum + 
			timeStep  * featureDim * mixNum + 
			mixIndex  * featureDim + featIndex); 
		
	    if (timeStep == 0){
		// skip the first time step
		return;
	    }else{
		*(mdnPara + pos_mean) = (*(mdnPara + pos_mean) + 
					 *(linearPart + idx)   + 
					 *(biasPart   + mixIndex * featureDim + featIndex)
					 );
	    }	    	    	    
	}
    };

    // Accumulating the statistics for BP on the linear regression part W^T o+b
    // -1 * posteriorP(k) * (O_t - (u + W_k ^ T O_t-1 + b_k)) / var^k_d / var^k_d
    struct ShiftBiasStep2
    {
	int featureDim;
	int mixNum;
	int totalTime;
	int startDOut;
	int layerSizeOut;

	real_t   *linearPart;   // Wx'
	real_t   *biasPart;     // b
	real_t   *target;       // x
	real_t   *mdnPara;      // 
	real_t   *postPbuff;
	bool    tieVar;

	// from 1 to timesteps * num_mixture
	__host__ __device__ void operator() (const int idx) const
	{
	    
	    int temp      = idx % (featureDim * mixNum); 
	    int featIndex = temp % featureDim; 
	    int timeStep  = idx / (featureDim * mixNum);
	    int mixIndex  = temp / featureDim;

	    // skip the first time step
	    if (timeStep == 0)
		return;
	    
	    // set the pointer
	    int pos_mean, pos_var, pos_data;
	    pos_mean = (totalTime * mixNum + 
			timeStep  * featureDim * mixNum + 
			mixIndex  * featureDim + featIndex); 
	    pos_var  = (totalTime * (mixNum + mixNum * featureDim)      + 
			timeStep  *  mixNum * (tieVar ? 1 : featureDim) + 
			mixIndex  * (tieVar ? 1 : featureDim)           +
			(tieVar ? 0 : featIndex)); 
	    
	    /*** FATAL ERROR **
	     * Posterior probability should be updated 
	     * Particularly, the size of the posterior probability buffer  will change !!!
	     * Split ShiftBias into ShiftBiasStep1 and ShiftBiasStep2
	     ******/
	    // pointer to the posterior P and sum of posterior P
	    const real_t *postP   = postPbuff + timeStep  * mixNum + mixIndex;
	    const real_t *sumPost = postPbuff + totalTime * mixNum + timeStep;
	    real_t posterior = helpers::safeExp((*postP) - (*sumPost));
	    
	    // point to the targets data x
	    pos_data = (layerSizeOut * timeStep) + startDOut + featIndex;
	    
	    // save x - u - wx'-b to dataBuff now
	    *(linearPart + idx) = (-1 * posterior * 
				   (*(target + pos_data) - *(mdnPara + pos_mean)) /
				   (*(mdnPara + pos_var)) / (*(mdnPara + pos_var)));
	    // No need to re-order the data
	    // save \phi(i)\sigma()(x-u-wx'-b) in time order
	    // pos_data = (mixIndex * totalTime + timeStep) * featureDim + featIndex; 
	    // *(tmpBuff+pos_data)=-1* posterior * (*(mdnPara + pos_var)) * (*(linearPart + idx));
	    
	}
    };

    //ShiftBiasStep1TiedCase
    struct ChangeMeanofMDN
    {
	// Shift the mean value u => u + w^To + b
	// This function is used by mixture_dyn and mixture_dynSqr
	// The difference is the source of the parameter w and b
	// mixture_dyn: w, b are the trainable weight of mixture_dyn
	// mixture_dynSqr: a, b are the trainable weight of predicted by the network
	int startDOut;
	int layerSizeOut;
	int featureDim;
	int mixNum;
	int totalTime;
	int trainableAPos;      // w, the w predicted by the network, I name it as a now
	int trainableBPos;      // b, the b which is predicted by the network
	int stepBack;           // how many steps to look back ?

	real_t   *linearPart;   // w, where w is trainable but shared across time steps
	real_t   *biasPart;     // b, where b is trainable but shared across time steps
	real_t   *targets;      // o_t-1
	real_t   *mdnPara;      // 
	
	bool      tieVar;

	// from 1 to timesteps * num_mixture
	__host__ __device__ void operator() (const int idx) const
	{
	    
	    int temp      = idx  % (featureDim * mixNum); 
	    int featIndex = temp %  featureDim; 
	    int timeStep  = idx  / (featureDim * mixNum);
	    int mixIndex  = temp /  featureDim;
	    
	    if (timeStep < stepBack){
		// skip the first time step
		return;
	    }
	    
	    int pos_mean, pos_data;
	    // Add to the mean value
	    pos_mean = (totalTime * mixNum + 
			timeStep  * featureDim * mixNum + 
			mixIndex  * featureDim + featIndex); 
	    pos_data = ((timeStep - stepBack)  * layerSizeOut) + startDOut + featIndex;
	    
	    /********************* FATAL ERROR *******************
	     * b can only be added once when stepBack==1
	     *****************************************************/
	    if (linearPart != NULL && biasPart != NULL){
		*(mdnPara + pos_mean) = ((*(mdnPara      + pos_mean))  + 
					 ((*(linearPart  + featIndex)) * (*(targets+pos_data))) + 
					 ((stepBack == 1)?(*(biasPart  + featIndex)):0)
					 );
	    }else{
		int pos_a, pos_b;
		int backOrder = (trainableBPos - trainableAPos) / featureDim;
		pos_a = totalTime * trainableAPos + timeStep * backOrder * featureDim + 
		    (stepBack - 1) * featureDim + featIndex;
		pos_b = totalTime * trainableBPos + timeStep * featureDim + featIndex;
		*(mdnPara + pos_mean) = ((*(mdnPara   + pos_mean))  + 
					 ((*(mdnPara  + pos_a)) * (*(targets+pos_data))) + 
					 ((stepBack == 1) ? (*(mdnPara   + pos_b)):0)
					 );
	    }
	}
    };

    struct ShiftBiasStep1TiedCaseDimensionAxis
    {
	// Shift the mean value u => u + w^To + b
	// This function is used by mixture_dyn and mixture_dynSqr
	// The difference is the source of the parameter w and b
	int startDOut;
	int layerSizeOut;
	int featureDim;
	int mixNum;
	int totalTime;
	int trainableAPos;      // w, the w predicted by the network, I name it as a now
	int trainableBPos;      // b, the b which is predicted by the network
	int stepBack;           // how many steps to look back ?

	real_t   *linearPart;   // w, where w is trainable but shared across time steps
	real_t   *biasPart;     // b, where b is trainable but shared across time steps
	real_t   *targets;      // o_t-1
	real_t   *mdnPara;      // 
	
	bool      tieVar;

	// from 1 to timesteps * num_mixture
	__host__ __device__ void operator() (const int idx) const
	{
	    
	    int timeStep  = idx  / (featureDim * mixNum);
	    int temp      = idx  % (featureDim * mixNum); 
	    int mixIndex  = temp /  featureDim;
	    int featIndex = temp %  featureDim;
	    
	    if (featIndex < stepBack){
		// skip the first dimension
		return;
	    }
	    
	    int pos_mean, pos_data;
	    // Add to the mean value
	    pos_mean = (totalTime * mixNum + 
			timeStep  * featureDim * mixNum + 
			mixIndex  * featureDim + featIndex); 
	    pos_data = (timeStep  * layerSizeOut) + startDOut + featIndex - stepBack;
	    
	    if (linearPart != NULL && biasPart != NULL){
		*(mdnPara + pos_mean) = ((*(mdnPara      + pos_mean))  + 
					 ((*(linearPart)) * (*(targets+pos_data))) + 
					 ((stepBack==1)  ? (*(biasPart)):0)
					 );
	    }else{
		/* not implemented for context-dependent case */
	    }
	}
    };

    // ShiftBiasStep2TiedCase
    struct AccumulateGradient
    {
	// Accumulating the statistics for BP on the linear regression part W^T o+b
	// -1 * posteriorP(k) * (O_t - (u + W_k ^ T O_t-1 + b_k)) * O_t-1 / var^k_d / var^k_d
	// -1 * posteriorP(k) * (O_t - (u + W_k ^ T O_t-1 + b_k)) / var^k_d / var^k_d
	
	int featureDim;
	int mixNum;
	int totalTime;
	int startDOut;
	int layerSizeOut;
	int backOrder;

	real_t   *gradBuf;
	real_t   *target;       // x
	real_t   *mdnPara;      // 
	real_t   *postPbuff;
	bool      tieVar;

	// from 1 to timesteps * num_mixture
	__host__ __device__ void operator() (const int idx) const
	{
	    
	    int temp      = idx % (featureDim * mixNum); 
	    int featIndex = temp % featureDim; 
	    int mixIndex  = temp / featureDim;
	    int temp2     = idx  / (featureDim * mixNum);
	    int timeStep  = temp2 % totalTime;
	    int backStep  = temp2 / totalTime + 1;


	    // skip the first time step
	    if (timeStep < backStep)
		return;
	    
	    // set the pointer
	    int pos_mean,  pos_var, pos_data, pos_dataShift;
	    int pos_buffW, pos_buffb;
	    pos_mean = (totalTime * mixNum + 
			timeStep  * featureDim * mixNum + 
			mixIndex  * featureDim + featIndex); 
	    pos_var  = (totalTime * (mixNum + mixNum * featureDim)      + 
			timeStep  *  mixNum * (tieVar ? 1 : featureDim) + 
			mixIndex  * (tieVar ? 1 : featureDim)           +
			(tieVar ? 0 : featIndex)); 
	    
	    // pointer to the posterior P and sum of posterior P
	    const real_t *postP   = postPbuff + timeStep  * mixNum + mixIndex;
	    const real_t *sumPost = postPbuff + totalTime * mixNum + timeStep;
	    real_t posterior = helpers::safeExp((*postP) - (*sumPost));
	    
	    // point to the targets data x
	    /***********p******** Fatal Error ****************************
	     * : how could I just use pos_dataShift as pos_data ??? 
	     ************************************************************/
	    pos_data = (layerSizeOut * (timeStep)) + startDOut + featIndex;
	    pos_dataShift = (layerSizeOut * (timeStep - backStep)) + startDOut + featIndex;
	    
	    // Note, dimension -> backstep -> mixture -> time
	    pos_buffW = (timeStep * mixNum + mixIndex) * featureDim * backOrder +
		        (backStep-1) * featureDim + featIndex;
	    
	    //pos_buffW = (timeStep * mixNum + mixIndex) * featureDim + featIndex;
	    //pos_buffb = pos_buffW + totalTime * featureDim * mixNum;
	    
	    
	    real_t grad = (-1 * posterior * (*(target + pos_data) - *(mdnPara + pos_mean)) /
			   (*(mdnPara + pos_var)) / (*(mdnPara + pos_var)));
	    
	    *(gradBuf + pos_buffW) = grad * (*(target + pos_dataShift));
	    
	    if (backStep == 1){
		// do this for one time when backStep == 1
		/* *** FATAL ERROR **
		 *  Can't use pos_buffW here
		 * ******************* */
		//pos_buffb = pos_buffW + backOrder * totalTime * featureDim * mixNum;
		pos_buffb = backOrder * totalTime * featureDim * mixNum;
		pos_buffb+= (timeStep * mixNum + mixIndex) * featureDim + featIndex;

		*(gradBuf + pos_buffb) = grad;
	    }
	}
    };
    
    // Block 1025x10

    struct TanhAutoRegWeightStep1
    {
	int     featureDim;
	int     backOrder;
	real_t *weight;
	real_t *weightOut;
        __host__ __device__ void operator() (const int &Idx) const
        {
            *(weightOut + Idx) =  (Idx < (backOrder * featureDim))? 
		(activation_functions::Tanh::fn(*(weight+Idx))):0;
        }
    };

    struct TanhAutoRegWeightStep2
    {
	int     featureDim;
	real_t *weight;
	real_t *weightOut;
        __host__ __device__ void operator() (const int &idx) const
        {
	    *(weightOut + idx) = (idx < featureDim) ? 
		((*(weight + idx)) + (*(weight + idx + featureDim))) : 
		(-1 * (*(weight + idx - featureDim)) * (*(weight + idx)));
        }
    };
    
    struct TanhAutoRegWeightTime
    {
	int     featureDim;
	int     backOrder;
	int     NNOutputSize;
	int     startD;
	real_t *NNOutput;
	real_t *weightOut;
	const char   *patTypes;
	
        __host__ __device__ void operator() (const int &Idx) const
        {

	    int timeStep  = Idx  / (featureDim * backOrder);
	    int temp      = Idx  % (featureDim * backOrder);
	    int featIdx   = temp % featureDim;
	    int backStep  = temp / featureDim + 1;
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
                return;
	    
	    int a1, a2;
	    // point to the first 
	    a1  = timeStep * NNOutputSize + startD + featIdx;
	    // point to the next one or void
	    
	    if (backOrder == 2){
		a2  = timeStep * NNOutputSize + startD + featIdx + featureDim;
		*(weightOut + Idx) = (backStep == 1)? 
		    ((activation_functions::Tanh::fn(*(NNOutput+a1))) + 
		     (activation_functions::Tanh::fn(*(NNOutput+a2)))) :
		    ((activation_functions::Tanh::fn(*(NNOutput+a1))) * 
		     (activation_functions::Tanh::fn(*(NNOutput+a2))) * -1);
	    }else{
		*(weightOut + Idx) = 
		    (activation_functions::Tanh::fn(*(NNOutput+a1)));
	    }
        }
    };
    
    struct TanhAutoRegConvolution
    {
	// This function converts the poles of cascade form AR into coefficients of classical form
	// input: weightPoles alpha_1, alpha_2, ... alpha_N, the raw pole (before tanh) of AR
	//        (1-tanh(alpha_1)z^-1) * ... * (1-tanh(alpha_N)z^-1)
	// output: [1 a_1 a_2 a_3 ... a_N], of AR 1 + \sum_i^N a_i z^-1
	// 
	// Note: if flagnegative == true, 
	//          the output is [1 a_1 ... a_N], where the AR is 1 - \sum_i^N a_i z^-1
	//       if tanhFactor   == true
	//          the output is [1*(1-tanh(factor)^2) a_1*(1-tanh(factor)^2)...]
	//          used for back-propgation
	int     featureDim;
	int     backOrder;
	real_t *weightPoles;     // pointer to the poles of AR
	real_t *weightCoef1;     // pointer to the buffer for the coefficient of classical form
	real_t *weightCoef2;     // pointer to the second buffer 
	real_t *tanhFactor;      // for back-propagation, multiply (1-alpha^2)

	bool    flagnegative;       // whether multiply -1

	__host__ __device__ void operator() (const int dimIdx) const
	{
	    // assume weightCoef1 and weightCoef2 are initilized as zero
	    // if backOrder == 0 in back-propagation stage (1-order AR)
	    //    only need to multiply the (1-tanh(alpha)^2)
	    // Note: backOrder == this->backOrder -1
	    if (backOrder>0){
		// for n == 1
		weightCoef1[dimIdx] = 1;
		weightCoef1[dimIdx + featureDim] = -1 * 
		    activation_functions::Tanh::fn(weightPoles[dimIdx]);

		// for n = 2:(backOrder+1)
		for (int n = 2; n <= backOrder; n++){
		    for (int i = 1; i <= n; i++){
			weightCoef2[dimIdx + i*featureDim] = 
			    (weightCoef1[dimIdx + (i-1) * featureDim]
			     * (-1) *
			     activation_functions::Tanh::fn(weightPoles[dimIdx+(n-1)*featureDim]));
		    }
		    for (int i = 1; i <= n; i++){
			weightCoef1[dimIdx + i*featureDim] = (weightCoef1[dimIdx + i * featureDim]
							      +
							      weightCoef2[dimIdx + i * featureDim]);
		    }
		}
	    
		// multiply -1
		if (flagnegative){
		    for (int n = 1; n <= backOrder; n++){
			weightCoef1[dimIdx + n*featureDim] = -1 * weightCoef1[dimIdx+n*featureDim];
		    }
		}
	    }else{
		weightCoef1[dimIdx] = 1;
	    }
	    
	    if (tanhFactor){
		// Note, the first coefficients should also be changed
		for (int n = 0; n <= backOrder; n++){
		    weightCoef1[dimIdx + n*featureDim] = weightCoef1[dimIdx + n*featureDim] * 
			(1 - activation_functions::Tanh::fn(tanhFactor[dimIdx])
			   * activation_functions::Tanh::fn(tanhFactor[dimIdx]));
		}
	    }
	}
    };


    struct TanhAutoRegConvolutionARComplex
    {
	// Similar to the TanhAutoRegConvolution above,
	// for AR complex poles
	
	int     featureDim;
	int     casOrder;
	real_t *weightPoles;     // pointer to the poles of AR
	real_t *weightCoef1;     // pointer to the buffer for the coefficient of classical form
	real_t *weightCoef2;     // pointer to the second buffer 
	real_t *tanhFactor;      // for back-propagation, multiply (1-alpha^2)

	bool    flagnegative;       // whether multiply -1
	bool    aRealPole;
	
	__host__ __device__ void operator() (const int dimIdx) const
	{
	    // assume weightCoef1 and weightCoef2 are initilized as zero
	    // if casOrder == 0 in back-propagation stage (2nd order AR)
	    //    only need to multiply the (1-tanh(alpha)^2)
	    //    
	    
	    if (casOrder>0){
		
		// for k == 1, the first filter
		weightCoef1[dimIdx] = 1;
		if (aRealPole){
		    // weightCoef1[dimIdx + featureDim]   = -1 * weightPoles[dimIdx];
		    weightCoef1[dimIdx + featureDim]   = weightPoles[dimIdx];
		    weightCoef1[dimIdx + featureDim*2] = 0;
		}else{
		    //weightCoef1[dimIdx + featureDim]   = -2 * weightPoles[dimIdx] 
		    // * sqrt(weightPoles[dimIdx + featureDim]);
		    //weightCoef1[dimIdx + featureDim*2] = -1 * weightPoles[dimIdx + featureDim];
		    weightCoef1[dimIdx + featureDim]   = 2 * weightPoles[dimIdx] 
			                                 * sqrt(weightPoles[dimIdx + featureDim]);
		    weightCoef1[dimIdx + featureDim*2] = weightPoles[dimIdx + featureDim];
		}
		

		// for the rest filter
		for (int k = 2; k <= casOrder; k++){
		    int n = (k-1) * 2;
		    for (int i = 1; i <= (n+2); i++){
			if (i < (n+2)){
			    //weightCoef2[dimIdx + i*featureDim] = 
			    //	(weightCoef1[dimIdx + (i-1) * featureDim]
			    //	 * (-2) * weightPoles[dimIdx + (k-1)*2*featureDim] 
			    //	 * sqrt(weightPoles[dimIdx + (k-1)*2*featureDim +featureDim]));
			    weightCoef2[dimIdx + i*featureDim] = 
				(weightCoef1[dimIdx        + (i-1) * featureDim]
				 * weightPoles[dimIdx      + (k-1)*2*featureDim] 
				 * sqrt(weightPoles[dimIdx + (k-1)*2*featureDim +featureDim]));
			}else{
			    weightCoef2[dimIdx + i*featureDim] = 0;
			}
			if (i>1){
			    //weightCoef2[dimIdx + i*featureDim] += 
			    //  (weightCoef1[dimIdx + (i-2) * featureDim] * 
			    //   -1 * weightPoles[dimIdx + (k-1)*2*featureDim +featureDim]);
			    weightCoef2[dimIdx + i*featureDim] += 
				(weightCoef1[dimIdx + (i-2) * featureDim] * 
				 weightPoles[dimIdx + (k-1) * 2 * featureDim + featureDim]);
			}
			
		    }
		    for (int i = 1; i <= (n+2); i++){
			weightCoef1[dimIdx + i*featureDim] = (weightCoef1[dimIdx + i * featureDim]
							      +
							      weightCoef2[dimIdx + i * featureDim]);
		    }
		}
	    
		// multiply -1
		if (flagnegative){
		    for (int n = 1; n <= casOrder * 2; n++){
			weightCoef1[dimIdx + n*featureDim] = -1 * weightCoef1[dimIdx+n*featureDim];
		    }
		}
	    }else{
		// 2nd order filter case
		weightCoef1[dimIdx] = 1;
		
	    }
	    
	    // 
	    if (tanhFactor){
		
	    }
	}
    };

    struct TanhAutoRegGradientPre
    {
	// preparing for computing the gradients of classical coefficients
	// w.r.t the poles
	int featureDim;       // dimension of the feature
	int backOrder;        // the order of the classifical form AR
	int coefNum;          // number of coefficients in each filter
	
	real_t *weightPtr;
	real_t *weightBuff;
	
	__host__ __device__ void operator() (const int idx) const
	{
	    
	    int featIndex = idx % featureDim; 
	    int temp      = idx / featureDim;  
	    int orIndex1  = temp % (backOrder-coefNum); // 
	    int orIndex2  = temp / (backOrder-coefNum); // 
	    
	    // let's call the filter based on the N-1 sub-filter as leave-one cascade filter
	    // orIndex1: order of the coefficients in each leave-one cascade filter
	    // orIndex2: which leave-one cascade filter ?
	    // coefNum == 1 => cascade of 1st order filter, AR based on real poles
	    // coefNum == 2 => cascade of 2nd order filter, AR based on complex poles
	    
	    // for coefNum ==1, the index of filter coefficients
	    //                              0        backOrder-2
	    // orIndex2  = 0            => [1,2 ..., backOrder-1]      (#coef = backOrder-1)
	    // orIndex2  = 1            => [0,2,..., backOrder-1] 
	    // ...
	    // orIndex2  = backOrder-1  => [0,1,2,...,backOrder-2]
	    
	    // for coefNum ==2, the index of filter coefficients
	    //                                0 1,       backOrder-3
	    // orIndex2  = 0              => [2,3,...,   backOrder-1]  (#coef = backOrder-2)
	    // orIndex2  = 1              => [0,1,4,5,...backOrder-1]
	    // orIndex2  = backOrder/2-1  => [0,1,2,..., backOrder-3]  
	    
	    

	    // orIndex1 < orIndex2 decides which order should be skippped
	    real_t *sourcePtr = weightPtr + (((orIndex1/coefNum) < orIndex2)?
					     (orIndex1 * featureDim + featIndex):
					     ((orIndex1+coefNum) * featureDim + featIndex));
	    
	    // backOrder + 1, here means the all coefficients + bias
	    real_t *targetPtr = weightBuff + orIndex2 * featureDim * (backOrder + 1) +
		                orIndex1 * featureDim + featIndex;
	    
	    (*targetPtr) = (*sourcePtr);
	}
    };

    struct TanhAutoRegGradientTransform
    {
	int featureDim;
	int backOrder;
	real_t *rawGradients;   // pointer to the gradient w.r.t to coefficients of classical AR
	real_t *weightBuff;     // pointer to the graident w.r.t to pole of cascade AR
	real_t *tempResult;     // to store the results
	
	__host__ __device__ void operator() (const int idx) const
	{
	    
	    int featIndex = idx % featureDim; 
	    int orIndex1  = idx / featureDim;
	    
	    real_t *gradientPtr;
	    real_t *factorPtr;
	    real_t temp = 0.0;
	    for (int n = 0; n < backOrder; n++){
		gradientPtr = rawGradients + n * featureDim + featIndex;
		factorPtr   = weightBuff   + 2 * orIndex1 * (backOrder+1) * featureDim + 
		              n * featureDim + featIndex;
		temp = temp + (*gradientPtr) * (*factorPtr);
	    }
	    (* (tempResult + orIndex1 * featureDim + featIndex)) = temp;
	}
    };

    struct TanhAutoRegGradientTransformARComplex
    {
	int     featureDim;
	int     backOrder;
	real_t *rawGradients;   // pointer to the gradient w.r.t to coefficients of classical AR
	real_t *weightBuff;     // pointer to the graident w.r.t to pole of cascade AR
	real_t *tempResult;     // to store the results
	real_t *alphabeta;      // the transformed coefficients tanh(\alpha) sigmoid(\beta)
	
	bool    aRealPole;
	
	__host__ __device__ void operator() (const int idx) const
	{
	    
	    int featIndex = idx % featureDim;     // dimension of the feature
	    int filterIdx = idx / featureDim; 
	    	    
	    real_t *gradientPtr;
	    real_t *factorPtr;
	    
	    real_t a1     = 0.0;
	    real_t a2     = 0.0;
	    
	    // the number of coefficients should backOrder-1
	    for (int n = 0; n < (backOrder-1); n++){
		
		// backorder + 1 here denotes distance in the memory
		factorPtr   = weightBuff   + 2 * filterIdx * (backOrder+1) * featureDim + 
		              n * featureDim + featIndex;
		
		// gradients for the first coefficients in the 2nd-order filter
		gradientPtr = rawGradients + n     * featureDim + featIndex;
		a1          = a1 + (*gradientPtr)  * (*factorPtr);

		// gradients for the second coefficients in the 2nd-order filter
		gradientPtr = rawGradients + (n+1) * featureDim + featIndex;
		a2          = a2 + (*gradientPtr)  * (*factorPtr);
	    }
	    
	    real_t alpha = *(alphabeta+(filterIdx*2)*featureDim + featIndex); // tanh（alpha）
	    real_t beta  = *(alphabeta+(filterIdx*2+1)*featureDim + featIndex);// sigmoid（beta）
	    
	    if (filterIdx == 0 && aRealPole){
		// the pole of the 1st order filter 
		(* (tempResult + (filterIdx * 2) * featureDim + featIndex)) = 
		    a1 * (-1) * (1 - alpha * alpha);

		// the filter with only one pole based on tanh
		(* (tempResult + (filterIdx * 2 + 1) * featureDim + featIndex)) = 0;
	    }else{
		// gradient w.r.t to the alpha
		(* (tempResult + (filterIdx * 2) * featureDim + featIndex)) = 
		    a1 * (-2) * sqrt(beta) * (1 - alpha * alpha);

		// gradient w.r.t to the beta
		(* (tempResult + (filterIdx * 2 + 1) * featureDim + featIndex)) = 
		    (a1 * (-1) * alpha / sqrt(beta) - a2) * beta * (1-beta);
	    }
	}
    };
    
    struct ARRMDNGradientsSetZero
    {
	int     keepGradient;   // which order should be updated ? (0 means the 1st order)
	int     featureDim;     // dimension of the feature vector
	real_t *gradients;      

	__host__ __device__ void operator() (const int idx) const
	{
	    if (keepGradient < 0 || keepGradient == (idx / featureDim)){
		
	    }else{
		*(gradients + idx) = 0.0;
	    }
	}
    };

    struct ARComplexPolesParameterTransform
    {
	int     featDim;
	int     filterNum;
	bool    withRealPole;
	real_t *paraMem;        // ptr to the original parameter
	real_t *paraBuf;        // to store the transformed parameter
	__host__ __device__ void operator() (const int idx) const
	{
	    int paraIdx    = idx     / featDim;
	    int filterIdx  = paraIdx / 2;
	    
	    paraIdx    = paraIdx % 2;
	    if (filterIdx == 0 && withRealPole){
		// for the first filter with only one real pole
		*(paraBuf + idx) = ((paraIdx == 0) ? 
				    (activation_functions::Tanh::fn(*(paraMem + idx))) :
				    (0.0));
	    }else{
		*(paraBuf + idx) = ((paraIdx == 0) ? 
				    (activation_functions::Tanh::fn(*(paraMem + idx))) :
				    (activation_functions::Logistic::fn(*(paraMem + idx))));
	    }
	}

    };


    // Block 1025x07 Gradient calculation in old methods
    // Block 1025x09 Gradient calculation in old methods

    struct ComputeMixtureError
    {
	// Calculat the -log likelihood for mixture model
	int   startD;
	int   startDOut;
	int   layerSizeOut;
	int   mixture_num;
	int   featureDim;
	bool  tieVar;

	const char *patTypes;
	real_t     *meanDis;  // where is the mixture distancen calculated by the function above
	                      //  this memory block will save the likelihood 
	                      //  of each mixture
	real_t *mdnPara;      // point to the parameter of the mdnPara
	int totalTime;        // number of time steps * parallel sentence

	// from 1 to timesteps
	__host__ __device__ real_t operator() (const int timeStep) const
	{
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return 0;
	    
	    // point to the mixture distance calculated above
	    real_t tmp         = helpers::NumericLimits<real_t>::logZero();
	    real_t tmptmp;
	    real_t *meanDisPtr = meanDis + timeStep * mixture_num;
	    
	    // point to the weight
	    int pos = timeStep*mixture_num;
	    const real_t *mixtureW = mdnPara + pos; 
	    
	    // point to the variance
	            pos = totalTime * (mixture_num + mixture_num * featureDim);
	    real_t *var = mdnPara + pos + timeStep * mixture_num * (tieVar?1:featureDim); 
	    
	    // calculate over each mixture
	    for (int i = 0; i<mixture_num; i++){
		//tmptmp = log(*(mixtureW+i))+(*(meanDisPtr+i))-featureDim/2*log(2*PI_DEFINITION);
		tmptmp = log(*(mixtureW+i))-(*(meanDisPtr+i))-featureDim/2*log(2*PI_DEFINITION);

                #ifdef ALTER_TIEVAR
		tmptmp = tmptmp - featureDim*helpers::safeLog(*(var));
		for (int j = 1; j<mixture_num; j++)
		    *(var+j) = 0;
                #else
		if (tieVar)
		    tmptmp = tmptmp - featureDim*helpers::safeLog(*(var+i));
		else
		    for (int j = 0; j<featureDim; j++)
			tmptmp = tmptmp - helpers::safeLog(*(var+i*featureDim+j));
                #endif

		if (tmptmp < helpers::NumericLimits<real_t>::lSMALL())
		    tmptmp = helpers::NumericLimits<real_t>::lSMALL();
		
		//tmptmp = helpers::safeExp(tmptmp);
		tmp    = helpers::logAdd(tmp, tmptmp);
		
		// Save w_i phi_i (likelihood on this mixture)
		// this likelihood will be used for calculating the posterior 
		// in backpropagation stage
		*(meanDisPtr+i) = tmptmp; 
	    }
	    // save sum_i^mixture_num w_i phi_i for calculating the posterior
	    meanDisPtr = meanDis + totalTime * mixture_num + timeStep;
	    *meanDisPtr= tmp;

	    //return -1*helpers::safeLog(tmp);
	    return -1*(tmp);
	}
    };
    

    struct ComputeBPmixtureWeight
    {
	// BP for the mixture weight of mixture unit
	int mixture_num;
	int NNOutputSize;
	int startD;
	const char *patTypes;
	const real_t *meanDis; 
	const real_t *mdnPara; // mean value of the mixture
	real_t *errors;        // outputerrors (gradient buffer) of preceding layer
	int totalTime;   //

	// from 1 to timesteps * numMixture 
	__host__ __device__ void operator() (const int outputIdx) const
	{
	    
	    const int timeStep = outputIdx / mixture_num;
	    const int mixtureI = (outputIdx % mixture_num);

	    // to the posterior 
	    const real_t *postP  = meanDis + timeStep * mixture_num + mixtureI;
	    const real_t *sumPost= meanDis + totalTime * mixture_num + timeStep;

	    // to the output of MDN (mixture weight)
	    int pos = timeStep * mixture_num + mixtureI;
	    const real_t *alpha  = mdnPara + pos;

	    if (patTypes[timeStep] == PATTYPE_NONE){
	    
	    }else{
		// pointer to the memory array to store the gradient
		pos = timeStep * NNOutputSize + startD + mixtureI;
		//*(errors + pos) = (*alpha) - (*postP)/(*sumPost);

		// Note, postP and sumPost is the log likelihood
		// use helpers::safeExp((*postP)-(*sumPost))
		*(errors + pos) = (*alpha) - helpers::safeExp((*postP)-(*sumPost));
	    }
	}

    };


    struct ComputeBPmixtureMeanVariance
    {
	// BP for the mixture mean and variance of mixture unit
	// Note: the gradients from each dimension of data for variance are stored localled.
	//       They are summed up and pushed to the NN output layer in ComputeBPAccumVariance 
	int layerSize;
	int startD;
	int startDOut;
	int layerSizeOut;
	int mixture_num;
	int featureDim;
	
	bool tieVar;

	const char *patTypes;
	const real_t *meanDis; // the mixture lielihookd w_i Phi_i and the sum 
	const real_t *mdnPara; // mean value of the mixture
	const real_t *target;  // target data
	real_t *errors;        // outputerrors (gradient buffer) of preceding layer
                               //  for mean
	real_t *varBuff;       // buffer of variance grandients
                               // 
	int totalTime;         //

	// from 1 to timesteps * numMixture*(featureDim) 
	__host__ __device__ void operator() (const int outputIdx) const
	{
	    
	    const int timeStep = outputIdx / (mixture_num * featureDim);

	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;
	    
	    const int tmp = outputIdx % (mixture_num * featureDim);
	    const int mixtureI = tmp / featureDim;
	    const int featureI = tmp % featureDim;
	    
	    // pointer to the mean gradient buffer
	    int meanshift  = timeStep * layerSize+startD+mixtureI*featureDim + featureI;
	    real_t *errorm = errors + meanshift;

	    // pointer to the variance gradient
            #ifdef ALTER_TIEVAR
	    //int varshift2   = timeStep*layerSize+startD+mixture_num*featureDim;
            #else
	    //int varshift2   = timeStep*layerSize+startD+mixture_num*featureDim + mixtureI;
            #endif
	    // If variance is not tied, the gradients can be directly propagated to
	    // previous layer without BPAccumVariance
	    int varshift   = (timeStep * (mixture_num * featureDim) + 
			      mixtureI * featureDim + featureI);
	    real_t *errorv = (tieVar? 
			      (varBuff + varshift) : 
			      (errors + meanshift + mixture_num * featureDim));

	    
	    // pointer to the target data y
	    const real_t *tardata= target + timeStep * layerSizeOut + startDOut + featureI;

	    // pointer to the mean parameter in MDNUnit
	    meanshift= totalTime * mixture_num + 
		       timeStep * mixture_num * featureDim + 
		       mixtureI * featureDim + 
		       featureI;

	    // pointer to the variance parameter in MDNUnit
            #ifdef ALTER_TIEVAR
	    varshift = totalTime*mixture_num*(1+featureDim) + 
		       timeStep*mixture_num;
	    
            #else
	    varshift = (totalTime*mixture_num*(1+featureDim) + 
			(tieVar? 
			 (timeStep*mixture_num + mixtureI) : 
			 (featureDim*(timeStep*mixture_num+mixtureI) + featureI)));
            #endif

	    const real_t *mean  = mdnPara + meanshift;
	    const real_t *var   = mdnPara + varshift;

	    
	    // pointer to the posterior P and sum of posterior P
	    const real_t *postP = meanDis + timeStep * mixture_num + mixtureI;
	    const real_t *sumPost=meanDis + totalTime* mixture_num + timeStep;
	    real_t posterior = helpers::safeExp((*postP) - (*sumPost));

	    // calculate and store the gradient of mean
	    (*errorm)  = posterior*(*mean - *tardata)/(*var)/(*var);
	    // (*errorv2) += posterior*featureDim - (*errorm)*(*mean - *tardata);

	    // calculate and store the gradient of variance
	    // STUPID MISTAKE
	    // (*errorv)  = posterior*featureDim - (*errorm)*(*mean - *tardata);
	    // How could I multiply featureDim here ! For each dimension, it should be 1
	    (*errorv)  = posterior - (*errorm)*(*mean - *tardata);
	    
	}

    };
    
    struct ComputeBPAccumVariance
    {
	// Accumulate the gradidents from each dimension of data
	// and push the sum to the NN output layer
	int layerSize;
	int startD;
	int mixture_num;
	int featureDim;

	const char *patTypes;
	real_t *errors;        // outputerrors of preceding layer
	real_t *varBuff;

	// from 1 to timesteps * numMixture*(featureDim) 
	__host__ __device__ void operator() (const int timeStep) const
	{
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;
	    
	    // start of the variance part of one frame
	    int varshiftTar  = timeStep*layerSize+startD+mixture_num*featureDim;
	    int varshiftSrc  = timeStep*(mixture_num*featureDim);
	    real_t temp = 0.0;

            #ifdef ALTER_TIEVAR
	    for (int i = 0; i < mixture_num; i++){
		for (int j = 0; j < featureDim; j++){
		    temp += *(varBuff + varshiftSrc + i*featureDim + j);
		}
	    }
	    *(errors+varshiftTar) = temp;
            #else
	    for (int i = 0; i < mixture_num; i++){
		for (int j = 0; j < featureDim; j++){
		    temp += *(varBuff + varshiftSrc + i*featureDim + j);
		}
		*(errors+varshiftTar+i) = temp;
		temp = 0.0;
	    }
            #endif
	    
	}

    };
    
    struct ComputeBPAccumForMixtureDynSqr
    {
	int layerSize;
	int startD;
	int layerSizeOut;
	int startDOut;
	int mixtureNum;
	int featureDim;
	int trainableAPos;  // start dimension for the trainable A 
	int trainableBPos;  // start dimension for the trainable B
	int backOrder;
	int totalTime;
	int tanh;
	const char *patTypes;
	real_t *errors;
	real_t *outputs;
	real_t *targets;
	real_t *paraVec;
	
	__host__ __device__ void operator() (const int idx) const
	{
	    	    
	    int timeStep  = idx  / (featureDim * backOrder);
	    int temp      = idx  % (featureDim * backOrder);
	    int featIndex = temp % featureDim;
	    int backStep  = temp / featureDim + 1;

	    int timePos   = timeStep * layerSize + startD;
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;

	    // pointer to the target data
	    int pos_data1  = (timeStep>0)?((timeStep - 1)*layerSizeOut+startDOut+featIndex):-1; 
	    int pos_data2  = (timeStep>1)?((timeStep - 2)*layerSizeOut+startDOut+featIndex):-1; 
	    int pos_datan  = (timeStep>=backStep)?
		((timeStep - backStep)*layerSizeOut+startDOut+featIndex):-1;
	    
	    // pointer to gradients of mean 
	    int pos_mean_g = timePos + mixtureNum    + featIndex; 
	    int pos_a_g    = timePos + trainableAPos + featIndex + (backStep - 1) * featureDim; 
	    int pos_b_g    = timePos + trainableBPos + featIndex; 
	    
	    /* FATAL ERROR 
	       pos_a1 and pos_a2 points to the tanh(a1) + tanh(a2) and tanh(a1) * tanh(a2)
	       it should point to the tanh(a1), tanh(a2)
	     */
	    //int pos_a1     = (totalTime * trainableAPos + timeStep * backOrder * featureDim + 
	    // featIndex);
	    // int pos_a2     = (backOrder == 2)?
	    // ((totalTime * trainableAPos + timeStep * backOrder * featureDim + 
	    // featureDim + featIndex)):-1;
	    int pos_a1     = timePos + trainableAPos + featIndex;
	    int pos_a2     = (backOrder == 2)?(pos_a1 + featureDim):-1;
	    
	    // set the gradients for the first time step as zero
	    *(errors + pos_a_g) = 0;
	    if (backStep == 1){
		*(errors + pos_b_g) = 0;
	    }

	    for (int i = 0; i < mixtureNum; i++){
		real_t tempResult;

		// for the B part
		if (backStep == 1){
		    *(errors + pos_b_g) += ((*(errors  + pos_mean_g + i * featureDim)));
		}
		
		// for the A part
		if (tanh){
		    // I can't allocate additional memory space to store tanh(a1) and tanh(a2)
		    real_t tanha1 = activation_functions::Tanh::fn(*(outputs + pos_a1));
		    real_t tanha2 = ((pos_a2 > 0)? 
				     (activation_functions::Tanh::fn(*(outputs + pos_a2))):0);
		    if (backStep==1){
			// backStep ==1
			tempResult  = (pos_data1 >= 0)?(*(targets + pos_data1)):0;
			tempResult -= (pos_data2 >= 0)?((*(targets + pos_data2)) * tanha2):0;
			tempResult *= (1 - tanha1 * tanha1);
		    }else{
			// backStep ==2
			tempResult  = (pos_data1 >= 0)?(*(targets + pos_data1)):0;
			tempResult -= (pos_data2 >= 0)?((*(targets + pos_data2)) * tanha1):0;
			tempResult *= (1 - tanha2 * tanha2);
		    }
		}else{
		    tempResult  = (pos_datan >= 0)?(*(targets + pos_datan)):0;
		}
		*(errors + pos_a_g) += ((*(errors  + pos_mean_g + i * featureDim)) * 
					tempResult);
	    }
	    
	}
	
    };
	    
    /******************************************
     generation from the MDNUnits
     ******************************************/

    // Definition for the operation involved
    struct SamplingSigmoid
    {
	int NNTargetSize;
	int startDTarget;
	int endDTarget;
	const char *patTypes;
	real_t *NNTarget;

	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    int outputIdx  = t.get<1>();
	    real_t prob    = t.get<0>();

	    // layer size
	    // The data is not continuous, thus, we have to 
	    const int timeStep = outputIdx / (endDTarget - startDTarget);
	    const int dimStep  = (outputIdx % (endDTarget - startDTarget)) + startDTarget;
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return ;	    
	    
	    // output the probability
	    real_t *data = NNTarget + (NNTargetSize * timeStep ) + dimStep;
	    (*data) = prob;
	}
    };

    struct SamplingSoftmax
    {
        int paradim;
	int layerSizeOut;
	int startDOut;
	real_t *output;       // targets data
	const real_t *prob;   // mdn parameter (softmax)

	// from 1 to timesteps
        __host__ __device__ void operator() (const int outputIdx) const
        {

	    real_t *targetClass = (output + (outputIdx*layerSizeOut+startDOut));
	    
	    real_t temp = 0.0;	    
	    int pos = 0;
	    for (int i = 0; i<paradim; i++){
		pos = outputIdx * paradim + i;
		if (prob[pos]>temp){
		    temp = prob[pos];
		    *targetClass = (real_t)i;
		}
	    }
        }
    };
    
    struct SamplingMixture
    {
	int featureDim;
	int layerSizeOut;
	int startDOut;
	int mixtureNum;
	int totalTime;
	bool tieVar;
	real_t para;
	real_t *paraPtr;
	real_t *targets;
	real_t *mdnPara;
	
	// from timesteps * featureDim
	__host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
	{
	    real_t seed = t.get<0>();
	    int idx = t.get<1>();
	    
	    int timeStep = idx / (featureDim);
	    int dimStep  = idx % (featureDim);

	    const real_t *mixture = mdnPara + timeStep*mixtureNum;
	    real_t tmp = 0.0;
	    int flag = 0;
	    for (int i = 0; i<mixtureNum; i++){
		if ((*(mixture+i)) > tmp){
		    tmp = (*(mixture+i));
		    flag = i;
		}
	    }
	    int pos = totalTime *mixtureNum + timeStep * mixtureNum * featureDim;
	    const real_t *mean = mdnPara + pos + flag*featureDim + dimStep;
	    
	        pos = totalTime *(mixtureNum + mixtureNum * featureDim);

            #ifdef ALTER_TIEVAR
	    const real_t *var = mdnPara + pos + timeStep*mixtureNum;
            #else
	    const real_t *var = mdnPara + pos + 
		(tieVar ? (flag+timeStep*mixtureNum):
		 ((timeStep * mixtureNum + flag) * featureDim + dimStep));
            #endif	    

	    pos = timeStep * layerSizeOut + startDOut + dimStep;
	    // To support variance scale for each dimension
	    // *(targets+pos) = (*var)*para*seed + (*mean);
	    if (paraPtr == NULL)
		*(targets+pos) = (*var)*para*seed + (*mean);
	    else
		*(targets+pos) = (*var) * (*(paraPtr+startDOut+dimStep)) * seed + (*mean);
	}
    };

    struct GetParameterMixture
    {
	// 'Generating' (directly output) the parameter of Mixture model
	int featureDim;
	int NNOutputSize;
	int mixtureNum;
	int totalTime;
	int startDimIn;
	real_t *targets;
	real_t *mdnPara;
	const char *patTypes;
	bool tieVar;

	// from timesteps * featureDim
	__host__ __device__ void operator() (const int timeStep) const
	{
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;
	    
	    // pointer to the weight
	    int pos_weight = timeStep*mixtureNum;
	    int pos_mean   = totalTime*mixtureNum + timeStep*mixtureNum*featureDim;
	    int pos_var    = (totalTime*(mixtureNum + mixtureNum*featureDim) + 
			      timeStep * mixtureNum * (tieVar?1:featureDim));
	    int pos_output = timeStep*NNOutputSize;
	    
	    int bias = startDimIn;
	    for (int i = 0; i<mixtureNum; i++, bias++){
		*(targets + pos_output + bias) = *(mdnPara + pos_weight + i);
	    }
	    for (int i = 0; i<mixtureNum*featureDim; i++, bias++){
		*(targets + pos_output + bias) = *(mdnPara + pos_mean + i);
	    }
	    for (int i = 0; i<(mixtureNum*(tieVar?1:featureDim)); i++, bias++){
		*(targets + pos_output + bias) = *(mdnPara + pos_var + i);
	    }	    
	}
    };

    struct GetParameterMixtureDynSqr
    {
	// 'Generating' (directly output) the parameter of Mixture model
	int featureDim;
	int NNOutputSize;
	int mixtureNum;
	int totalTime;
	int startDimIn;
	int trainableAPos;
	int trainableBPos;
	real_t *targets;
	real_t *mdnPara;
	const char *patTypes;
	bool tieVar;

	// from timesteps * featureDim
	__host__ __device__ void operator() (const int timeStep) const
	{
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;
	    
	    // pointer to the weight
	    int pos_weight = timeStep*mixtureNum;
	    int pos_mean   = totalTime*mixtureNum + timeStep*mixtureNum*featureDim;
	    int pos_var    = (totalTime*(mixtureNum + mixtureNum*featureDim) + 
			      timeStep * mixtureNum * (tieVar?1:featureDim));
	    int pos_a      = totalTime * trainableAPos + timeStep * featureDim;
	    int pos_b      = totalTime * trainableBPos + timeStep * featureDim;
	    int pos_output = timeStep*NNOutputSize;
	    
	    int bias = startDimIn;
	    for (int i = 0; i<mixtureNum; i++, bias++){
		*(targets + pos_output + bias) = *(mdnPara + pos_weight + i);
	    }
	    for (int i = 0; i<mixtureNum*featureDim; i++, bias++){
		*(targets + pos_output + bias) = *(mdnPara + pos_mean + i);
	    }
	    for (int i = 0; i<(mixtureNum*(tieVar?1:featureDim)); i++, bias++){
		*(targets + pos_output + bias) = *(mdnPara + pos_var + i);
	    }	    
	    for (int i = 0; i<featureDim; i++, bias++){
		*(targets + pos_output + bias) = *(mdnPara + pos_a + i);
	    }
	    for (int i = 0; i<featureDim; i++, bias++){
		*(targets + pos_output + bias) = *(mdnPara + pos_b + i);
	    }
	}
    };


    struct copyMixtureWeightforEMGen
    {	
	// To initialize the EM-style generation
	//  I use the predicted weight. Thus, fake w_i as w_i Phi_i.
	//  Then, the EM will start with w_i / sum_k w_k as the posterior
	//  for the i-th mixture
	int         mixture_num;
	const char *patTypes;
	real_t     *meanDis; 
	real_t     *mdnPara;      // mean value of the mixture
	int         totalTime;    // current max time * number of parallel sentence

	// from 1 to timesteps
	__host__ __device__ void operator() (const int timeStep) const
	{
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;
	    
	    real_t tmp = helpers::NumericLimits<real_t>::logZero();
	    real_t tmptmp;
	    real_t *meanDisPtr = meanDis + timeStep * mixture_num;
	    
	    int pos = timeStep*mixture_num;
	    const real_t *mixtureW = mdnPara + pos; // point to the weight
	    	    
	    for (int i = 0; i<mixture_num; i++){
		tmptmp = helpers::safeLog(*(mixtureW+i));
		tmp    = helpers::logAdd(tmp, tmptmp);
		// save  w_i
		*(meanDisPtr+i) = tmptmp; 
	    }
	    // save sum_i^mixture_num w_i
	    meanDisPtr  = meanDis + totalTime * mixture_num + timeStep;
	    *meanDisPtr = tmp;
	    //return -1*helpers::safeLog(tmp);
	}

    };
    
    // EM iteration
    struct initIterEMGen
    {
	// 
	int featureDim;
	int mixtureNM;
	int totalTime;
	int outputSize;
	int startDOut;
	bool tieVar;

	real_t *postP;
	real_t *mdnPara;
	real_t *targets;
	const char *patTypes;
	
	__host__ __device__ void operator() (const int idx) const
	{
	    int timeStep = idx / (featureDim);
	    int dimOutput= idx % (featureDim);

	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;
	    const real_t *m, *v, *p, *q;
	    /*const real_t *w;
	      int pos_w    = timeStep*mixtureNM;*/

	    
	    /****  FATAL ERROR: remember to shift by dimOutput ****/
	    // pointer to the Parameter of MDNUnit
	    int pos_mean = totalTime*mixtureNM + timeStep*featureDim*mixtureNM + dimOutput;
	    int pos_var  = tieVar ? 
		(totalTime*(mixtureNM+featureDim*mixtureNM)+timeStep*mixtureNM) :
		(pos_mean + totalTime*mixtureNM*featureDim);
	    int pos_postM = timeStep  * mixtureNM;           
	    int pos_postS = totalTime * mixtureNM + timeStep;

	    real_t tmp1=0.0;
	    real_t tmp2=0.0;
	    

	    // for each mixture
	    for(int i = 0; i < mixtureNM; i++){
		v = mdnPara + pos_var   + (i*(tieVar?1:featureDim));  // pointer to var
		m = mdnPara + pos_mean  + (i*featureDim);             // pointer to mean
		p = postP   + pos_postM + i;  // pointer to the likelihood of this mixture
		q = postP   + pos_postS;      // pointer to the sum of mixture likelihood
		
		// poster/var^2
		tmp2 += helpers::safeExp((*p)-(*q))/((*v)*(*v));
		// poster*mean/var^2
		tmp1 += ((*m)*(helpers::safeExp((*p)-(*q))))/((*v)*(*v));
	    }
	    
	    //
	    // sum_i (poster_i * mean_i /var_i) / sum_i (poster_i /var_i) 
	    tmp1 = tmp1/tmp2;
	    // tmp1 = mdnPara[pos_mean+widx*featureDim];
	    int pos_tar = timeStep * outputSize + startDOut + dimOutput;
	    *(targets+pos_tar) = tmp1;
	}
    };


    
    /********************************************************
     other utilizes
    *******************************************************/

    real_t safeLog(real_t x)
    {
	if (x < 1.1754944e-038f)
	    return -1e30f;
	else
	    return std::log(x);
    }

    #ifdef DEBUG_LOCAL
    real_t safeExp(real_t x)
    {
	if (x <= -1e30f){
	    return 0;
	}else if(x >= 88.722839)
	    return 3.4028235e+038f;
	else
	    return std::exp(x);
    }

    real_t logAdd(real_t x, real_t y)
    {
	real_t minLogExp = -69.0776;
	real_t lSMALL    = -0.5e10;
	real_t logZero   = -1e30;
	if (x>y){
	    if ((y-x) < minLogExp){
		if (x < lSMALL)
		    return logZero;
		else
		    return x;
	    }else{
		return x + std::log(1.0 + std::exp(y-x));
	    }
	}
	else{
	    if ((x-y) < minLogExp)
		{
		    if (y < lSMALL)
			return logZero;
		    else
			return y;
		}
	    else
		{
		    return y + std::log(1.0 + std::exp(x-y));
		}
	    }
    }
    #endif    
    
} // namespace anonymous
} // namespace internal


/********************************************************
 definition of the MDN Units

  for simplicity, let's define the output from previous output layer as a_i_*, i is the dimension
  a_i_g: input to the MDNUnit_sigmoid
  a_i_s: ... to MDNUnit_softmax
  a_i_mk: ... to MDNUnit_mixture, the mixture weight
  a_ij_mu: ... to MDNUnit_mixture, the mixture mean (i-th mixture, j-th dimension)

  a_i_ms:  ... to MDNUnit_mixture, the mixture variance (i-th mixture, shared by all dimension)  
or 
  a_ij_ms:  ... to MDNUnit_mixture, the mixture variance (i-th mixture, j-th dimension)  
 *******************************************************/


namespace layers {
    
    // This function defines the number of parameters required for Mixture_dyn unit
    int MixtureDynWeightNum(int featureDim, int mixNum, int backOrder, int dynDirection){
	#ifdef MIXTUREDYNDIAGONAL
	// Diagonal matrix shared by the single mixture
	if (dynDirection == MDNUNIT_TYPE_1_DIRECT){
	    return (featureDim * backOrder + featureDim);
	}else if (dynDirection == MDNUNIT_TYPE_1_DIRECD){
	    return (backOrder + 1);
	}else{
	    return (backOrder + 1) * (featureDim + 1);
	}

	#else
	// Full transformation matrices for different mixtures
	return featureDim * (featureDim + 1)  * mixNum;
	#endif
    }		    
    
    /************************************************
     * MDNUnit Definition
     ************************************************/
    // virtual class of MDNUnit
    template <typename TDevice>
    MDNUnit<TDevice>::MDNUnit(int startDim, int endDim, int startDimOut, int endDimOut, 
			      int type, int paraDim, Layer<TDevice> &precedingLayer,
			      int outputSize, const int trainable)
	: m_startDim        (startDim)
	, m_endDim          (endDim)
	, m_type            (type)
	, m_paraDim         (paraDim)
	, m_precedingLayer  (precedingLayer)
	, m_startDimOut     (startDimOut)
	, m_endDimOut       (endDimOut)
	, m_layerSizeIn     (precedingLayer.size())
	, m_layerSizeTar    (outputSize)
	, m_trainable       (trainable)
    {
	// initilize the parameter vec
	int n = m_precedingLayer.patTypes().size();
	m_paraVec.resize(m_paraDim*n, 0.0);
	
	/*
	 here, I assume the each dimension of the same kind of parameter 
	 should be adjacent to each other.
	 
	 for mixture Unit, 
	 [mixture_weight_1_time_1, mixture_weight_2_time_1, ... mixture_weight_k_time_1,
	  mixture_weight_1_time_2, mixture_weight_2_time_2, ... mixture_weight_k_time_2,
	  ...
	  mixture_weight_1_time_N, mixture_weight_2_time_N, ... mixture_weight_k_time_N,
	  mixture_mean_1_1_time_1, mixture_mean_1_2_time_1, ... mixture_mean_1_D_time_1,
	  ...]
	*/

	// Add 0621: To read in the MDN mixture variance scale vector
	// for generation 
	const Configuration &config = Configuration::instance();        
	if (config.mdnVarScaleGen().size() > 0){
	    std::ifstream ifs(config.mdnVarScaleGen().c_str(), 
			      std::ifstream::binary | std::ifstream::in);
	    if (!ifs.good())
		throw std::runtime_error(std::string("Fail to open "+config.mdnVarScaleGen()));
	   
	    std::streampos numEleS, numEleE;
	    numEleS = ifs.tellg();
	    ifs.seekg(0, std::ios::end);
	    numEleE = ifs.tellg();
	    long int tmpnumEle  = (numEleE-numEleS)/sizeof(real_t);
	    ifs.seekg(0, std::ios::beg);
            
	    if (tmpnumEle == this->m_layerSizeTar){             
		real_t tempVal;
		cpu_real_vector tempVec;
		tempVec.resize(tmpnumEle, 0.0);
		for (int i = 0; i < tmpnumEle; i++){
		    ifs.read((char *)&tempVal, sizeof(real_t)); 
		    tempVec[i] = tempVal;
		}
		m_varScale = tempVec;
		ifs.close();
	    }else{
		ifs.close();
		printf("Dimension mismatch: %d (vector) VS %d (tartget feature)", 
		       (int)tmpnumEle, (int)this->m_layerSizeTar);
		throw std::runtime_error("Dimension unmatch");
	    }
            
	}else{
	    m_varScale.clear();
	}

    }

    template <typename TDevice>
    MDNUnit<TDevice>::~MDNUnit()
    {
    }

    template <typename TDevice>
    const int& MDNUnit<TDevice>::paraDim() const
    {
	return m_paraDim;
    }

    template <typename TDevice>
    const MDNUnit<TDevice>::real_vector& MDNUnit<TDevice>::varScale() const
    {
	return m_varScale;
    }
    
    template <typename TDevice>
    int MDNUnit<TDevice>::flagTrainable() const
    {
	return m_trainable;
    }

    template <typename TDevice>
    bool MDNUnit<TDevice>::flagVariance() const
    {
	return false; // default, false for all units
    }

    template <typename TDevice>
    int MDNUnit<TDevice>::tanhRegType() const
    {
	return 0; // default, false for all units
    }
    
    template <typename TDevice>
    void MDNUnit<TDevice>::linkWeight(real_vector& weights, real_vector& weightsUpdate)
    {	
	// default, untrainable units doesn't require trainable weight
    }

    // set the current training epoch number
    template <typename TDevice>
    void MDNUnit<TDevice>::setCurrTrainingEpoch(const int currTrainingEpoch)
    {
	m_currTrainingEpoch = currTrainingEpoch;
    }
    
    template <typename TDevice>
    int & MDNUnit<TDevice>::getCurrTrainingEpoch()
    {
	return m_currTrainingEpoch;
    }


    /********************************************************
     MDNUnit_sigmoid Definition
    *******************************************************/
    template <typename TDevice>
    MDNUnit_sigmoid<TDevice>::MDNUnit_sigmoid(
	int startDim, int endDim, int startDimOut, int endDimOut, int type, 
	Layer<TDevice> &precedingLayer, int outputSize, const int trainable)
	: MDNUnit<TDevice>(startDim, endDim, startDimOut, endDimOut, 
			   type, (endDim - startDim), precedingLayer,
			   outputSize, trainable)
    {
	// nothing else to be initialized
    }
    
    template <typename TDevice>
    bool MDNUnit_sigmoid<TDevice>::flagValid()
    {
	// just check something
	bool tempFlag = ((this->m_startDim > 0) &&
			 (this->m_endDim   < this->m_layerSizeIn) && 
			 (!this->m_trainable));
	if (!tempFlag){
	    printf("\tSigmoid unit\t");
	}
	return tempFlag;
    }
    
    template <typename TDevice>
    MDNUnit_sigmoid<TDevice>::~MDNUnit_sigmoid()
    {
    }

    template <typename TDevice>
    void MDNUnit_sigmoid<TDevice>::initPreOutput(
		const MDNUnit_sigmoid<TDevice>::cpu_real_vector &mVec, 
		const MDNUnit_sigmoid<TDevice>::cpu_real_vector &vVec)
    {	
	Layer<TDevice> *tmpPtr = &this->m_precedingLayer;
	TrainableLayer<TDevice> *tLayer = dynamic_cast<layers::TrainableLayer<TDevice>*>(tmpPtr);
	if (tLayer){
	    int tmpSize = tLayer->size() * (1 + tLayer->precedingLayer().size());
	    if (tmpSize != tLayer->weights().size()){
		printf("The layer before MDN is not feedforward. No method to initialize it\n");
	    }else{
		// set w to zero, set b to mean+variance
		// w starts at precedingSize * startDim
		thrust::fill(tLayer->weights().begin()+
			     this->m_startDim*tLayer->precedingLayer().size(),
			     tLayer->weights().begin()+
			     this->m_startDim*tLayer->precedingLayer().size()+
			     (this->m_endDim - this->m_startDim) * tLayer->precedingLayer().size(),
			     (real_t)0.0);
		
		// 
		Cpu::real_vector biasInit;
		biasInit.reserve((this->m_endDim - this->m_startDim));
		const Configuration &config = Configuration::instance();
		static boost::mt19937 *gen = NULL;
		if (!gen) {
		    gen = new boost::mt19937;
		    gen->seed(config.randomSeed());
		}
		boost::random::normal_distribution<real_t> dist(0, 1);
		for (int i =0; i<(this->m_endDim - this->m_startDim); i++)
		    biasInit.push_back(dist(*gen));
		 
		// set b
		thrust::copy(biasInit.begin(),
			     biasInit.end(),
			     tLayer->weights().begin() +
			     tLayer->size()* tLayer->precedingLayer().size() + 
			     this->m_startDim);
		
	    }
	}else{
	    throw std::runtime_error("MDN previous layer can not be untrainable");
	}
    }

    template <typename TDevice>
    void MDNUnit_sigmoid<TDevice>::computeForward()
    {
	
	// sigmoid, o_i_g = sigmoid(a_i_g), where a_i_g is the output of the previous hidden layer
	{{
		internal::ComputeSigmoid fn;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.startD       = this->m_startDim;
		fn.endD         = this->m_endDim;
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNoutput     = helpers::getRawPointer(this->m_precedingLayer.outputs());
		
		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		n = n*(this->m_paraDim);
		
		thrust::transform(
		   thrust::counting_iterator<int>(0),
		   thrust::counting_iterator<int>(0)+n,
		   this->m_paraVec.begin(),
		   fn);
		#ifdef DEBUG_LOCAL
		Cpu::real_vector tmp1 = this->m_paraVec;
		Cpu::real_vector tmp2 = this->m_precedingLayer.outputs();
		for (int i = 0; i<n; i++){
		    int timestep = i / (fn.endD-fn.startD);
		    int dimstep =  i % (fn.endD-fn.startD) + fn.startD;
		    //printf("SigForward: %f %f\n", tmp1[i], 
		    //tmp2[timestep * fn.NNOutputSize + dimstep]);
		}
		#endif

	}}	    
	
    }

    template <typename TDevice>
    void MDNUnit_sigmoid<TDevice>::getEMOutput(const real_t para, real_vector &targets)
    {	
	// no EM (no mixture at all), just sampling 
	this->getOutput(para, (targets));
    }
	
    template <typename TDevice>
    void MDNUnit_sigmoid<TDevice>::getOutput(const real_t para, real_vector &targets)
    {
	// Here, probability p(1) is directly used as output
	// sampling output
	{{
		internal::SamplingSigmoid fn;
		fn.NNTargetSize = this->m_layerSizeTar;
		fn.startDTarget = this->m_startDimOut;
		fn.endDTarget   = this->m_endDimOut;
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNTarget     = helpers::getRawPointer(targets); 

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		n = n*this->m_paraDim;
		thrust::for_each(
			 thrust::make_zip_iterator(
			     thrust::make_tuple(this->m_paraVec.begin(), 
						thrust::counting_iterator<int>(0))),
		         thrust::make_zip_iterator(
			     thrust::make_tuple(this->m_paraVec.begin()+n, 
						thrust::counting_iterator<int>(0)+n)),
			 fn);
	}}
    }

    template <typename TDevice>
    void MDNUnit_sigmoid<TDevice>::getParameter(real_t *targets)
    {
	// STUPID MISTAKE
	/*{{
		internal::CopySimple fn;
		...
	}}*/
	{{
		// the same function for getOutput
		// only the position of output in the output vector
		internal::SamplingSigmoid fn;
		fn.NNTargetSize = this->m_precedingLayer.size();
		fn.startDTarget = this->m_startDim;  // position in the output parameter vector
		fn.endDTarget   = this->m_endDim;    // position in the output parameter vector
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNTarget     = targets; 

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		n = n*this->m_paraDim;
		thrust::for_each(
			 thrust::make_zip_iterator(
			     thrust::make_tuple(this->m_paraVec.begin(), 
						thrust::counting_iterator<int>(0))),
		         thrust::make_zip_iterator(
			     thrust::make_tuple(this->m_paraVec.begin()+n, 
						thrust::counting_iterator<int>(0)+n)),
			 fn);
	}}
    }
	
    template <typename TDevice>
    real_t MDNUnit_sigmoid<TDevice>::calculateError(real_vector &targets)
    {
	// - sum_n_1_N sum_m_1_M ( (output_n_m>0) * log p(1|x) + (output_n_m<0) * log (1-p(1|x))
	// where N is the timestep, M is the dimension
	real_t tmp = 0.0;
	{{
		internal::ComputeSigmoidError fn;
		fn.startD    = this->m_startDim;
		fn.endD      = this->m_endDim;
		fn.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.targets   = helpers::getRawPointer(targets);
		fn.layerSizeOut = this->m_layerSizeTar;
		fn.startDOut = this->m_startDimOut;
		
		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		n = n*this->m_paraDim; // = this->m_endDim - this->m_startDim
		
		tmp = thrust::transform_reduce(
		         thrust::make_zip_iterator(
			     thrust::make_tuple(this->m_paraVec.begin(), 
						thrust::counting_iterator<int>(0))),
		         thrust::make_zip_iterator(
			     thrust::make_tuple(this->m_paraVec.begin()+n, 
						thrust::counting_iterator<int>(0)+n)),
			 fn,
			 (real_t)0,
			 thrust::plus<real_t>()
		      );
		
		#ifdef DEBUG_LOCAL
		Cpu::real_vector tmp1 = targets;
		Cpu::real_vector tmp2 = this->m_paraVec;
		int PosTar;
		real_t data, target;
		real_t prop = 0.0;
		for (int i = 0; i<(n/this->m_paraDim); i++){
		    for (int j = 0; j < this->m_paraDim; j++){
			data = tmp2[i*this->m_paraDim + j];
			target= tmp1[i*fn.layerSizeOut + fn.startDOut + j];
			if (target>0)
			    prop += -1*internal::safeLog(data);
			else
			    prop += -1*internal::safeLog(1-data);
		    }
		}
		printf("Prob: %f\t", prop);
		#endif

	}}
	return tmp;
    }

    template <typename TDevice>
    void MDNUnit_sigmoid<TDevice>::computeBackward(real_vector &targets)
    {
	// calculate the gradient w.r.t the input to this sigmoid unit
	// pay attention to the assumption:
	// 1. the output of sigmoid unit \in (0,1)
	// 2. the training data of sigmoid \in {-a,a}, due to the normalization on
	//    the target feature
	{{
		internal::ComputeSigmoidBP fn;
		fn.startD       = this->m_startDim;
		fn.endD         = this->m_endDim;
		fn.layerSizeOut = this->m_layerSizeTar;
		fn.startDOut    = this->m_startDimOut;
		fn.layerSizeIn  = this->m_precedingLayer.size();

		fn.errors    = helpers::getRawPointer(this->m_precedingLayer.outputErrors());
		fn.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.targets   = helpers::getRawPointer(targets);

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		n = n*this->m_paraDim;
		
		thrust::for_each(
		         thrust::make_zip_iterator(
			     thrust::make_tuple(this->m_paraVec.begin(), 
						thrust::counting_iterator<int>(0))),
		         thrust::make_zip_iterator(
			     thrust::make_tuple(this->m_paraVec.begin()+n, 
						thrust::counting_iterator<int>(0)+n)),
			 fn);

		#ifdef DEBUG_LOCAL
		Cpu::real_vector tmp1 = targets;
		Cpu::real_vector tmp2 = this->m_paraVec;
		Cpu::real_vector tmp3 = this->m_precedingLayer.outputErrors();

		real_t data, target;
		real_t prop = 0.0;
		for (int i = 0; i<(n/this->m_paraDim); i++){
		    for (int j = 0; j < this->m_paraDim; j++){
			data = tmp2[i*this->m_paraDim + j];
			target= tmp1[i*fn.layerSizeOut + fn.startDOut + j];
			if (target>0)
			    prop = -1+data;
			else
			    prop = 1*data;
			
			printf("Back sigmoid: %f\t", prop);
		    }
		    printf("\n");
		}
		#endif
		
		
	}}
		
	
    }

    /********************************************************
     MDNUnit_softmax
    *******************************************************/
    template <typename TDevice>
    MDNUnit_softmax<TDevice>::MDNUnit_softmax(
	int startDim, int endDim, int startDimOut, int endDimOut, int type, 
	Layer<TDevice> &precedingLayer, int outputSize, const int trainable)
        : MDNUnit<TDevice>(startDim, endDim, startDimOut, endDimOut, 
			   type, endDim-startDim, precedingLayer,
			   outputSize, trainable)
    {   
	// special strategy for vec is unecessary
	m_offset.resize(this->m_precedingLayer.patTypes().size(), 0.0);
	
	// assume ont softmax unit only corresponds to one dimension of the output
	
	if ((endDimOut - startDimOut) != 1){
	    throw std::runtime_error("Check MDN configure. SoftMax => one dimensional target");
	}
	
	//throw std::runtime_error("WARNING: code on softmax is incomplete\n");
	
    }

    template <typename TDevice>
    bool MDNUnit_softmax<TDevice>::flagValid()
    {
	// just check something
	bool tempFlag = ((this->m_startDim > 0) &&
			 (this->m_endDim   < this->m_layerSizeIn) && 
			 (!this->m_trainable));
	if (!tempFlag){
	    printf("\tSoftmax unit\t");
	}
	return tempFlag;
    }

    
    template <typename TDevice>
    MDNUnit_softmax<TDevice>::~MDNUnit_softmax()
    {
    }

    template <typename TDevice>
    void MDNUnit_softmax<TDevice>::initPreOutput(
	const MDNUnit_softmax<TDevice>::cpu_real_vector &mVec, 
	const MDNUnit_softmax<TDevice>::cpu_real_vector &vVec)
    {
	// no need on mVec and vVec
	Layer<TDevice> *tmpPtr = &this->m_precedingLayer;
	TrainableLayer<TDevice> *tLayer = dynamic_cast<layers::TrainableLayer<TDevice>*>(tmpPtr);
	if (tLayer){
	    int tmpSize = tLayer->size() * (1 + tLayer->precedingLayer().size());
	    if (tmpSize != tLayer->weights().size()){
		printf("The layer before MDN is not feedforward. No method to initialize it\n");
	    }else{
		// set w to zero, set b to mean+variance
		// w starts at precedingSize * startDim
		thrust::fill(tLayer->weights().begin()+
			     this->m_startDim*tLayer->precedingLayer().size(),
			     tLayer->weights().begin()+
			     this->m_startDim*tLayer->precedingLayer().size()+
			     (this->m_endDim - this->m_startDim) * tLayer->precedingLayer().size(),
			     (real_t)0.0);
		
		// 
		Cpu::real_vector biasInit;
		biasInit.reserve((this->m_endDim - this->m_startDim));
		const Configuration &config = Configuration::instance();
		static boost::mt19937 *gen = NULL;
		if (!gen) {
		    gen = new boost::mt19937;
		    gen->seed(config.randomSeed());
		}
		boost::random::normal_distribution<real_t> dist(0, 1);
		for (int i =0; i<(this->m_endDim - this->m_startDim); i++)
		    biasInit.push_back(dist(*gen));
		 
		// set b
		thrust::copy(biasInit.begin(),
			     biasInit.end(),
			     tLayer->weights().begin() +
			     tLayer->size()* tLayer->precedingLayer().size() + 
			     this->m_startDim);
		
	    }
	}else{
	    throw std::runtime_error("MDN previous layer can not be untrainable");
	}
    }


    template <typename TDevice>
    void MDNUnit_softmax<TDevice>::computeForward()
    {
	// calculate the offset 
	{{
		internal::CalculateOffsetFn fn;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.startD       = this->m_startDim;
		fn.endD         = this->m_endDim;
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNoutputs    = helpers::getRawPointer(this->m_precedingLayer.outputs());

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		//n = n*(this->m_paraDim);
		
		thrust::transform(
		   thrust::counting_iterator<int>(0),
		   thrust::counting_iterator<int>(0)+n,
		   this->m_offset.begin(),
		   fn);
	}}	    

	// calculate the Exp
	{{
		internal::CalculateExpFn fn;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.startD    = this->m_startDim;
		fn.endD      = this->m_endDim;
		fn.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNOutput    = helpers::getRawPointer(this->m_precedingLayer.outputs());
		fn.offset    = helpers::getRawPointer(this->m_offset);
		
		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		n = n*(this->m_paraDim);
		
		thrust::transform(
		   thrust::counting_iterator<int>(0),
		   thrust::counting_iterator<int>(0)+n,
		   this->m_paraVec.begin(),
		   fn);
	}}

	// sum up
	{{
		internal::SumUpOutputsFn fn;
		fn.dimSize = this->m_paraDim;
		fn.outputs = helpers::getRawPointer(this->m_paraVec);

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		// n = n*(this->m_paraDim);
		
		thrust::for_each(
		   thrust::make_zip_iterator(
				thrust::make_tuple(this->m_offset.begin(),  
						   thrust::counting_iterator<int>(0))),
		   thrust::make_zip_iterator(
				thrust::make_tuple(this->m_offset.begin()+n,  
						   thrust::counting_iterator<int>(0)+n)),
		   fn);
	}}
	
	// normalize
        {{
            internal::NormalizeOutputsFn fn;
            fn.layerSize = this->m_paraDim;
            fn.normFacts = helpers::getRawPointer(this->m_offset);
	    
	    int n =this->m_precedingLayer.curMaxSeqLength();
	    n = n*this->m_precedingLayer.parallelSequences();
	    n = n*this->m_paraDim;

            thrust::for_each(
		thrust::make_zip_iterator(
			 thrust::make_tuple(this->m_paraVec.begin(),
					    thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(
			 thrust::make_tuple(this->m_paraVec.begin()+n, 
					    thrust::counting_iterator<int>(0)+n)),
                fn
                );
        }}
    }

    template <typename TDevice>
    void MDNUnit_softmax<TDevice>::getEMOutput(const real_t para,real_vector &targets)
    {
    }

    template <typename TDevice>
    void MDNUnit_softmax<TDevice>::getOutput(const real_t para,real_vector &targets)
    {
	{{    
	    internal::SamplingSoftmax fn;
	    fn.paradim = this->m_paraDim;
	    fn.startDOut = this->m_startDimOut;
	    fn.output    = helpers::getRawPointer(targets);
	    fn.prob      = helpers::getRawPointer(this->m_paraVec);
	    fn.layerSizeOut = this->m_layerSizeTar;
	    
	    int n = this->m_precedingLayer.curMaxSeqLength();
	    n = n*this->m_precedingLayer.parallelSequences();
		
	    thrust::for_each(
		  thrust::counting_iterator<int>(0),
		  thrust::counting_iterator<int>(0)+n,
		  fn);

	}}
	
    }

    template <typename TDevice>
    void MDNUnit_softmax<TDevice>::getParameter(real_t *targets)
    {
	// copy directly
	{{
		internal::CopySimple fn;
		fn.Output   = targets;
		fn.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		n = n*this->m_paraDim;
		thrust::for_each(
			 thrust::make_zip_iterator(
			     thrust::make_tuple(this->m_paraVec.begin(), 
						thrust::counting_iterator<int>(0))),
		         thrust::make_zip_iterator(
			     thrust::make_tuple(this->m_paraVec.begin()+n, 
						thrust::counting_iterator<int>(0)+n)),
			 fn);
	}}
    }

    template <typename TDevice>
    real_t MDNUnit_softmax<TDevice>::calculateError(real_vector &targets)
    {   
	real_t tmp=0.0;
	{{    
	    internal::ComputeCrossEntropyErrorFn fn;
	    fn.layerSize = this->m_paraDim;
	    fn.startDOut = this->m_startDimOut;
	    fn.output    = helpers::getRawPointer(targets);
	    fn.prob      = helpers::getRawPointer(this->m_paraVec);
	    fn.layerSizeOut = this->m_layerSizeTar;
	    
	    int n = this->m_precedingLayer.curMaxSeqLength();
	    n = n*this->m_precedingLayer.parallelSequences();
	    
	    /*Cpu::real_vector temp1 = targets;
	    Cpu::real_vector temp2 = this->m_paraVec;
	    for (int i = 0; i<n; i++){
		int tarclass = temp1[i * fn.layerSizeOut + fn.startDOut];
		int idx = i * fn.layerSize + tarclass;
		tmp += -1*internal::safeLog(temp2[idx]);
		}*/

	    tmp = thrust::transform_reduce(
		  thrust::counting_iterator<int>(0),
		  thrust::counting_iterator<int>(0)+n,
		  fn,
		  (real_t)0.0,
		  thrust::plus<real_t>()
		  );

	}}
	return tmp;
    }
    template <typename TDevice>
    void MDNUnit_softmax<TDevice>::computeBackward(real_vector &targets)
    {
	// BP for softmax unit, it is similar to the sigmoid function
	// however, the output is a number that has not been normalized (start from 0)
	{{
		internal::ComputeSoftMaxBP fn;
		fn.startD       = this->m_startDim;
		fn.startDOut    = this->m_startDimOut;
		fn.paraDim      = this->m_endDim - this->m_startDim;
		fn.layerSizeOut = this->m_layerSizeTar;
		fn.layerSizeIn  = this->m_precedingLayer.size();

		fn.errors      = helpers::getRawPointer(this->m_precedingLayer.outputErrors());
		fn.patTypes    = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.targetData  = helpers::getRawPointer(targets);
		fn.prob        = helpers::getRawPointer(this->m_paraVec);

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		n = n*this->m_paraDim;
		
		thrust::for_each(
				 thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0)+n,
				 fn);
	    
	}}
    }



    /********************************************************
     MDNUnit_mixture
    *******************************************************/
    template <typename TDevice>
    MDNUnit_mixture<TDevice>::MDNUnit_mixture(
	int startDim, int endDim, int startDimOut, int endDimOut, int type, 
	int featureDim, Layer<TDevice> &precedingLayer, int outputSize,
	const bool tieVar, const int trainable)
        : MDNUnit<TDevice>(startDim, endDim, startDimOut, endDimOut, 
			   type, (endDim-startDim), precedingLayer,
			   outputSize, trainable)
	, m_numMixture    (type) // I forget why I used type to name m_numMixture
	, m_featureDim    (featureDim)
	, m_varFloor      (0.0)
	, m_tieVar        (tieVar)
    {                                                          
	
	// offset for the mixture weight (in the same way as softmax)
	m_offset.resize(this->m_precedingLayer.patTypes().size(), 0.0);	
	
	// intermediate matrix to store the \sum_dim (t-\mu)^2 and sum_mixture \sum_dim (t-\mu)^2
	m_tmpPat.resize(this->m_precedingLayer.patTypes().size()*(m_numMixture+1), 0.0);
	    
	// for BP variance accumulation (only used for tied variance)
	if (m_tieVar)
	    m_varBP.resize(this->m_precedingLayer.patTypes().size()*(endDim-startDim)*type, 0.0);
			 
    }

    template <typename TDevice>
    MDNUnit_mixture<TDevice>::~MDNUnit_mixture()
    {                                                           
    }

    template <typename TDevice>
    bool MDNUnit_mixture<TDevice>::flagValid()
    {
	bool tempFlag (this->m_featureDim == 
		       (this->m_tieVar ? 
			(((this->m_endDim-this->m_startDim)-2*this->m_numMixture)/
			 this->m_numMixture) :
			(((this->m_endDim-this->m_startDim)-this->m_numMixture)/2/
			 this->m_numMixture)));
	tempFlag = tempFlag && (this->m_featureDim == (this->m_endDimOut - this->m_startDimOut));
	if (!tempFlag){
	    printf("\tMixture unit check failed. Please check dimension and tieVariance\t");
	}
	return tempFlag;
    }

    
    // Note: initPreOutput is used when I found it hard to control
    //       the gradients in the initial training stage
    //       However, this initializing below will lead to gradient vanishing because
    //       the W of the last hidden layer is small
    //       Thus, I use the pre-trained network from MSE criterion as the initial 
    //       weight and then train the network. 
    // More to be done here
    template <typename TDevice>
    void MDNUnit_mixture<TDevice>::initPreOutput(
		const MDNUnit_mixture<TDevice>::cpu_real_vector &mVec, 
		const MDNUnit_mixture<TDevice>::cpu_real_vector &vVec)
    {
	Layer<TDevice> *tmpPtr = &this->m_precedingLayer;
	TrainableLayer<TDevice> *tLayer = dynamic_cast<layers::TrainableLayer<TDevice>*>(tmpPtr);

	if (tLayer){
	    int tmpSize = tLayer->size() * (tLayer->precedingLayer().size()+1);
	    if (tmpSize != tLayer->weights().size()){
		printf("The layer before MDN is not feedforward. No method to initialize it\n");
	    }else{
		// check mVec and vVec
		Cpu::real_vector mVecTmp;
		if (mVec.size() != this->m_layerSizeTar){
		    mVecTmp.resize(this->m_layerSizeTar, 0.0);
		}else{
		    mVecTmp = mVec;
		}

		// get the average variance
		Cpu::real_vector vVecTmp;
		real_t vVecAver(0.0);
		if (vVec.size() != this->m_layerSizeTar){
		    vVecTmp.resize(this->m_layerSizeTar, 1.0);
		}else{
		    vVecTmp = vVec;
		}
		for (int i = 0; i<this->m_layerSizeTar; i++)
		    vVecAver = vVecAver * i / (i+1) + vVecTmp[i] / (i+1); 
		
		Cpu::real_vector wInit;
		const Configuration &config = Configuration::instance();
		static boost::mt19937 *gen = NULL;
		if (!gen) {
		    gen = new boost::mt19937;
		    gen->seed(config.randomSeed());
		}
		
		wInit.reserve((this->m_endDim-this->m_startDim) * 
			      tLayer->precedingLayer().size());
		boost::random::uniform_real_distribution<real_t> dist1(
			      -1*config.getWInitPara()/tLayer->precedingLayer().size(), 
			       config.getWInitPara()/tLayer->precedingLayer().size());
		for (int i =0; 
		     i<((this->m_endDim-this->m_startDim) * tLayer->precedingLayer().size()); 
		     i++)
		    wInit.push_back(dist1(*gen));
		
		// set w to uniform distribution, set b to mean+variance
		// w starts at precedingSize * startDim
		thrust::copy(wInit.begin(),
			     wInit.end(),
			     tLayer->weights().begin()+
			     this->m_startDim*tLayer->precedingLayer().size()
			     );
		
		// for bias 
		Cpu::real_vector biasInit;
		biasInit.reserve((this->m_endDim - this->m_startDim));
		boost::random::normal_distribution<real_t> dist(0, 1);
		for (int i =0; i<(this->m_endDim - this->m_startDim); i++)
		    biasInit.push_back(1.0/(real_t)m_numMixture);
		
		
		// adjust the bias for mean and variance
		real_t step = (config.getVarInitPara()*2)/(m_numMixture+1);
		real_t start= -1*config.getVarInitPara()+step;
		for (int i =0; i<m_numMixture; i++){
		    for (int j=0; j<m_featureDim; j++){
			biasInit[m_numMixture + i*m_featureDim + j] = 
			    mVecTmp[this->m_startDimOut+j] + 
			    (step * i + start) * vVecTmp[this->m_startDimOut+j];
			if (!m_tieVar){
			    biasInit[m_numMixture*(m_featureDim+1) + i*m_featureDim + j] = 
				internal::safeLog(vVecAver);
			}
		    }
		    if (m_tieVar)
			biasInit[m_numMixture*(m_featureDim+1) + i] = internal::safeLog(vVecAver);
		}		
		    
		
		// set the variance floor
		this->m_varFloor = config.getVFloorPara() * vVecAver;

		// set bias
		thrust::copy(biasInit.begin(),
			     biasInit.end(),
			     tLayer->weights().begin() +
			     tLayer->size()* tLayer->precedingLayer().size() + 
			     this->m_startDim);
		
	    }
	}else{
	    throw std::runtime_error("MDN previous layer can not be untrainable");
	}
    }

    template <typename TDevice>
    void MDNUnit_mixture<TDevice>::computeForward()
    {   
	//
	// softmax part for mixture weight
	//
	// calculate the offset for mixture weight
	{{
		internal::CalculateOffsetFn fn;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.startD       = this->m_startDim;
		fn.endD         = this->m_startDim+this->m_numMixture;
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNoutputs    = helpers::getRawPointer(this->m_precedingLayer.outputs());

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		//n = n*(this->m_paraDim);
		
		thrust::transform(
		   thrust::counting_iterator<int>(0),
		   thrust::counting_iterator<int>(0)+n,
		   this->m_offset.begin(),
		   fn);
		
	}}	    

	// calculate the exp(w_k - offset) 
	{{
		internal::CalculateExpFn fn;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.startD    = this->m_startDim;
		fn.endD      = this->m_startDim + this->m_numMixture;
		fn.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNOutput  = helpers::getRawPointer(this->m_precedingLayer.outputs());
		fn.offset    = helpers::getRawPointer(this->m_offset);
		
		int n =  this->m_precedingLayer.curMaxSeqLength();
		    n = n*this->m_precedingLayer.parallelSequences();
		    n = n*(this->m_numMixture);
		
		thrust::transform(
		   thrust::counting_iterator<int>(0),
		   thrust::counting_iterator<int>(0)+n,
		   this->m_paraVec.begin(),
		   fn);
	}}

	// sum up \sum_k exp(w_k)
	{{
		internal::SumUpOutputsFn fn;
		fn.dimSize   = this->m_numMixture;
		fn.outputs   = helpers::getRawPointer(this->m_paraVec);

		int n =        this->m_precedingLayer.curMaxSeqLength();
		    n = n *    this->m_precedingLayer.parallelSequences();
		// n = n*(this->m_paraDim);
		
		thrust::for_each(
		   thrust::make_zip_iterator(
				 thrust::make_tuple(this->m_offset.begin(),  
						    thrust::counting_iterator<int>(0))),
		   thrust::make_zip_iterator(
				 thrust::make_tuple(this->m_offset.begin()+n,  
						    thrust::counting_iterator<int>(0)+n)),
		   fn);
	}}
	
	// normalize
        {{
		internal::NormalizeOutputsFn fn;
		fn.layerSize = this->m_numMixture;
		fn.normFacts = helpers::getRawPointer(this->m_offset);
		
		int n = this->m_precedingLayer.curMaxSeqLength();
		    n = n * this->m_precedingLayer.parallelSequences()*this->m_numMixture;

		thrust::for_each(
		    thrust::make_zip_iterator(
				thrust::make_tuple(this->m_paraVec.begin(),
						   thrust::counting_iterator<int>(0))),
		    thrust::make_zip_iterator(
				thrust::make_tuple(this->m_paraVec.begin()+n, 
						   thrust::counting_iterator<int>(0)+n)),
		    fn);

        }}
	
	//
	// the mean part (unnessary to change anything. But need to copy from NN output to MDN)
	//
	{{
		internal::CopyMean fn;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.featureDim   = this->m_numMixture*this->m_featureDim;
		fn.startD       = this->m_startDim + this->m_numMixture;
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNOutput     = helpers::getRawPointer(this->m_precedingLayer.outputs());


		// n:  number of timesteps
		// n2: number of the mean parameters  (time * mixNum * featDim)
		int n  = this->m_precedingLayer.curMaxSeqLength();
		    n  = n * this->m_precedingLayer.parallelSequences();
		int n2 = n * this->m_numMixture*this->m_featureDim;
		
		//Cpu::real_vector tmptempParaVec1 = this->m_paraVec;
		
		thrust::for_each(
		    thrust::make_zip_iterator(
			  thrust::make_tuple(this->m_paraVec.begin()+n*this->m_numMixture,
					     thrust::counting_iterator<int>(0))),
		    thrust::make_zip_iterator(
                          thrust::make_tuple(this->m_paraVec.begin()+n2+n*this->m_numMixture, 
					     thrust::counting_iterator<int>(0)+n2)),
		    fn);
		
		//Cpu::real_vector tmptempParaVec2 = this->m_paraVec;
		//printf("\n");
	}}

	//
	// the variance part
	//
	// calculate the Exp
	{{
		internal::CalculateExpSimpleFnForVar fn;
		fn.NNOutputSize =  this->m_precedingLayer.size();
		fn.startD       =  this->m_startDim + this->m_numMixture*(1+this->m_featureDim);
		fn.endD         = (this->m_startDim + this->m_numMixture*(1+this->m_featureDim) +
				   this->m_numMixture * (this->m_tieVar?1:this->m_featureDim));
		
		fn.varFloor     = this->m_varFloor;
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNOutput     = helpers::getRawPointer(this->m_precedingLayer.outputs());

		// total time step
		int timeStep = this->m_precedingLayer.curMaxSeqLength();
		timeStep     = timeStep*this->m_precedingLayer.parallelSequences();
		
		// total number of parameter for variance 
		int paraNum  = timeStep*this->m_numMixture*(this->m_tieVar?1:this->m_featureDim);

		thrust::transform(
		   thrust::counting_iterator<int>(0),
		   thrust::counting_iterator<int>(0)+paraNum,
		   (this->m_paraVec.begin() +   // point to var part
		    timeStep * (this->m_numMixture*(this->m_featureDim+1))), 
		   fn);

		#ifdef DEBUG_LOCAL
		if(0){
		Cpu::real_vector temp_vec1;
		Cpu::real_vector temp_vec2;
		temp_vec1 = this->m_paraVec;
		temp_vec2 = this->m_precedingLayer.outputs();
		real_t tmp(0.0);
		int timeStep = (6390/this->m_numMixture);
		int mixIndex = (6390%this->m_numMixture);
		int pos_var  = timeStep * this->m_numMixture * (1 + this->m_featureDim) +
		    timeStep*this->m_numMixture + mixIndex;
		
		for (int i = this->m_featureDim-10; i<this->m_featureDim; i++){
		    tmp += (temp_vec1[pos_var+i]);	
		}
		}
		#endif
	}}
    }


    template <typename TDevice>
    void MDNUnit_mixture<TDevice>::getEMOutput(const real_t para, real_vector &targets)
    {

	const Configuration &config = Configuration::instance();

	int totalTime = this->m_precedingLayer.curMaxSeqLength();
	totalTime     = totalTime*this->m_precedingLayer.parallelSequences();
	int time      = totalTime*(this->m_endDimOut - this->m_startDimOut);
	
	/*Modify */
	// initialization of the output 
	// thrust::fill(this->m_tmpPat.begin(), this->m_tmpPat.end(), 0.0);

	// initialization of the output using predicted weights
	{{
		internal::copyMixtureWeightforEMGen fn;
		fn.mixture_num  = this->m_numMixture;		
		fn.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.meanDis   = helpers::getRawPointer(this->m_tmpPat);
		fn.mdnPara   = helpers::getRawPointer(this->m_paraVec);

		int n = this->m_precedingLayer.curMaxSeqLength();
		n = n * this->m_precedingLayer.parallelSequences();
		
		fn.totalTime = n;
		thrust::for_each(thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0)+n,
				 fn
				 );
				
		#ifdef DEBUG_LOCAL
		Cpu::real_vector tvec1;
		Cpu::real_vector tvec2;
		tvec1 = this->m_paraVec;
		tvec2 = this->m_tmpPat;
		for (int i = 0; i < n; i++){
		    printf("");
		}
		#endif
	}}

	
	real_t outP   = 0.0;     
	bool finish   = false;
	int iter      = 0;
	while(!finish)
	{   
	    {{

		// initIterEMGen: compute the argmax_o_new \sum_m p(m|o_old) log p(o_new, m)
		internal::initIterEMGen fn;
		fn.featureDim   = this->m_featureDim;
		fn.mixtureNM    = this->m_numMixture;
		fn.outputSize   = this->m_layerSizeTar;
		fn.startDOut    = this->m_startDimOut;
		fn.totalTime    = totalTime;
		fn.tieVar       = this->m_tieVar;

		fn.postP        = helpers::getRawPointer(this->m_tmpPat);
		fn.mdnPara      = helpers::getRawPointer(this->m_paraVec);
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.targets      = helpers::getRawPointer(targets);
		thrust::for_each(
				  thrust::counting_iterator<int>(0),
				  thrust::counting_iterator<int>(0)+time,
				  fn);

		#ifdef DEBUG_LOCAL
		Cpu::real_vector mdnPara;
		Cpu::real_vector postP;
		Cpu::real_vector tar;
		mdnPara = this->m_paraVec;
		postP   = this->m_tmpPat;
		tar     = targets;
		for (int idx = 0; idx < time; idx ++){
		    int timeStep = idx / (fn.featureDim);
		    int dimOutput= idx % (fn.featureDim);

		    real_t m, v, p, q;
		    int pos_mean = fn.totalTime * fn.mixtureNM + 
			timeStep * fn.featureDim * fn.mixtureNM + dimOutput;
		    int pos_var  = fn.totalTime * (fn.mixtureNM + fn.featureDim * fn.mixtureNM) 
			+ timeStep * fn.mixtureNM;
		    if (!this->m_tieVar)
			pos_var = pos_mean + fn.totalTime * fn.mixtureNM * fn.featureDim;

		    int pos_postM= timeStep * fn.mixtureNM;
		    int pos_postS= fn.totalTime*fn.mixtureNM + timeStep;

		    real_t tmp1=0.0;
		    real_t tmp2=0.0;
		    for(int i = 0; i < fn.mixtureNM; i++){
			v = mdnPara[pos_var];
			m = mdnPara[pos_mean];
			p = postP[pos_postM];
			q = postP[pos_postS];

			tmp2 += exp(p-q)/((v)*(v));
			tmp1 += ((m)*(exp(p-q)))/((v)*(v));
			pos_var  += this->m_tieVar?1:fn.featureDim; 
			pos_mean += fn.featureDim; 
			pos_postM+=1;
		    }
		    tmp1 = tmp1/tmp2;
		    printf("Time %d %d %f:\n", timeStep, dimOutput, tmp1);
		}
		#endif


	    }}

	    // iteration
	    {{		
		// calculate the posterior probability using the functions in calculateError
		internal::ComputeMixtureDistance fn;
		fn.startDOut    = this->m_startDimOut;
		fn.mixture_num  = this->m_numMixture;
		fn.featureDim   = this->m_featureDim;
		fn.layerSizeOut = this->m_layerSizeTar;
		fn.tieVar       = this->m_tieVar;
		
		fn.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.output    = helpers::getRawPointer(targets);
		fn.mdnPara   = helpers::getRawPointer(this->m_paraVec);

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		fn.totaltime = n;
		n = n*this->m_numMixture;
		
		thrust::transform(thrust::counting_iterator<int>(0),
				  thrust::counting_iterator<int>(0)+n,
				  this->m_tmpPat.begin(),
				  fn);		

		#ifdef DEBUG_LOCAL
		Cpu::real_vector temp_vec1;
		Cpu::real_vector temp_vec2;
		Cpu::real_vector temp_vec3;
		Cpu::real_vector temp_vec4;
		
		temp_vec1 = this->m_paraVec;
		temp_vec2 = targets;
		temp_vec3 = this->m_tmpPat;
		temp_vec4  = this->m_precedingLayer.patTypes();
		printf("MixtureDistance\n");
		for (int t = 0; t < n; t++){
		    real_t tmp=0.0;
		    real_t tmptmp;

		    int timeStep = (t/fn.mixture_num);
		    int mixIndex = (t%fn.mixture_num);
		    int pos_data = fn.layerSizeOut * timeStep + fn.startDOut;
		    int pos_mean = fn.totaltime * fn.mixture_num + 
			timeStep*fn.featureDim*fn.mixture_num+mixIndex*fn.featureDim;
		    int pos_var  = fn.totaltime * (fn.mixture_num + 
						   fn.mixture_num * fn.featureDim) +
			timeStep*fn.mixture_num + mixIndex;
		    if (!this->m_tieVar)
			pos_var  = fn.totaltime * (fn.mixture_num + 
						   fn.mixture_num * fn.featureDim) +
			    timeStep*fn.mixture_num*fn.featureDim + mixIndex*fn.featureDim;
		    for (int i = 0; i<fn.featureDim; i++){
			tmp += ((temp_vec2[pos_data+i]) - temp_vec1[pos_mean+i]) * 
			    ((temp_vec2[pos_data+i]) - temp_vec1[pos_mean+i])
			    /(temp_vec1[pos_var+(this->m_tieVar?0:1)])
			    /(temp_vec1[pos_var+(this->m_tieVar?0:1)])/2;
		    }
		    printf("%f %f %f %f\t", temp_vec1[pos_mean], temp_vec2[pos_data], 
			   temp_vec1[pos_var], tmp);
		    if (mixIndex==(fn.mixture_num - 1))
			printf("\n");
		}
		printf("\nEnd\n");
		#endif
	    }}

	    {{
		internal::ComputeMixtureError fn;
		fn.startD       = this->m_startDim;
		fn.startDOut    = this->m_startDimOut;
		fn.mixture_num  = this->m_numMixture;
		fn.featureDim   = this->m_featureDim;
		fn.layerSizeOut = this->m_layerSizeTar;
		fn.tieVar       = this->m_tieVar;
		

		fn.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.meanDis   = helpers::getRawPointer(this->m_tmpPat);
		fn.mdnPara   = helpers::getRawPointer(this->m_paraVec);

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		
		fn.totalTime = n;

		outP = thrust::transform_reduce(thrust::counting_iterator<int>(0),
					       thrust::counting_iterator<int>(0)+n,
					       fn,
					       (real_t)0.0,
					       thrust::plus<real_t>());
		if (outP != outP){
		    printf("\t\t Fail to converge\n");
		}else{
		    printf("\t\t Output likelihood/dim (-log): %f\n", outP/n/this->m_featureDim);
		}

		#ifdef DEBUG_LOCAL
		
		Cpu::real_vector tvec1;
		Cpu::real_vector tvec2;
		tvec1 = this->m_paraVec;
		tvec2 = this->m_tmpPat;
		for (int t = 0; t < n; t++){
		    real_t tmp=-1e30;
		    real_t tmptmp;
		    
		    int meanPos = t * fn.mixture_num;
		    int mixPos  = t * fn.mixture_num;
		    int varPos  = fn.totalTime*(fn.mixture_num+fn.mixture_num*fn.featureDim)+
			t * fn.mixture_num * (this->m_tieVar?1:fn.featureDim);
		    
		    for (int i = 0; i<fn.mixture_num; i++){
			
			tmptmp = std::log(tvec1[mixPos+i])-(tvec2[meanPos+i]);
			tmptmp = tmptmp - fn.featureDim/2*std::log(2*PI_DEFINITION);
			// change this line according to ALTER_TIEVAR
			//tmptmp = tmptmp - fn.featureDim*std::log(tvec1[varPos]);
			if (this->m_tieVar)
			    tmptmp = tmptmp - fn.featureDim*std::log(tvec1[varPos+i]);
			else{
			    for (int j = 0; j<fn.featureDim; j++)
				tmptmp = tmptmp - std::log(tvec1[varPos+i*fn.featureDim+j]);
			}
			    
			//tmptmp = std::exp(tmptmp);
			tmp   = internal::logAdd(tmp, tmptmp);
			printf("%f\t", tmptmp);
			printf("%f\t", tvec2[t*fn.mixture_num + i]);
		    }
		    printf("%f \n", tvec2[fn.totalTime*fn.mixture_num + t]);
		    printf("%f \n", tmp);
		}
		
		#endif

	    }}
	    iter++;
	    
	    if (iter >= config.EMIterNM())
		finish = true;
	}
	
    }

    template <typename TDevice>
    void MDNUnit_mixture<TDevice>::getOutput(const real_t para,real_vector &targets)
    {

	int time = this->m_precedingLayer.curMaxSeqLength();
	time = time*this->m_precedingLayer.parallelSequences();
	time = time*(this->m_endDimOut - this->m_startDimOut);
	
	Cpu::real_vector temp;
	Gpu::real_vector temp2;
	temp.reserve(time);
	
	const Configuration &config = Configuration::instance();

	static boost::mt19937 *gen = NULL;
	if (!gen) {
	    gen = new boost::mt19937;
	    gen->seed(config.randomSeed());
	}
	
	boost::random::normal_distribution<real_t> dist(0, 1);
	for (size_t i = 0; i < time; ++i)
	    temp.push_back(dist(*gen));

			
	// copy to GPU
	temp2 = temp;	
	{{
		internal::SamplingMixture fn;
		fn.featureDim   = this->m_featureDim;
		fn.layerSizeOut = this->m_layerSizeTar;
		fn.startDOut    = this->m_startDimOut;
		fn.mixtureNum   = this->m_numMixture;
		fn.totalTime    = (int)(time/(this->m_featureDim));
		fn.para         = para;
		fn.paraPtr      = ( (this->m_varScale.size()>0) ?
				    (helpers::getRawPointer(this->m_varScale)) : NULL );
		fn.targets      = helpers::getRawPointer(targets);
		fn.mdnPara      = helpers::getRawPointer(this->m_paraVec);
		fn.tieVar       = this->m_tieVar;

		thrust::for_each(
  			 thrust::make_zip_iterator(
			     thrust::make_tuple(temp2.begin(), 
						thrust::counting_iterator<int>(0))),
		         thrust::make_zip_iterator(
			     thrust::make_tuple(temp2.begin()+time, 
						thrust::counting_iterator<int>(0)+time)),
			 fn);
		
                #ifdef DEBUG_LOCAL
		Cpu::real_vector mdnPara = this->m_paraVec;
		for (int t = 0; t<time; t++){
		    int timeStep = t / (fn.featureDim);
		    int dimStep  = t % (fn.featureDim);
		    
		    
		    real_t tmp = 0.0;
		    int flag = 0;
		    for (int i = 0; i<fn.mixtureNum; i++){
			if (mdnPara[timeStep*fn.mixtureNum + i] > tmp){
			    tmp = mdnPara[timeStep*fn.mixtureNum + i];
			    flag = i;
			}
		    }
		    
		    int pos = fn.totalTime*fn.mixtureNum + timeStep*fn.mixtureNum*fn.featureDim;
		    const real_t mean = mdnPara[pos + flag*fn.featureDim + dimStep];
		    pos = fn.totalTime*(fn.mixtureNum+fn.mixtureNum*fn.featureDim) ;
                    #ifdef ALTER_TIEVAR
		    const real_t var = mdnPara[pos + timeStep*fn.mixtureNum];
                    #else
		    real_t var = mdnPara[pos + timeStep*fn.mixtureNum + flag];
		    if (!this->m_tieVar)
			var = mdnPara[pos + (timeStep*fn.mixtureNum+flag)*fn.featureDim + dimStep];
                    #endif	    

		    pos = timeStep * fn.layerSizeOut + fn.startDOut + dimStep;
		    printf("%d %f\t", pos, var*para*temp[t] + mean);
		    
		}
                #endif
		
	}}	
    }

    template <typename TDevice>
    void MDNUnit_mixture<TDevice>::getParameter(real_t *targets)
    {

	//
	{{
		internal::GetParameterMixture fn;

		int time = this->m_precedingLayer.curMaxSeqLength();
		time = time*this->m_precedingLayer.parallelSequences();
		
		fn.targets      = targets;
		fn.featureDim   = this->m_featureDim;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.startDimIn   = this->m_startDim;
		fn.mixtureNum   = this->m_numMixture;
		fn.totalTime    = time;
		fn.targets      = targets;
		fn.mdnPara      = helpers::getRawPointer(this->m_paraVec);
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.tieVar       = this->m_tieVar;

		thrust::for_each(thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0)+time,
				 fn);
	}}
    }


    template <typename TDevice>
    real_t MDNUnit_mixture<TDevice>::calculateError(real_vector &targets)
    {   
	
	// step1: calculate the sum_dim (t_n_d - \mu_d_k)^2
	// and save the result to m_tmpPat[0 : totalTime*mixture_num]
	{{
		internal::ComputeMixtureDistance fn;
		fn.startDOut   = this->m_startDimOut;
		fn.mixture_num = this->m_numMixture;
		fn.featureDim  = this->m_featureDim;
		fn.layerSizeOut= this->m_layerSizeTar;
		fn.tieVar      = this->m_tieVar;

		fn.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.output    = helpers::getRawPointer(targets);
		fn.mdnPara   = helpers::getRawPointer(this->m_paraVec);

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		fn.totaltime = n;
		n = n*this->m_numMixture;

		thrust::transform(thrust::counting_iterator<int>(0),
				  thrust::counting_iterator<int>(0)+n,
				  this->m_tmpPat.begin(),
				  fn);		

                #ifdef DEBUG_LOCAL
		Cpu::real_vector temp_vec1;
		Cpu::real_vector temp_vec2;
		Cpu::real_vector temp_vec3;
		temp_vec1 = this->m_paraVec;
		temp_vec2 = targets;
		temp_vec3 = this->m_tmpPat;
		printf("MixtureDistance: data, mean, var (one dimension), dis_over_dim\n");
		for (int t = 0; t < n; t++){
		    real_t tmp=0.0;
		    real_t tmptmp;

		    int timeStep = (t/fn.mixture_num);
		    int mixIndex = (t%fn.mixture_num);
		    int pos_data = fn.layerSizeOut * timeStep + fn.startDOut;
		    int pos_mean = fn.totaltime * fn.mixture_num + 
			timeStep*fn.featureDim*fn.mixture_num+mixIndex*fn.featureDim;
		    int pos_var  = fn.totaltime * (fn.mixture_num + 
						   fn.mixture_num * fn.featureDim) +
			timeStep*fn.mixture_num + mixIndex;
		    if (!this->m_tieVar){
			pos_var = fn.totaltime * (fn.mixture_num + 
						  fn.mixture_num * fn.featureDim) +
			    (timeStep*fn.mixture_num+mixIndex)*fn.featureDim;
		    }
		    for (int i = 0; i<fn.featureDim; i++){
			tmp += (temp_vec2[pos_data+i] - temp_vec1[pos_mean+i]) * 
			    (temp_vec2[pos_data+i] - temp_vec1[pos_mean+i])
			    /(temp_vec1[pos_var + (fn.tieVar?0:i)])
			    /(temp_vec1[pos_var + (fn.tieVar?0:i)])/2;
		    }
		    printf("%03.4f %03.4f %03.4e %03.4e\t", 
			   temp_vec2[pos_data], temp_vec1[pos_mean], 
			   temp_vec1[pos_var+mixIndex], tmp);
		    if (mixIndex==(fn.mixture_num - 1))
			printf("\n");
		}
		printf("\nEnd\n");
                #endif

	}}
	

	// step2: calcualte the - log likelihood
	//     save w_i p_i to m_tmpPat[0 : totalTime*mixture_num]
	// and save the sum_i^mixture_num w_i p_i to m_tmpPat[totalTime*mixture_num:end]
	//     (for both likelihood calculation and back-propagation)
	real_t mixError = 0.0;
	{{
		internal::ComputeMixtureError fn;
		fn.startD       = this->m_startDim;
		fn.startDOut    = this->m_startDimOut;
		fn.mixture_num  = this->m_numMixture;
		fn.featureDim   = this->m_featureDim;
		fn.layerSizeOut = this->m_layerSizeTar;
		fn.tieVar       = this->m_tieVar;

		fn.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.meanDis   = helpers::getRawPointer(this->m_tmpPat);
		fn.mdnPara   = helpers::getRawPointer(this->m_paraVec);

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		
		fn.totalTime = n;

                #ifndef DEBUG_LOCAL
		mixError = thrust::transform_reduce(thrust::counting_iterator<int>(0),
						    thrust::counting_iterator<int>(0)+n,
						    fn,
						    (real_t)0.0,
						    thrust::plus<real_t>());
                #endif

                #ifdef DEBUG_LOCAL
		Cpu::real_vector tvec1;
		Cpu::real_vector tvec2;
		tvec1 = this->m_paraVec;
		tvec2 = this->m_tmpPat;
		printf("Calculate Error (each time step): each mixture, sum\n");
		for (int t = 0; t < n; t++){
		    real_t tmp=-1e30;
		    real_t tmptmp;
		    
		    int meanPos = t * fn.mixture_num;
		    int mixPos  = t * fn.mixture_num;
		    int varPos  = fn.totalTime*(fn.mixture_num+fn.mixture_num*fn.featureDim)+
			t * fn.mixture_num;
		    
		    if (!this->m_tieVar){
			varPos  = fn.totalTime*(fn.mixture_num+fn.mixture_num*fn.featureDim)+
			t * fn.mixture_num * fn.featureDim;
		    
		    }

		    for (int i = 0; i<fn.mixture_num; i++){
			
			tmptmp = std::log(tvec1[mixPos+i])-(tvec2[meanPos+i]);
			tmptmp = tmptmp - fn.featureDim/2*std::log(2*PI_DEFINITION);
			if (this->m_tieVar)
			    tmptmp = tmptmp - fn.featureDim*std::log(tvec1[varPos+i]);
			else{
			    for(int j = 0 ; j<fn.featureDim; j++)
				tmptmp = tmptmp - std::log(tvec1[varPos+i*fn.featureDim+j]);
			}
			
			if (tmptmp!=tmptmp || tmptmp < -0.5e10f){
			    tmptmp = -0.5e10f;
			}
			//tmptmp = std::exp(tmptmp);
			tmp   = internal::logAdd(tmp, tmptmp);
			//printf("%f\t", tvec2[t*fn.mixture_num + i]);
			printf("%f\t", tmptmp);
			tvec2[t*fn.mixture_num + i] = tmptmp;
		    }
		    //printf("%f \n", tvec2[fn.totalTime*fn.mixture_num + t]);
		    printf("%f \n", tmp);
		    tvec2[fn.totalTime*fn.mixture_num + t] = tmp;
		    mixError -= tmp;
		}
		printf("Error done\n");
		this->m_tmpPat = tvec2;
                #endif
	}}
	return mixError;
    }                

    template <typename TDevice>
    void MDNUnit_mixture<TDevice>::computeBackward(real_vector &targets)
    {                          
	
	// clean the outputErrors
	{{
		// In the original case, each dimension in outputErrors will be assigned new value
		// Thus, no need to reset outputErrors
		// However, for updating the variance here, we accumulate the gradients.
		// Thus, need to reset outputErrors
		// thrust::fill(this->m_precedingLayer.outputErrors().begin(),
		//              this->m_precedingLayer.outputErrors().end(),
		//	        (real_t)0.0);
		
		// 
		// STUPID ERROR !!! Reset outputErrors in each computeBackward will wipe
		// up the gradients of previous MDNUnit.
	}}
	

	// step1: update the mixture weight
	{{
		internal::ComputeBPmixtureWeight fn;
		fn.mixture_num = this->m_numMixture;
		fn.NNOutputSize= this->m_precedingLayer.size();
		fn.startD      = this->m_startDim;
		fn.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.meanDis   = helpers::getRawPointer(this->m_tmpPat);
		fn.mdnPara   = helpers::getRawPointer(this->m_paraVec);
		fn.errors    = helpers::getRawPointer(this->m_precedingLayer.outputErrors());
		

		int n = this->m_precedingLayer.curMaxSeqLength();
		n     = n * this->m_precedingLayer.parallelSequences();
		
		fn.totalTime = n;
		n     = n * this->m_numMixture;
		thrust::for_each(thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0)+n,
				 fn
				 );

                #ifdef DEBUG_LOCAL
		printf("Gradient for mixture weight\n");
		Cpu::real_vector mdnPara = this->m_paraVec;
		Cpu::real_vector meanDis = this->m_tmpPat;
		Cpu::real_vector errors  = this->m_precedingLayer.outputErrors();
		for (int i = 0; i<n; i++){
		    int outputIdx = i;
		    const int timeStep = outputIdx / fn.mixture_num;
		    const int mixtureI = (outputIdx % fn.mixture_num);

		    // to the posterior 
		    const int postP  = timeStep * fn.mixture_num + mixtureI;
		    const int sumPost= fn.totalTime * fn.mixture_num + timeStep;

		    // to the output of MDN (mixture weight)
		    int pos = timeStep * fn.mixture_num + mixtureI;
		    const real_t sigma  = mdnPara[pos];

		    // Time, gradient
		    // store the gradients
		    pos = timeStep * fn.NNOutputSize + fn.startD + mixtureI;
		    if (mixtureI == fn.mixture_num - 1){
			printf("(only last dim) %d %f\t", 
			       timeStep, sigma - std::exp(meanDis[postP]-meanDis[sumPost]));
			printf("\n");
		    }
		}
		printf("GrafEnd\n");
                #endif

		
	}}
	
	// step2: update the mixture mean and variance
	{{

		internal::ComputeBPmixtureMeanVariance fn;
		fn.layerSize    = this->m_precedingLayer.size();
		fn.startD       = this->m_startDim + this->m_numMixture;
		fn.startDOut    = this->m_startDimOut;
		fn.layerSizeOut = this->m_layerSizeTar;
		fn.featureDim   = this->m_featureDim;
		fn.mixture_num  = this->m_numMixture;
		fn.tieVar       = this->m_tieVar;
		
		fn.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.meanDis   = helpers::getRawPointer(this->m_tmpPat);
		fn.mdnPara   = helpers::getRawPointer(this->m_paraVec);
		fn.errors    = helpers::getRawPointer(this->m_precedingLayer.outputErrors());
		fn.target    = helpers::getRawPointer(targets);
		fn.varBuff   = helpers::getRawPointer(this->m_varBP);

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		
		fn.totalTime = n;
		n = n * this->m_numMixture * this->m_featureDim;
 		thrust::for_each(thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0)+n,
				 fn);
		
		// For untied case, the gradients have been assigned to individual dimension
		// no need to merge the gradients for tiedVariance
		if (this->m_tieVar){
		    internal::ComputeBPAccumVariance fn2;
		    fn2.layerSize   = this->m_precedingLayer.size();
		    fn2.startD      = this->m_startDim + this->m_numMixture;
		    fn2.featureDim  = this->m_featureDim;
		    fn2.mixture_num = this->m_numMixture;

		    fn2.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		    fn2.errors    = helpers::getRawPointer(this->m_precedingLayer.outputErrors());
		    fn2.varBuff   = helpers::getRawPointer(this->m_varBP);

		    n =this->m_precedingLayer.curMaxSeqLength();
		    n = n*this->m_precedingLayer.parallelSequences();		
		    thrust::for_each(thrust::counting_iterator<int>(0),
				     thrust::counting_iterator<int>(0)+n,
				     fn2);
		}
		
                #ifdef DEBUG_LOCAL
		
		Cpu::real_vector errors = this->m_precedingLayer.outputErrors();
		Cpu::real_vector mdnPara= this->m_paraVec;
		Cpu::real_vector meanDis= this->m_tmpPat;
		Cpu::real_vector target = targets;
		Cpu::real_vector varBuf = this->m_varBP;
		Cpu::real_vector errorVBuf;
		real_t errorm(0.0), errorv(0.0);
		printf("Gradient for mean and variance of each mixture: time errorm errorv ...\n");
		n = fn.totalTime * this->m_numMixture * this->m_featureDim;
		
		errorVBuf.resize(n, 0.0);
		for (int i = 0; i<n; i++){
		    int outputIdx = i;
		    const int timeStep = outputIdx / (fn.mixture_num * fn.featureDim);

		    const int tmp = outputIdx % (fn.mixture_num * fn.featureDim);
		    const int mixtureI = tmp / fn.featureDim;
		    const int featureI = tmp % fn.featureDim;
	    
		    // pointer to the mean gradient
		    int meanshift_error = timeStep*fn.layerSize+fn.startD+
			mixtureI*fn.featureDim + featureI;
		    //real_t errorm = errors[meanshift_pos];

		    //real_t errorv = errors[varshift];
	    
		    // pointer to the target data y
		    const real_t tardata= target[timeStep*fn.layerSizeOut+fn.startDOut+featureI];

		    // pointer to the mean
		    int meanshift= fn.totalTime * fn.mixture_num + 
			timeStep * fn.mixture_num * fn.featureDim + 
			mixtureI * fn.featureDim + 
			featureI;
		    
		    int varshift = fn.tieVar?
			(fn.totalTime * fn.mixture_num * (1 + fn.featureDim) + 
			 timeStep * fn.mixture_num + mixtureI):
			(fn.totalTime * fn.mixture_num * (1 + fn.featureDim) + 
			 (timeStep * fn.mixture_num + mixtureI)*fn.featureDim + featureI);
	    
		    const real_t mean  = mdnPara[meanshift];
		    const real_t var   = mdnPara[varshift];

		    // pointer to the posterior P and sum of posterior P
		    const real_t postP = meanDis[timeStep * fn.mixture_num + mixtureI];
		    const real_t sumPost=meanDis[fn.totalTime* fn.mixture_num + timeStep];
		    real_t posterior = internal::safeExp((postP) - (sumPost));
		    (errorm) = posterior*(mean - tardata)/(var)/(var);
		    
		    (errorv) += posterior - (errorm)*(mean - tardata);
		    
		    errorVBuf[i] = posterior - (errorm)*(mean - tardata);
		    /*if (errorVBuf[i]<-5 || errorVBuf[i]>5)
		      printf("get");*/
		    if (mixtureI == 0 && featureI ==0)
			printf("%d\t", timeStep);
		    if (featureI==fn.featureDim - 1){
			printf("%d %3.4f %3.4f \t\t", mixtureI, errorm, errorv); 
			/*if (errorv > 100 || errorv < -100.0){
			    printf("get");
			    for (int j=0; j<fn.featureDim; j++)
				printf("%3.4f ", errorVBuf[i-j]);
				}*/
			errorv = 0;
			if (mixtureI==(fn.mixture_num - 1)){
			    printf("\n");
			}
		    }
		}
		
		printf("GrafEnd\n");
                #endif

	}}

    }
    
    template <typename TDevice>
    bool MDNUnit_mixture<TDevice>::flagVariance() const
    {
	return m_tieVar; // default for all units
    }

    /********************************************************
     MDNUnit_mixture_dyn Definition
    *******************************************************/
    template <typename TDevice>
    MDNUnit_mixture_dyn<TDevice>::MDNUnit_mixture_dyn(
	int startDim,       int endDim, int startDimOut, int endDimOut,      int type, 
	Layer<TDevice> &precedingLayer, int outputSize,  const bool tieVar, 
	int weightStart, int weightNum, int backOrder,   const int trainable, 
	const int dynDirection, const bool realPole, const int tanhRegOpt)
        : MDNUnit_mixture<TDevice>(startDim, endDim, startDimOut, endDimOut, 
				   type, endDimOut - startDimOut, precedingLayer, 
				   outputSize, tieVar, trainable)
    {
	const Configuration &config = Configuration::instance();
	
	// type refers to the number of mixture here
	int numMixture = type;
	
	m_dynDirection = dynDirection;
	m_weightStart  = weightStart;
	m_weightNum    = weightNum;
	m_backOrder    = backOrder;
	
	
	// initialize the dataBuff, which has the same wides as output of this MDNUnit
	m_maxTime      = precedingLayer.maxSeqLength();
	
	#ifdef MIXTUREDYNDIAGONAL
	// to save the intermediate results, such as the gradients
	m_dataBuff.resize (m_maxTime   * this->m_featureDim * 
			   (numMixture * m_backOrder + numMixture), 0.0);
	#else
	printf("Backorder > 1 is not implemented for full matrix regression");
	m_dataBuff.resize (m_maxTime * this->m_featureDim * (numMixture + 1),    0.0);
	#endif

	m_tanhReg      = tanhRegOpt;
	
	if (m_tanhReg == MDNARRMDN_CASECADEREAL){
	    // for the tanh-based casecade filter form, the poles need to be transformed 
	    // into coefficients of AR filter
	    // m_wTransBuff: the buffer for coefficients convolution
	    // 
	    // buffer for convolution in forward: 
	    //   [0 this->m_featureDim*(m_backOrder+1)*2-1]
	    // buffer for convolution in backward: 
	    //   [this->m_featureDim*(m_backOrder+1)*2,this->m_featureDim*(m_backOrder+1)*(2+2N)-1]
	    //   here, every this->m_featureDim*(m_backOrder+1)*2 elements are one block for one 
	    //   order of the Pole-based AR model
	    // buffer for convolution in backward: 
	    //   [this->m_featureDim*(m_backOrder+1)*2+2N, ...]
	    //   the final this->m_featureDim*(m_backOrder+1)*N block is used to store the 
	    //   poles for each classifical form w.r.t gradient of one pole
	    // The remaining this->m_featureDim*(m_backOrder+1)*1 is used to store the weights
	    ///  temporily
	    m_casOrder     = m_backOrder;
	    m_casRealPole  = realPole;
	    m_wTransBuff.resize(this->m_featureDim*(m_backOrder+1)*(m_casOrder*3+2), 0);

	}else if (m_tanhReg == MDNARRMDN_CASECADECOMPLEX){
	    // for complex filters
	    m_casOrder     = m_backOrder/2;
	    m_casRealPole  = realPole;
	    m_wTransBuff.resize(this->m_featureDim*(m_backOrder+1)*(m_casOrder*3+3), 0);
	    
	}else{
	    // no need to use additional buffer 
	    m_wTransBuff.clear();
	    m_casOrder = 0;
	}
	
	// obsolete option
	m_arrmdnLearning  = config.arrmdnLearning();
	m_arrmdnUpInter   = config.arRMDNUpdateInterval();
 
	// Block 1025x03

	// length of the AR parameter 
	// Note: m_weightShiftToDim and m_wTransBuffShiftToDim are two general
	//       pointers. Their usesage are different in different cases
	if (dynDirection == MDNUNIT_TYPE_1_DIRECT){
	    // AR along the time axis
	    m_linearPartLength = this->m_featureDim;
	    m_biasPartLength   = this->m_featureDim;
	    
	    // m_wTransBuff contains two blocks for forward and backward computation
	    // 2*m_wTransBuffShiftToDim points to the first element for backward computation
	    m_wTransBuffShiftToDim = this->m_featureDim * (m_backOrder + 1);
	    
	    // m_weightShiftToDim points to the position for convolution in backward computation 
	    // used to store the filter coefficients with one filter left
	    m_weightShiftToDim     = this->m_featureDim * (m_backOrder + 1) * (m_casOrder+1) * 2;
	    
	    // to temporary store the transformed parameter
	    m_wTransBuffParaBK     = this->m_featureDim * (m_backOrder + 1) * (m_casOrder*3 + 2);
	    
	}else{
	    // Block 1025x02:
	    // ARRMDN along 
	    throw std::runtime_error("ARRMDN along dimension axis is no longer supported");
	}
	
	// m_oneVec is used to accumulate the gradient in backpropagation stage
	cpu_real_vector temp;	
	temp.resize(m_maxTime * numMixture * this->m_featureDim, 1.0);
	m_oneVec      = temp;
    }
    
    template <typename TDevice>
    MDNUnit_mixture_dyn<TDevice>::~MDNUnit_mixture_dyn()
    {                                                           
    }

    template <typename TDevice>
    bool MDNUnit_mixture_dyn<TDevice>::flagValid()
    {
	return this->MDNUnit_mixture<TDevice>::flagValid();
    }

    template <typename TDevice>
    int MDNUnit_mixture_dyn<TDevice>::tanhRegType() const
    {
	return m_tanhReg;
    }
    
     template <typename TDevice>
    void MDNUnit_mixture_dyn<TDevice>::linkWeight(real_vector& weights, 
						  real_vector& weightsUpdate)
    {                                                           
	m_weights          = &weights; // point to the vector data (but with bias)
	
	const Configuration &config = Configuration::instance();
	
	if (config.zeroFilter()){
	    // only set the weight part to zero, not the bias
	    thrust::fill(weights.begin() + m_weightStart, 
			 weights.begin() + m_weightStart + m_weightNum - m_biasPartLength, 
			 0.0);
	}

	// For Complex filter, if casRealPole==True, the first filter should be 1st order
	if (m_tanhReg == MDNARRMDN_CASECADECOMPLEX && m_casRealPole){
	    // set the 2nd order coefficients of the first filter to zero
	    thrust::fill(weights.begin() + m_weightStart + this->m_featureDim, 
			 weights.begin() + m_weightStart + this->m_featureDim * 2, 
			 0.0);
	}

	
	if (weights.size() < m_weightStart + m_weightNum){
	    printf("Unmatched weight pos: %d %d %d\n", (int)weights.size(), 
		   m_weightStart, m_weightNum);
	    throw std::runtime_error("Error in weight configuration of MDN unit");
	}
	m_weightsPtr       = helpers::getRawPointer(weights) + m_weightStart;
	
	m_weightUpdates    = &weightsUpdate;
	m_weightUpdatesPtr = helpers::getRawPointer(weightsUpdate) + m_weightStart;
	
    }
    
    template <typename TDevice>
    void MDNUnit_mixture_dyn<TDevice>::computeForward()
    {
	// do the normal feedforward for the MDN part
	this->MDNUnit_mixture<TDevice>::computeForward();

	// This code block is obsolete
	// Only ARRMDN over time dimension is supported
	// Block 1025x01

	// For AR based on casecade form 
	// conver the poles of AR to the coefficients of AR model
	if ((this->m_tanhReg >0) && this->m_backOrder >= 1){
	    
	    if (this->m_dynDirection != MDNUNIT_TYPE_1_DIRECT){
		printf("AR based on casecade form is not implemented along dimension axis");
		throw std::runtime_error("Implementation Error");
	    }
	    // initialize m_wTransBuff as zero
	    thrust::fill(this->m_wTransBuff.begin(), this->m_wTransBuff.end(), (real_t)0.0);
		

	    if (this->m_tanhReg == MDNARRMDN_CASECADEREAL){
		
		// AR with all real poles
		// convert the poles of casecade AR into coefficients of classical AR
		// based on convolution
		{{
		internal::TanhAutoRegConvolution fn;
		fn.backOrder   = this->m_backOrder;
		fn.featureDim  = this->m_featureDim;
		fn.weightPoles = this->m_weightsPtr;
		// the coefficient buffer
		fn.weightCoef1 = helpers::getRawPointer(this->m_wTransBuff);
		// the convolution buffer
		fn.weightCoef2 = helpers::getRawPointer(this->m_wTransBuff) 
		                 + m_wTransBuffShiftToDim; 
		fn.flagnegative= true;
		fn.tanhFactor  = NULL;
		    
		thrust::for_each(
				 thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0) + this->m_featureDim,
				 fn);
		
		}}

	    }else if (this->m_tanhReg == MDNARRMDN_CASECADECOMPLEX) {
		// AR with complex poles

		// step1. transform the parameter to the coefficients of 2nd order filter
		{{
			
		    // alpha = tanh(alpha), beta = sigmoid(beta)
		    // except the first filter when the AR order is an odd number
		    internal::ARComplexPolesParameterTransform fn;
		    fn.featDim      = this->m_featureDim;
		    fn.filterNum    = this->m_casOrder;
		    fn.withRealPole = this->m_casRealPole;
		    fn.paraMem      = this->m_weightsPtr;
		    fn.paraBuf      = helpers::getRawPointer(this->m_wTransBuff) 
			              + m_wTransBuffParaBK;
		    int n = this->m_featureDim * this->m_backOrder;
		    thrust::for_each(
				     thrust::counting_iterator<int>(0),
				     thrust::counting_iterator<int>(0) + n,
				     fn);
		}}
		
		// step2. convolution to get the coefficients for a classical AR filter
		{{
		    internal::TanhAutoRegConvolutionARComplex fn;
		    fn.casOrder    = this->m_casOrder;
		    fn.featureDim  = this->m_featureDim;
		    fn.weightPoles = helpers::getRawPointer(this->m_wTransBuff) 
			              + m_wTransBuffParaBK;

		    // the coefficient buffer
		    fn.weightCoef1 = helpers::getRawPointer(this->m_wTransBuff);

		    // the convolution buffer
		    fn.weightCoef2 = helpers::getRawPointer(this->m_wTransBuff) 
			             + m_wTransBuffShiftToDim; 
		    fn.flagnegative= true;
		    fn.tanhFactor  = NULL;
		    fn.aRealPole   = this->m_casRealPole;

		    thrust::for_each(
				     thrust::counting_iterator<int>(0),
				     thrust::counting_iterator<int>(0) + this->m_featureDim,
				     fn);
		}}
	    }
	    
	}else{
	    // no AR model
	}
    }

	
    template <typename TDevice>
    real_t MDNUnit_mixture_dyn<TDevice>::calculateError(real_vector &targets)
    {   
	
	// step1: calculate the (W_k^T (t-1)_n_d + b_i)
	{{
		
	     thrust::fill(this->m_dataBuff.begin(), this->m_dataBuff.end(), (real_t)0.0);
	     this->m_paral     = this->m_precedingLayer.parallelSequences();
	     this->m_totalTime = this->m_precedingLayer.curMaxSeqLength() * this->m_paral;

             #ifdef MIXTUREDYNDIAGONAL		
	     // Regressio on the time axis 
	     // one step to calculate wo_t1 + b, change the mean value
	     {{
		if (this->m_dynDirection == MDNUNIT_TYPE_1_DIRECT){
		    
		    for (int stepBack    = 1; stepBack <= this->m_backOrder; stepBack++){
			internal::ChangeMeanofMDN fn2;
			fn2.startDOut    = this->m_startDimOut;
			fn2.featureDim   = this->m_featureDim;
			fn2.layerSizeOut = this->m_layerSizeTar;
			fn2.mixNum       = this->m_numMixture;
			fn2.totalTime    = this->m_totalTime;
			fn2.tieVar       = this->m_tieVar;
			fn2.targets      = helpers::getRawPointer(targets);
		    
			// Block 1025x05
			if (this->m_tanhReg){
			    // the m_wTransBuff stores [1 a_1 ... a_N], 
			    //    where the AR is 1 - \sum_i^N a_i z^-1
			    fn2.linearPart = helpers::getRawPointer(this->m_wTransBuff) +
				             stepBack * this->m_featureDim;
			// AR classical form
			}else{
			    fn2.linearPart = this->m_weightsPtr +
				             (stepBack-1) * this->m_featureDim;
			}
		    
			fn2.biasPart     = this->m_weightsPtr + 
			                   this->m_backOrder*this->m_featureDim;
			
			fn2.mdnPara      = helpers::getRawPointer(this->m_paraVec);
			fn2.stepBack     = stepBack;
			
			fn2.trainableAPos= -1;   // this is useful for mxiture_dynSqr
			fn2.trainableBPos= -1;   // this is useful for mxiture_dynSqr
		
			int n =  this->m_totalTime * this->m_numMixture * this->m_featureDim;
			thrust::for_each(thrust::counting_iterator<int>(0),
					 thrust::counting_iterator<int>(0)+n,
					 fn2);
		    }
		}
	    }}

	    // Block 1025x04
	    // Obsolete for ARRMDN n dimension axis
	     
	     
            #else
	    // This block is used for AR model based on matrix transformation
	    //  x - Ax_t-1 - b
	    // Block dustbin.txt/0824x01
            #endif
	}}
	
	// Calculate the Error, just directly call the conventional methods
	return this->MDNUnit_mixture<TDevice>::calculateError(targets);
	
    }                
    
    template <typename TDevice>
    void MDNUnit_mixture_dyn<TDevice>::computeBackward(real_vector &targets)
    {   
	// The same as the conventional methods
	this->MDNUnit_mixture<TDevice>::computeBackward(targets);
	
	#ifdef MIXTUREDYNDIAGONAL
	{{
	    // Graidents computation:
	    //
	    // Step1: calculate the gradient for \delta{L}/\delta{a_k}, where
	    //        a_k is the coefficients of classical AR model 1-\sum_k{a_k}z^{-k}
	    // Step2: sum gradients over time and mixtures
	    // Step3: optionally, get the gradients w.r.t poles of casecade AR model
		
	    // gradient for the AR on time axis
	    if (this->m_dynDirection == MDNUNIT_TYPE_1_DIRECT && this->m_backOrder > 0){

		// step.1 calculate the gradient w.r.t coefficients of classical AR model
		// \delta{L}/\delta{a_k} = 
		// \sum_t^{T}\sum_m_^{M}phi(t,m)(o_t,d - \hat{mu}_t,d^m) / var_t,d^{m}^2 * o_t-k,d
		// t: time
		// m: mixture
		// k: k-th order of AR model
		// d: dimension
		// phi(t,m): posterior of mixture at time t
		// \hat{mu}: the mean transformed by AR

		// (obsolete implementation for Tanh AR)
		// Block 1025x9
		
		// for this->m_tanhReg and !this->m_tanhReg
		// accumulating the graidents is necessary
		internal::AccumulateGradient fn2;
		
		fn2.featureDim   = this->m_featureDim;
		fn2.mixNum       = this->m_numMixture;
		fn2.totalTime    = this->m_totalTime;
		fn2.startDOut    = this->m_startDimOut;
		fn2.layerSizeOut = this->m_layerSizeTar;
		    
		fn2.gradBuf      = helpers::getRawPointer(this->m_dataBuff);
		fn2.target       = helpers::getRawPointer(targets);
		fn2.mdnPara      = helpers::getRawPointer(this->m_paraVec);
		fn2.postPbuff    = helpers::getRawPointer(this->m_tmpPat);
		fn2.tieVar       = this->m_tieVar;
		fn2.backOrder    = this->m_backOrder;
		
		int n =  this->m_backOrder  * this->m_totalTime * 
		    this->m_numMixture * this->m_featureDim;
		thrust::for_each(thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0)+n,
				 fn2);
		
		// step2 update the gradients over time and mixtures
		/*************************************************
		 * Why did I use 1.0 / this->m_numMixture * this->m_totalTime ?
		 * But it turns out this learning rate works better !
		 *************************************************/
		// thrust::fill(this->m_oneVec.begin(), this->m_oneVec.end(), 1.0);
		if (this->m_arrmdnLearning == 1){
		    thrust::fill(this->m_oneVec.begin(), this->m_oneVec.end(),1.0);
		}else if (this->m_arrmdnLearning == 2){
		    thrust::fill(this->m_oneVec.begin(), this->m_oneVec.end(), 
				 1.0/this->m_numMixture / this->m_totalTime);
		}else{
		    thrust::fill(this->m_oneVec.begin(), this->m_oneVec.end(), 
				 1.0/this->m_numMixture * this->m_totalTime);
		} 
		
		helpers::Matrix<TDevice> onevec  (&this->m_oneVec, 
						  this->m_numMixture * this->m_totalTime, 
						  1);
		helpers::Matrix<TDevice> diffW   (&this->m_dataBuff, 
						  this->m_featureDim * this->m_backOrder,
						  this->m_totalTime * this->m_numMixture);
		helpers::Matrix<TDevice> gradW   (this->m_weightUpdates, 
						  this->m_featureDim * this->m_backOrder,
						  1,
						  this->m_weightStart
						  );
		// sum the gradients for a_k
		gradW.assignProduct(diffW, false, onevec, false);
		
		/******************* FATAL ERROR ******************
		 *  Remember to shift the gradb
		 **************************************************/
		// point to the weightUpdates of bias part
		helpers::Matrix<TDevice> diffB   (&this->m_dataBuff, 
						  this->m_featureDim,
						  this->m_totalTime * this->m_numMixture,
						  this->m_totalTime * this->m_numMixture *
						  this->m_backOrder * this->m_featureDim);
		helpers::Matrix<TDevice> gradb   (this->m_weightUpdates, 
						  this->m_featureDim, 
						  1,
						  this->m_weightStart + 
						  this->m_featureDim * this->m_backOrder
						  );
		// sum the gradients for the bias term
		gradb.assignProduct(diffB, false, onevec, false);

		
		// step3 
		// calculate the gradients w.r.t the poles of casecade AR
		if (this->m_tanhReg && this->m_backOrder >= 1){
   
		    // for AR with real poles
		    if (this->m_tanhReg == MDNARRMDN_CASECADEREAL){
			//   step1. prepare the weights, copy the poles to buffer
			//   The same procedure can be used for AR complex/real
			{{
			     // for AR real pole, each filter has only one coefficients
			     // for AR complex pole, it is two
			    
			    int temp      = ((this->m_backOrder==this->m_casOrder)?1:2);
			
			    internal::TanhAutoRegGradientPre fn;
			    fn.featureDim = this->m_featureDim;
			    fn.backOrder  = this->m_backOrder;
			    fn.weightPtr  = this->m_weightsPtr;
			    fn.weightBuff = helpers::getRawPointer(this->m_wTransBuff) +
				m_weightShiftToDim;
			    fn.coefNum    = temp;
			    
			    int n = this->m_featureDim*(this->m_backOrder-temp)*this->m_casOrder;
			    
			    // if backOrder > 1
			    if (n > 0){
				thrust::for_each(thrust::counting_iterator<int>(0),
						 thrust::counting_iterator<int>(0)+n,
						 fn);
			    }

			}}

			//   step2. convolution, get the gradients of classical coefficents
			//          to the poles
			{{
			    internal::TanhAutoRegConvolution fn;
			    fn.backOrder   = this->m_backOrder-1; // gradient !
			    fn.featureDim  = this->m_featureDim;  //
			    fn.flagnegative= false;
			
			    for (int order = 0; order<this->m_backOrder; order ++){
				fn.weightPoles = helpers::getRawPointer(this->m_wTransBuff) + 
				    m_weightShiftToDim + 
				    order * m_wTransBuffShiftToDim;
				// the coefficient buffer
				fn.weightCoef1 = helpers::getRawPointer(this->m_wTransBuff) + 
				    m_wTransBuffShiftToDim * 2 + 
				    2 * order * m_wTransBuffShiftToDim;
				// the convolution buffer
				fn.weightCoef2 = helpers::getRawPointer(this->m_wTransBuff) + 
				    m_wTransBuffShiftToDim * 3 + 
				    2 * order * m_wTransBuffShiftToDim;
				// 
				fn.tanhFactor  = this->m_weightsPtr + order * this->m_featureDim;
				thrust::for_each(thrust::counting_iterator<int>(0),
						 thrust::counting_iterator<int>(0)+
						 this->m_featureDim,
						 fn);
			    }			
			}}
			
			//   step3. get the gradients
			{{
			    internal::TanhAutoRegGradientTransform fn;
			    fn.featureDim  = this->m_featureDim;
			    fn.backOrder   = this->m_backOrder;
			    fn.rawGradients= this->m_weightUpdatesPtr;
			    fn.weightBuff  = helpers::getRawPointer(this->m_wTransBuff) + 
				m_wTransBuffShiftToDim * 2;
			    fn.tempResult  = helpers::getRawPointer(this->m_wTransBuff);
			
			    int n = this->m_featureDim * this->m_backOrder;
			    thrust::for_each(thrust::counting_iterator<int>(0),
					     thrust::counting_iterator<int>(0)+n,
					     fn);
			    // copy the tempResult back into gradient buffer
			    internal::CopySimple2 fn2;
			    fn2.Output     = this->m_weightUpdatesPtr;
			    fn2.in         = helpers::getRawPointer(this->m_wTransBuff);
			    thrust::for_each(thrust::counting_iterator<int>(0),
					     thrust::counting_iterator<int>(0) +
					     this->m_featureDim * this->m_backOrder,
					     fn2);
			}}

		    // AR with complex poles
		    }else if (this->m_tanhReg == MDNARRMDN_CASECADECOMPLEX){
			
			//   step1. prepare the weights, copy the poles to buffer
			//   The same procedure can be used for AR complex/real
			{{
			     // for AR real pole, each filter has only one coefficients
			     // for AR complex pole, it is two
			    
			    int temp      = ((this->m_backOrder==this->m_casOrder)?1:2);
			
			    internal::TanhAutoRegGradientPre fn;
			    fn.featureDim = this->m_featureDim;
			    fn.backOrder  = this->m_backOrder;
			    fn.weightPtr  = helpers::getRawPointer(this->m_wTransBuff) +
				            m_wTransBuffParaBK;
			    fn.weightBuff = helpers::getRawPointer(this->m_wTransBuff) +
				            m_weightShiftToDim;
			    fn.coefNum    = temp;
			    
			    int n = this->m_featureDim*(this->m_backOrder-temp)*this->m_casOrder;
			    
			    // if backOrder > 1
			    if (n > 0){
				thrust::for_each(thrust::counting_iterator<int>(0),
						 thrust::counting_iterator<int>(0)+n,
						 fn);
			    }

			}}

			
			//   step2. convolution, get the gradients of classical coefficents
			//          to the poles
			{{
			    internal::TanhAutoRegConvolutionARComplex fn;
			    fn.casOrder    = this->m_casOrder-1;
			    fn.featureDim  = this->m_featureDim;
			    fn.flagnegative= false;
			    fn.tanhFactor  = NULL;
			    for (int order = 0; order<this->m_casOrder; order ++){
				fn.aRealPole   = (this->m_casRealPole && order>0);
				fn.weightPoles = helpers::getRawPointer(this->m_wTransBuff) + 
				    m_weightShiftToDim + 
				    order * m_wTransBuffShiftToDim;
				// the coefficient buffer
				fn.weightCoef1 = helpers::getRawPointer(this->m_wTransBuff) + 
				    m_wTransBuffShiftToDim * 2 + 
				    2 * order * m_wTransBuffShiftToDim;
				// the convolution buffer
				fn.weightCoef2 = helpers::getRawPointer(this->m_wTransBuff) + 
				    m_wTransBuffShiftToDim * 3 + 
				    2 * order * m_wTransBuffShiftToDim;
				 				
				thrust::for_each(thrust::counting_iterator<int>(0),
						 thrust::counting_iterator<int>(0)+
						 this->m_featureDim,
						 fn);
			    }			
			}}

			//   step3. get the gradients
			{{
			    internal::TanhAutoRegGradientTransformARComplex fn;
			    fn.featureDim  = this->m_featureDim;
			    fn.backOrder   = this->m_backOrder;
			    
			    fn.rawGradients= this->m_weightUpdatesPtr;
			    
			    fn.weightBuff  = helpers::getRawPointer(this->m_wTransBuff) + 
				m_wTransBuffShiftToDim * 2; 
			    
			    fn.alphabeta   = helpers::getRawPointer(this->m_wTransBuff) +
				             m_wTransBuffParaBK;

			    fn.tempResult  = helpers::getRawPointer(this->m_wTransBuff);
			    
			    fn.aRealPole   = this->m_casRealPole;
				
			    int n = this->m_featureDim * this->m_casOrder;
			    thrust::for_each(thrust::counting_iterator<int>(0),
					     thrust::counting_iterator<int>(0)+n,
					     fn);
			    
			    // copy the tempResult back into gradient buffer
			    internal::CopySimple2 fn2;
			    fn2.Output     = this->m_weightUpdatesPtr;
			    fn2.in         = helpers::getRawPointer(this->m_wTransBuff);
			    thrust::for_each(thrust::counting_iterator<int>(0),
					     thrust::counting_iterator<int>(0) +
					     this->m_featureDim * this->m_backOrder,
					     fn2);
			    			    			    
			}}
		    }
		    
		// update the classical form AR order by order
		}else if (this->m_tanhReg == 0 && this->m_arrmdnUpInter > 0){
		    
		    int keepGradient = ((this->m_currTrainingEpoch-1)/this->m_arrmdnUpInter);
		    if (keepGradient < this->m_backOrder){
			// otherwise, update the specified order
			{{
			    internal::ARRMDNGradientsSetZero fn;
			    fn.keepGradient = keepGradient;
			    fn.featureDim   = this->m_featureDim;
			    fn.gradients    = this->m_weightUpdatesPtr;
			    thrust::for_each(thrust::counting_iterator<int>(0),
					     thrust::counting_iterator<int>(0) +
					     this->m_featureDim * this->m_backOrder,
					     fn);
			}}
		    }else{
			// if all coefficients have been updated for one time, just update all
		    }
		}	
	    }

	    // Block 1025x07
	    // no longer support the ARRMDN on dimension axis
	
	}}
	#else
	// ./dustbin.txt/Block 0824x02
	#endif
    }
    
    template <typename TDevice>
    void MDNUnit_mixture_dyn<TDevice>::getEMOutput(const real_t para, real_vector &targets)
    {
	// ??? to be implemented
	printf("To be implemented\n");
    }

    template <typename TDevice>
    void MDNUnit_mixture_dyn<TDevice>::getOutput(const real_t para,real_vector &targets)
    {
	
	int time                = this->m_precedingLayer.curMaxSeqLength();
	int datapointerperFrame = ((this->m_endDimOut - this->m_startDimOut) * 
				   this->m_precedingLayer.parallelSequences());
	int datapoint           = time * datapointerperFrame;
	
	/* ************* FATAL ERROR  ************************ //
	 *  Rember to initialize everthing (they shouldn't be here)
	 *  There should be a loadSequence() for all MDNUnits  
	 *  ************************************************* */
	this->m_paral     = this->m_precedingLayer.parallelSequences();
	this->m_totalTime = this->m_precedingLayer.curMaxSeqLength() * this->m_paral;

	
	// initialize the random number
	Cpu::real_vector tempRandom(datapoint, 0.0);
	real_vector randomSeedBuff;
	
	const Configuration &config = Configuration::instance();
	static boost::mt19937 *gen = NULL;
	if (!gen) {
	    gen = new boost::mt19937;
	    gen->seed(config.randomSeed());
	}
	boost::random::normal_distribution<real_t> dist(0, 1);
	for (size_t i = 0; i < datapoint; ++i)
	    tempRandom[i] = (dist(*gen));
	randomSeedBuff = tempRandom;	
	
	
	// get the MDN parameter 
	int startPos = 0;
	int endPos   = 0;
	for (int i = 0; i < time; i++)
	{{
	    
	    // Step1. for each time step, change the mean of the distribution
	    // Note:
	    //    computeForwardPass() is conducted before getOutput()
	    //    hence, no need to transform the tanh(alpha) to AR parameter
	    
	    if (this->m_dynDirection == MDNUNIT_TYPE_1_DIRECT || 
		this->m_dynDirection == MDNUNIT_TYPE_1_DIRECB ){
		
		// AR along the time axis
		for (int stepBack = 1; stepBack <= this->m_backOrder; stepBack++){
		    
                    #ifdef MIXTUREDYNDIAGONAL
		    if (i >= stepBack){    
			// one step to calculate wo_t1 + b, change the mean value
			internal::ChangeMeanofMDN fn2;
			fn2.startDOut    = this->m_startDimOut;
			fn2.featureDim   = this->m_featureDim;
			fn2.layerSizeOut = this->m_layerSizeTar;
			fn2.mixNum       = this->m_numMixture;
			fn2.totalTime    = this->m_totalTime;
			fn2.targets      = helpers::getRawPointer(targets);
			
			/*if (this->m_tanhReg && this->m_backOrder < 0){
			    
			  if (this->m_tanhReg && this->m_backOrder < 3){
			       
			  fn2.linearPart= helpers::getRawPointer(this->m_wTransBuff) + 
			  (stepBack - 1 + 2) * this->m_featureDim;
			  }else */
			      
		       if(this->m_tanhReg){
			    fn2.linearPart= helpers::getRawPointer(this->m_wTransBuff) + 
				stepBack * this->m_featureDim;
			}else{
			    fn2.linearPart= this->m_weightsPtr+(stepBack-1)*this->m_featureDim;
			}

			fn2.biasPart = this->m_weightsPtr+this->m_featureDim*this->m_backOrder;    
			fn2.mdnPara      = helpers::getRawPointer(this->m_paraVec);
			fn2.stepBack     = stepBack;
			
			fn2.trainableAPos= -1;   // this is not useful for mxiture_dynSqr
			fn2.trainableBPos= -1;   // this is not useful for mxiture_dynSqr

			startPos     = i     * this->m_numMixture * this->m_featureDim;
			endPos       = (i+1) * this->m_numMixture * this->m_featureDim;
		    
			thrust::for_each(thrust::counting_iterator<int>(0) + startPos,
					 thrust::counting_iterator<int>(0) + endPos,
					 fn2);
			/*
			  int startDOut    = this->m_startDimOut;
			  int featureDim   = this->m_featureDim;
			  int layerSizeOut = this->m_layerSizeTar;
			  int mixNum       = this->m_numMixture;
			  int totalTime    = this->m_totalTime;
			  Cpu::real_vector paraVec2 = this->m_paraVec;
			  Cpu::real_vector mWeight = *this->m_weights;
			  Cpu::real_vector targetVec = targets;
			  for (int idx = startPos; idx < endPos; idx++){
			  int temp      = idx % (featureDim * mixNum); 
			  int featIndex = temp % featureDim; 
			  int timeStep  = idx / (featureDim * mixNum);
			  int mixIndex  = temp / featureDim;
			  int pos_mean, pos_data;
			  pos_mean = (totalTime * mixNum + 
			  timeStep  * featureDim * mixNum + 
			  mixIndex  * featureDim + featIndex); 
			  pos_data = ((timeStep-1)  * layerSizeOut) + startDOut + featIndex;
			  real_t tmpResult = paraVec2[pos_mean] + 
			  mWeight[this->m_weightStart + featIndex] * targetVec[pos_data] +
			  mWeight[this->m_weightStart + this->m_featureDim + featIndex];
			  printf("%f%f%f\t",paraVec2[timeStep*mixNum],paraVec2[timeStep*mixNum+1],
			  paraVec2[timeStep*mixNum]+paraVec2[timeStep*mixNum+1]);
			  //printf("%f\t", tmpResult);
			}*/
		    }
		    #else
		    // Block 0824x03
		    #endif
		}
	    }
	    
	    // Block 1025x11
	    // AR along the dimension axis
	    // No longer supported

	    // Step2. Sampling
	    internal::SamplingMixture fn;
	    fn.featureDim   = this->m_featureDim;
	    fn.layerSizeOut = this->m_layerSizeTar;
	    fn.startDOut    = this->m_startDimOut;
	    fn.mixtureNum   = this->m_numMixture;
	    fn.totalTime    = time * this->m_precedingLayer.parallelSequences();
	    fn.para         = para;
	    fn.paraPtr      = ( (this->m_varScale.size()>0) ?
				(helpers::getRawPointer(this->m_varScale)) : NULL );
	    fn.targets      = helpers::getRawPointer(targets);
	    fn.mdnPara      = helpers::getRawPointer(this->m_paraVec);
	    fn.tieVar       = this->m_tieVar;

	    startPos    = i     * datapointerperFrame;
	    endPos      = (i+1) * datapointerperFrame;
	    thrust::for_each(
		  thrust::make_zip_iterator(
			thrust::make_tuple(randomSeedBuff.begin() + startPos, 
					   thrust::counting_iterator<int>(0)+ startPos)),
		  thrust::make_zip_iterator(
			thrust::make_tuple(randomSeedBuff.begin() + endPos, 
					   thrust::counting_iterator<int>(0)+endPos)),
		  fn);
	}}	
	
    }

    
    template <typename TDevice>
    void MDNUnit_mixture_dyn<TDevice>::getParameter(real_t *targets)
    {
	// Step1. generate (but now we have to specify the MDN parameter)
	{{
		if (this->m_varScale.size()==0){
		    printf("\n\tNote: to get the parameter sequence based on sampling\n");
		    printf("\t you should provide the vector to scale the variance");
		    printf("\t (--mdn_sampleParaVec)\n");
		    printf("\tNow, the parameter will be sampled with ratio=0.01.");
		}else{
		    printf("\n\tGenerating the parameter sequence based on sampling\n");
		    
		}
		real_vector tmpOutput(this->m_featureDim * 
				      this->m_precedingLayer.maxSeqLength() * 
				      this->m_precedingLayer.parallelSequences(), 0.0);
		this->getOutput(0.01, tmpOutput);
		
	}}

	// Step2. get the parameters
	{{
		internal::GetParameterMixture fn;
		int time = this->m_precedingLayer.curMaxSeqLength();
		time     = time*this->m_precedingLayer.parallelSequences();

		fn.targets      = targets;
		fn.featureDim   = this->m_featureDim;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.startDimIn   = this->m_startDim;
		fn.mixtureNum   = this->m_numMixture;
		fn.totalTime    = time;
		fn.targets      = targets;
		fn.mdnPara      = helpers::getRawPointer(this->m_paraVec);
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.tieVar       = this->m_tieVar;

		thrust::for_each(thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0)+time,
				 fn);
	}}
    }


    /************************************************************
     * MDNUnit_mixture_dynSqr 
     *
     ************************************************************/
    template <typename TDevice>
    MDNUnit_mixture_dynSqr<TDevice>::MDNUnit_mixture_dynSqr(
	int startDim,        int endDim, int startDimOut, int endDimOut,     int type, 
	Layer<TDevice> &precedingLayer, int outputSize,  const bool tieVar, const int backOrder,
	const int trainable)
        : MDNUnit_mixture<TDevice>(startDim, endDim, startDimOut, endDimOut, 
				   type, (endDimOut - startDimOut), 
				   precedingLayer, outputSize, tieVar, trainable)
    {
	m_backOrder   = backOrder;
	m_a_pos       = type * (1 + this->m_featureDim + (tieVar ? 1 : this->m_featureDim));
	m_b_pos       = m_a_pos + backOrder * this->m_featureDim;
	
	const Configuration &config = Configuration::instance();
	m_tanhReg     = 1; //config.tanhAutoregressive();
	if (m_backOrder > 2 && m_tanhReg){
	    printf("Tanh Autoregressive is not implemented for step order > 2");
	    m_tanhReg  = 0;
	}
	
    }

    template <typename TDevice>
    MDNUnit_mixture_dynSqr<TDevice>::~MDNUnit_mixture_dynSqr()
    {
	
    }
    
    template <typename TDevice>
    bool MDNUnit_mixture_dynSqr<TDevice>::flagValid()
    {
	
    	bool tempFlag (this->m_a_pos < this->m_endDim && this->m_b_pos < this->m_endDim);
	int tempFeatureDim = (this->m_endDim - this->m_startDim - this->m_numMixture - 
			      this->m_featureDim * (this->m_backOrder + 1));
	// get the dimension for mean and variance
	tempFeatureDim = this->m_tieVar ? 
	    ((tempFeatureDim - this->m_numMixture)/this->m_numMixture):
	    ((tempFeatureDim)/2/this->m_numMixture);
	// 
	tempFlag = tempFlag && (tempFeatureDim == this->m_featureDim) && 
	    (this->m_featureDim == (this->m_endDimOut - this->m_startDimOut));
	if (!tempFlag){
	    printf("\tMixture dynamic unit\t");
	}
	return tempFlag;
    }
    
    template <typename TDevice>
    void MDNUnit_mixture_dynSqr<TDevice>::computeForward()
    {   
	// do the normal feedforward for the MDN part
	this->MDNUnit_mixture<TDevice>::computeForward();
	
	// do the a part
	// the mean part (unnessary to change anything. But need to copy from NN output to MDN)
	if (this->m_tanhReg && this->m_backOrder < 3){
	    
	    
	    internal::TanhAutoRegWeightTime fn1;
	    
	    fn1.backOrder    = this->m_backOrder;
	    fn1.featureDim   = this->m_featureDim;
	    fn1.NNOutputSize = this->m_precedingLayer.size();
	    fn1.startD       = this->m_startDim + this->m_a_pos;
	    
	    fn1.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
	    fn1.NNOutput     = helpers::getRawPointer(this->m_precedingLayer.outputs());

	    // n:  number of timesteps
	    // n2: number of the mean parameters  (time * featDim)
	    int n  = this->m_precedingLayer.curMaxSeqLength();
	    n      = n * this->m_precedingLayer.parallelSequences();
	    int n2 = n * this->m_featureDim * this->m_backOrder;
	    
	    fn1.weightOut    = helpers::getRawPointer(this->m_paraVec) + n * this->m_a_pos;
	    thrust::for_each(thrust::counting_iterator<int>(0),
			     thrust::counting_iterator<int>(0)+n2,
			     fn1);
	    	    
	}else{
	    {{
		internal::CopyMean fn;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.featureDim   = this->m_featureDim * this->m_backOrder;
		fn.startD       = this->m_startDim + this->m_a_pos;
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNOutput     = helpers::getRawPointer(this->m_precedingLayer.outputs());

		// n:  number of timesteps
		// n2: number of the mean parameters  (time * featDim)
		int n  = this->m_precedingLayer.curMaxSeqLength();
		n  = n * this->m_precedingLayer.parallelSequences();
		int n2 = n * this->m_featureDim * this->m_backOrder;
		
		thrust::for_each(
		    thrust::make_zip_iterator(
			  thrust::make_tuple(this->m_paraVec.begin()+ n * this->m_a_pos,
					     thrust::counting_iterator<int>(0))),
		    thrust::make_zip_iterator(
                          thrust::make_tuple(this->m_paraVec.begin()+ n * this->m_a_pos + n2, 
					     thrust::counting_iterator<int>(0)+n2)),
		    fn);
	   }}
	}
	
	// copy the b part
	{{
		internal::CopyMean fn;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.featureDim   = this->m_featureDim;
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNOutput     = helpers::getRawPointer(this->m_precedingLayer.outputs());
		fn.startD       = this->m_startDim + this->m_b_pos;
		
		// n:  number of timesteps
		// n2: number of the mean parameters  (time * featDim)
		int n  = this->m_precedingLayer.curMaxSeqLength();
		    n  = n * this->m_precedingLayer.parallelSequences();
		int n2 = n * this->m_featureDim;

		thrust::for_each(
		    thrust::make_zip_iterator(
			  thrust::make_tuple(this->m_paraVec.begin()+ n * this->m_b_pos,
					     thrust::counting_iterator<int>(0))),
		    thrust::make_zip_iterator(
                          thrust::make_tuple(this->m_paraVec.begin()+ n * this->m_b_pos + n2, 
					     thrust::counting_iterator<int>(0)+n2)),
		    fn);	
	}}
	/*
	// n:  number of timesteps
	// n2: number of the mean parameters  (time * featDim)
	int n  = this->m_precedingLayer.curMaxSeqLength();
	n  = n * this->m_precedingLayer.parallelSequences();
	int n2 = n * this->m_featureDim;
	thrust::fill(this->m_paraVec.begin()+ n  * this->m_a_pos,
		     this->m_paraVec.begin()+ n  * this->m_b_pos + n2,
		     0.0);*/

    }

    template <typename TDevice>
    void MDNUnit_mixture_dynSqr<TDevice>::computeBackward(real_vector &targets)
    {                
	// step1: 
	// The same as the conventional methods
	this->MDNUnit_mixture<TDevice>::computeBackward(targets);
	
	// step2:
	// Accumulate gradients for the trainable A and B
	// A: for each time step * dim, \sum_mixture gradients_for_\mu * target
	// b: for each time step * dim, \sum_mixture gradients_for_\mu
	{{
		internal::ComputeBPAccumForMixtureDynSqr fn;
		fn.layerSize    = this->m_precedingLayer.size();
		fn.startD       = this->m_startDim;
		fn.startDOut    = this->m_startDimOut;
		fn.layerSizeOut = this->m_layerSizeTar;
		fn.featureDim   = this->m_featureDim;
		fn.mixtureNum   = this->m_numMixture;
		fn.backOrder    = this->m_backOrder;
		
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.errors       = helpers::getRawPointer(this->m_precedingLayer.outputErrors());
		fn.outputs      = helpers::getRawPointer(this->m_precedingLayer.outputs());
		fn.targets      = helpers::getRawPointer(targets);
		fn.trainableAPos= this->m_a_pos;  // start dimension for the trainable A 
		fn.trainableBPos= this->m_b_pos;  // start dimension for the trainable B
		
		fn.paraVec      = helpers::getRawPointer(this->m_paraVec);
		fn.totalTime    = (this->m_precedingLayer.curMaxSeqLength() *
				   this->m_precedingLayer.parallelSequences());
		
		fn.tanh         = this->m_tanhReg;
		
		int n = fn.totalTime * this->m_featureDim * this->m_backOrder;
		
		thrust::for_each(thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0)+n,
				 fn);
		/*
		Cpu::real_vector tempVec = this->m_precedingLayer.outputErrors();
		for (int idx = 0; idx < n; idx++){
		    int featIndex = idx % fn.featureDim;
		    int timeStep  = idx / fn.featureDim;
		    int timePos   = timeStep * fn.layerSize + fn.startD;
	    
		    // pointer to the target data
		    int pos_data  = (timeStep - 1) * fn.layerSizeOut + fn.startDOut + featIndex; 
		    int pos_mean  = timePos + fn.mixtureNum    + featIndex; 
		    int pos_a     = timePos + fn.trainableAPos + featIndex; 
		    int pos_b     = timePos + fn.trainableBPos + featIndex; 
		    printf("%f-%f\t", tempVec[pos_a], tempVec[pos_b]);

		    }*/

	}}
    }

    template <typename TDevice>
    real_t MDNUnit_mixture_dynSqr<TDevice>::calculateError(real_vector &targets)
    {   
	// step1: calculate the (W_k^T (t-1)_n_d + b_i)
	{{
		
		int totalTime    = this->m_precedingLayer.parallelSequences() * 
		    this->m_precedingLayer.curMaxSeqLength();
		// one step to calculate wo_t1 + b, change the mean value
		for (int stepBack    = 1; stepBack <= this->m_backOrder; stepBack++){
		    internal::ChangeMeanofMDN fn2;
		    fn2.startDOut    = this->m_startDimOut;
		    fn2.featureDim   = this->m_featureDim;
		    fn2.layerSizeOut = this->m_layerSizeTar;
		    fn2.mixNum       = this->m_numMixture;
		    fn2.totalTime    = totalTime;
		    
		    fn2.targets      = helpers::getRawPointer(targets);
		    
		    fn2.linearPart   = NULL;
		    fn2.biasPart     = NULL;

		    fn2.mdnPara      = helpers::getRawPointer(this->m_paraVec);
		    fn2.trainableAPos= this->m_a_pos;
		    fn2.trainableBPos= this->m_b_pos;
		    fn2.tieVar       = this->m_tieVar;
		    fn2.stepBack     = stepBack;
		    int n =  totalTime * this->m_numMixture * this->m_featureDim;
		    thrust::for_each(thrust::counting_iterator<int>(0),
				     thrust::counting_iterator<int>(0)+n,
				     fn2);
		}
		/*
		Cpu::real_vector tempVec = this->m_paraVec;
		for (int idx = 0; idx < n; idx++){
		    int temp      = idx % (fn2.featureDim * fn2.mixNum); 
		    int featIndex = temp % fn2.featureDim; 
		    int timeStep  = idx / (fn2.featureDim * fn2.mixNum);
		    int mixIndex  = temp / fn2.featureDim;
		    int pos_a = fn2.totalTime * fn2.trainableAPos + 
			timeStep * fn2.featureDim + featIndex;
		    int pos_b = fn2.totalTime * fn2.trainableBPos + 
			timeStep * fn2.featureDim + featIndex;
		    printf("%f-%f\t", tempVec[pos_a], tempVec[pos_b]);

		    }*/
	}}
	// Step2: Calculate the Error
	// just directly call the conventional methods
	return this->MDNUnit_mixture<TDevice>::calculateError(targets);

    }

    template <typename TDevice>
    void MDNUnit_mixture_dynSqr<TDevice>::getOutput(const real_t para, real_vector &targets)
    {
	int time                = this->m_precedingLayer.curMaxSeqLength();
	int datapointerperFrame = ((this->m_endDimOut - this->m_startDimOut) * 
				   this->m_precedingLayer.parallelSequences());
	int datapoint           = time * datapointerperFrame;
	
	int totalTime           = (this->m_precedingLayer.parallelSequences() * 
				   this->m_precedingLayer.curMaxSeqLength());

	// initialize the random number
	Cpu::real_vector tempRandom(datapoint, 0.0);
	real_vector randomSeedBuff;
	
	const Configuration &config = Configuration::instance();
	static boost::mt19937 *gen = NULL;
	if (!gen) {
	    gen = new boost::mt19937;
	    gen->seed(config.randomSeed());
	}
	boost::random::normal_distribution<real_t> dist(0, 1);
	for (size_t i = 0; i < datapoint; ++i)
	    tempRandom[i] = (dist(*gen));
	randomSeedBuff = tempRandom;	
	

	// get the MDN parameter (can only do it frame by frame now)
	int startPos = 0;
	int endPos   = 0;

	Cpu::real_vector tempVec1 = this->m_paraVec;
	for (int i = 0; i < time; i++)
	{{
		for (int stepBack    = 1; stepBack <= this->m_backOrder; stepBack++){
		    if (i < stepBack) break;
		    
		    // one step to calculate wo_t1 + b, change the mean value
		    
		    internal::ChangeMeanofMDN fn2;
		    fn2.startDOut    = this->m_startDimOut;
		    fn2.featureDim   = this->m_featureDim;
		    fn2.layerSizeOut = this->m_layerSizeTar;
		    fn2.mixNum       = this->m_numMixture;
		    fn2.totalTime    = totalTime;
		
		    fn2.targets      = helpers::getRawPointer(targets);
		    fn2.mdnPara      = helpers::getRawPointer(this->m_paraVec);
		    fn2.trainableAPos= this->m_a_pos;
		    fn2.trainableBPos= this->m_b_pos;
		
		    fn2.linearPart   = NULL;
		    fn2.biasPart     = NULL;
		    fn2.stepBack     = stepBack;
		    startPos     = i     * this->m_numMixture * this->m_featureDim;
		    endPos       = (i+1) * this->m_numMixture * this->m_featureDim;
		    
		    thrust::for_each(thrust::counting_iterator<int>(0) + startPos,
				     thrust::counting_iterator<int>(0) + endPos,
				     fn2);
		}
		
		internal::SamplingMixture fn;
		fn.featureDim   = this->m_featureDim;
		fn.layerSizeOut = this->m_layerSizeTar;
		fn.startDOut    = this->m_startDimOut;
		fn.mixtureNum   = this->m_numMixture;
		fn.totalTime    = time * this->m_precedingLayer.parallelSequences();
		fn.para         = para;
		fn.paraPtr      = ( (this->m_varScale.size()>0) ?
				    (helpers::getRawPointer(this->m_varScale)) : NULL );
		fn.targets      = helpers::getRawPointer(targets);
		fn.mdnPara      = helpers::getRawPointer(this->m_paraVec);
		fn.tieVar       = this->m_tieVar;

		startPos    = i     * datapointerperFrame;
		endPos      = (i+1) * datapointerperFrame;
		thrust::for_each(
  			 thrust::make_zip_iterator(
			     thrust::make_tuple(randomSeedBuff.begin() + startPos, 
						thrust::counting_iterator<int>(0)+ startPos)),
		         thrust::make_zip_iterator(
			     thrust::make_tuple(randomSeedBuff.begin() + endPos, 
						thrust::counting_iterator<int>(0)+endPos)),
			 fn);
		
		Cpu::real_vector tempVec2 = this->m_paraVec;
		//printf("");
	}}

    }
   
    template <typename TDevice>
    void MDNUnit_mixture_dynSqr<TDevice>::getEMOutput(const real_t para, real_vector &targets)
    {
	// ??? to be implemented
	printf("To be implemented\n");
    }

    template <typename TDevice>
    void MDNUnit_mixture_dynSqr<TDevice>::getParameter(real_t *targets)
    {
	// Step1. generate (but now we have to specify the MDN parameter)
	{{
		if (this->m_varScale.size()==0){
		    printf("\n\tNote: to get the parameter sequence based on sampling\n");
		    printf("\t you should provide the vector to scale the variance");
		    printf("\t (--mdn_sampleParaVec)\n");
		    printf("\tBut now, the parameter predicted by NN will be output.");
		}else{
		    printf("\n\tGenerating the parameter sequence based on sampling\n");
		    real_vector tmpOutput(this->m_featureDim * 
					  this->m_precedingLayer.maxSeqLength() * 
					  this->m_precedingLayer.parallelSequences(), 0.0);
		    // this 0.01 just set the flag to generate the output
		    this->getOutput(0.01, tmpOutput); 
		}
	}}

	// Step2. get the parameters
	{{
		internal::GetParameterMixtureDynSqr fn;
		int time = this->m_precedingLayer.curMaxSeqLength();
		time     = time*this->m_precedingLayer.parallelSequences();

		fn.targets      = targets;
		fn.featureDim   = this->m_featureDim;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.startDimIn   = this->m_startDim;
		fn.mixtureNum   = this->m_numMixture;
		fn.totalTime    = time;
		fn.targets      = targets;
		fn.mdnPara      = helpers::getRawPointer(this->m_paraVec);
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.tieVar       = this->m_tieVar;
		fn.trainableAPos= this->m_a_pos;
		fn.trainableBPos= this->m_b_pos;
		thrust::for_each(thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0)+time,
				 fn);
	}}

    }


    template <typename TDevice>
    void MDNUnit_mixture_dynSqr<TDevice>::initPreOutput(
		const MDNUnit_mixture_dynSqr<TDevice>::cpu_real_vector &mVec, 
		const MDNUnit_mixture_dynSqr<TDevice>::cpu_real_vector &vVec)
    {
	this->MDNUnit_mixture<TDevice>::initPreOutput(mVec, vVec);
	
	
	Layer<TDevice> *tmpPtr = &this->m_precedingLayer;
	TrainableLayer<TDevice> *tLayer = dynamic_cast<layers::TrainableLayer<TDevice>*>(tmpPtr);
	if (tLayer){

	    thrust::fill(tLayer->weights().begin() + 
			 (this->m_startDim + this->m_a_pos)*tLayer->precedingLayer().size(),
			 tLayer->weights().begin() +
			 (this->m_startDim + this->m_a_pos + 2*this->m_featureDim) * 
			 tLayer->precedingLayer().size(),
			 0.0);

	    
	    int biasPos = tLayer->size()*tLayer->precedingLayer().size();
	    if ((tLayer->weights().size() - biasPos) < (this->m_endDim)){
		throw std::runtime_error("Impossible error in initializing MDNUnit_mixture_dynSqr");
	    }
	    thrust::fill(tLayer->weights().begin()+ biasPos + this->m_startDim + this->m_a_pos, 
			 tLayer->weights().begin()+ biasPos + this->m_startDim + this->m_a_pos + 
			 2 * this->m_featureDim,
			 0.0);
			 }
    }

    
    template class MDNUnit<Cpu>;
    template class MDNUnit<Gpu>;
    template class MDNUnit_sigmoid<Cpu>;
    template class MDNUnit_sigmoid<Gpu>;
    template class MDNUnit_mixture<Cpu>;
    template class MDNUnit_mixture<Gpu>;
    template class MDNUnit_softmax<Cpu>;
    template class MDNUnit_softmax<Gpu>;
    template class MDNUnit_mixture_dyn<Cpu>;
    template class MDNUnit_mixture_dyn<Gpu>;
    template class MDNUnit_mixture_dynSqr<Cpu>;
    template class MDNUnit_mixture_dynSqr<Gpu>;


} // namespace layers

