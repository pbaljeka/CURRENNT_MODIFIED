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

#include "MDNLayer.hpp"
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


//#define DEBUG_LOCAL 1
//#define ALTER_TIEVAR 1


namespace internal {
namespace {
    
    struct CopySimple
    {

	real_t *Output; // output address to store data
	const char *patTypes;

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

    // Definition for the operation involved
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
	    // The data is not continuous, thus, we have to 
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
        int NNOutputSize;
	int startD;
	int endD;
        const real_t *NNoutputs;
        const char *patTypes;

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

	    // bias to the start of one frame
            const real_t *offOutputs = &NNoutputs[patIdx * NNOutputSize + startD];

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
	int NNOutputSize;
	int startD;
	int endD;
	const real_t *offset;
	const char *patTypes;
	const real_t *NNOutput;

	__host__ __device__ real_t operator() (const int &outputIdx) const
	{
	    // layer size
	    // The data is not continuous, thus, we have to 
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
	int NNOutputSize;
	int startD;
	int endD;
	real_t varFloor;
	const char   *patTypes;
	const real_t *NNOutput;

	__host__ __device__ real_t operator() (const int &outputIdx) const
	{
	    // layer size
	    // The data is not continuous, thus, we have to 
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
        int layerSize;
        const real_t *outputs;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            int patIdx = t.get<1>();

            // check if the pattern belongs to a sequence
            if (t.get<0>() == SKIP_MARKER)
                return;

            // sum up the outputs
            const real_t *offOutputs = &outputs[patIdx * layerSize];

            real_t sum = 0;
            for (int i = 0; i < layerSize; ++i)
                sum += offOutputs[i];

            // store the result
            t.get<0>() = sum;
        }
    };

    struct NormalizeOutputsFn
    {
        int layerSize;

        const real_t *normFacts;

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
        int NNOutputSize;
	int featureDim;
	int startD;
	const real_t *NNOutput; // output or NN
	const char *patTypes;

        __host__ __device__ void operator() (const thrust::tuple<real_t&, int> &t) const
        {
            // unpack the tuple
            int outputIdx = t.get<1>();

            // calculate the pattern index
            const int timeStep = outputIdx / featureDim;
	    const int dimStep  = (outputIdx % featureDim) + startD;

	    if (patTypes[timeStep] == PATTYPE_NONE)
                return;
            // store the result
            t.get<0>() = *(NNOutput + timeStep * NNOutputSize + dimStep);
        }
    };
    
    // Definition for calculate Errors
    struct ComputeSigmoidError
    {

	int startD;
	int endD;
	int startDOut;
	int layerSizeOut;
	const char *patTypes;
	const real_t *targets;  // targets data
	
	// from 1 to timesteps * para_dim
	__host__ __device__ real_t operator() (const thrust::tuple<real_t, int> &t) const
	{
	    
	    int outputIdx = t.get<1>();  // index
	    real_t prob   = t.get<0>();  // output of NN (probability)

	    // layer size
	    // The data is not continuous, thus, we have to 
	    const int timeStep = outputIdx / (endD - startD);
	    const int dimStep  = (outputIdx % (endD - startD)) + startDOut;
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return 0;
	    
	    // target data
	    const real_t *data = targets + (layerSizeOut * timeStep) + dimStep;
	    
	    // 
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

	    // layer size
	    // The data is not continuous, thus, we have to 
	    const int timeStep = outputIdx / (endD - startD);
	    const int dimStep  = (outputIdx % (endD - startD));
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;
	    
	    // target data
	    const real_t *data = targets + (layerSizeOut * timeStep) + dimStep + startDOut;
	    
	    //
	    const int pos_error= layerSizeIn * timeStep + dimStep + startD;
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
                long int Idx    = outputIdx * layerSize + (long int)targetClass;
                real_t targetProb = helpers::max(helpers::NumericLimits<real_t>::min(), 
						 prob[Idx]);
                return -1*log(targetProb);
            }
        }
    };


    struct ComputeMixtureDistance
    {
	int startDOut;
	int layerSizeOut;
	int mixture_num;
	int featureDim;
	int totaltime;
	const char *patTypes;
	const real_t *output;   // targets data
	const real_t *mdnPara;  // mean value of the mixture
	
	// from 1 to timesteps * num_mixture
	__host__ __device__ real_t operator() (const int idx) const
	{
	    
	    int timeStep = idx / mixture_num; //t.get<0>();
	    int mixIndex = idx % mixture_num; //t.get<1>(); 
	    
	    // point to the targets data
	    int pos_data = (layerSizeOut * timeStep)+startDOut;
		
	    const real_t *data, *mean, *var;

	    if (patTypes[timeStep] == PATTYPE_NONE)
		return 0;
	    
	    // point to the mixture data (mean)
	    int pos =  totaltime * mixture_num;
	    int pos_mean = pos+timeStep*featureDim*mixture_num+mixIndex*featureDim;
	    pos     =  totaltime * (mixture_num + mixture_num * featureDim);

#ifdef ALTER_TIEVAR
	    int pos_var  = pos+timeStep*mixture_num;
#else
	    int pos_var  = pos+timeStep*mixture_num+mixIndex;
#endif
	    var  = mdnPara + pos_var;

	    real_t tmp = 0.0;
	    for (int i = 0; i<featureDim; i++){
		data = output  + pos_data + i;
		mean = mdnPara + pos_mean + i;
		tmp += (*data-*mean)*(*data-*mean)/((*var)*(*var))/2;
		
	    }
	    return tmp;
	}
    };


    struct ComputeMixtureError
    {
	int startD;
	int endD;
	int startDOut;
	int layerSizeOut;
	int mixture_num;
	int featureDim;

	const char *patTypes;
	real_t *meanDis; 
	real_t *mdnPara;   // mean value of the mixture
	int totalTime;   //

	// from 1 to timesteps
	__host__ __device__ real_t operator() (const int timeStep) const
	{
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return 0;
	    
	    real_t tmp = helpers::NumericLimits<real_t>::logZero();
	    real_t tmptmp;
	    real_t *meanDisPtr = meanDis + timeStep * mixture_num;
	    
	    int pos = timeStep*mixture_num;
	    const real_t *mixtureW = mdnPara + pos; // point to the weight
	    
	    pos     = totalTime*(mixture_num+mixture_num*featureDim);
	    real_t *var = mdnPara + pos + timeStep * mixture_num; 
	    
	    for (int i = 0; i<mixture_num; i++){
		//tmptmp = log(*(mixtureW+i))+(*(meanDisPtr+i))-featureDim/2*log(2*PI_DEFINITION);
		tmptmp = log(*(mixtureW+i))-(*(meanDisPtr+i))-featureDim/2*log(2*PI_DEFINITION);
#ifdef ALTER_TIEVAR
		tmptmp = tmptmp - featureDim*helpers::safeLog(*(var));
		for (int j = 1; j<mixture_num; j++)
		    *(var+j) = 0;
#else
		tmptmp = tmptmp - featureDim*helpers::safeLog(*(var+i));
#endif
		if (tmptmp < helpers::NumericLimits<real_t>::lSMALL())
		    tmptmp = helpers::NumericLimits<real_t>::lSMALL();
		
		//tmptmp = helpers::safeExp(tmptmp);
		tmp    = helpers::logAdd(tmp, tmptmp);
		// save  w_i p_i
		*(meanDisPtr+i) = tmptmp; 
	    }
	    // save sum_i^mixture_num w_i p_i
	    meanDisPtr = meanDis + totalTime * mixture_num + timeStep;
	    *meanDisPtr= tmp;
	    //return -1*helpers::safeLog(tmp);
	    return -1*(tmp);
	}
    };
    
    struct ComputeBPmixtureWeight
    {
	int mixture_num;
	int NNOutputSize;
	int startD;
	const char *patTypes;
	const real_t *meanDis; 
	const real_t *mdnPara; // mean value of the mixture
	real_t *errors;        // outputerrors of preceding layer
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
	    const real_t *sigma  = mdnPara + pos;

	    if (patTypes[timeStep] == PATTYPE_NONE){
	    
	    }else{
		// store the gradients
		pos = timeStep * NNOutputSize + startD + mixtureI;
		//*(errors + pos) = (*sigma) - (*postP)/(*sumPost);
		*(errors + pos) = (*sigma) - helpers::safeExp((*postP)-(*sumPost));
	    }
	}

    };

    struct ComputeBPmixtureMeanVariance
    {
	int layerSize;
	int startD;
	int startDOut;
	int layerSizeOut;
	int mixture_num;
	int featureDim;

	const char *patTypes;
	const real_t *meanDis; 
	const real_t *mdnPara; // mean value of the mixture
	const real_t *target;  // target data
	real_t *errors;        // outputerrors of preceding layer
	real_t *varBuff;

	int totalTime;   //

	// from 1 to timesteps * numMixture*(featureDim) 
	__host__ __device__ void operator() (const int outputIdx) const
	{
	    
	    const int timeStep = outputIdx / (mixture_num * featureDim);

	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;

	    const int tmp = outputIdx % (mixture_num * featureDim);
	    const int mixtureI = tmp / featureDim;
	    const int featureI = tmp % featureDim;
	    
	    // pointer to the mean gradient
	    int meanshift = timeStep*layerSize+startD+mixtureI*featureDim + featureI;
	    real_t *errorm = errors + meanshift;

	    // pointer to the variance gradient

#ifdef ALTER_TIEVAR
	    //int varshift2   = timeStep*layerSize+startD+mixture_num*featureDim;
#else
	    //int varshift2   = timeStep*layerSize+startD+mixture_num*featureDim + mixtureI;
#endif

	    int varshift   = timeStep*(mixture_num*featureDim)+mixtureI*featureDim+featureI;
	    real_t *errorv = varBuff + varshift;

	    //real_t *errorv2= errors  + varshift2;
	    
	    // pointer to the target data y
	    const real_t *tardata= target + timeStep*layerSizeOut + startDOut + featureI;

	    // pointer to the mean
	    meanshift= totalTime * mixture_num + 
		       timeStep * mixture_num * featureDim + 
		       mixtureI * featureDim + 
		       featureI;
#ifdef ALTER_TIEVAR
	    varshift = totalTime*mixture_num*(1+featureDim) + 
		       timeStep*mixture_num;
	    
#else
	    varshift = totalTime*mixture_num*(1+featureDim) + 
		       timeStep*mixture_num +
		       mixtureI;
#endif

	    const real_t *mean  = mdnPara + meanshift;
	    const real_t *var   = mdnPara + varshift;

	    // pointer to the posterior P and sum of posterior P
	    const real_t *postP = meanDis + timeStep * mixture_num + mixtureI;
	    const real_t *sumPost=meanDis + totalTime* mixture_num + timeStep;
	    real_t posterior = helpers::safeExp((*postP) - (*sumPost));
	    (*errorm)  = posterior*(*mean - *tardata)/(*var)/(*var);
	    // (*errorv2) += posterior*featureDim - (*errorm)*(*mean - *tardata);

	    // STUPID MISTAKE
	    // (*errorv)  = posterior*featureDim - (*errorm)*(*mean - *tardata);
	    // How could I multiply featureDim here ! For each dimension, it should be 1
	    (*errorv)  = posterior - (*errorm)*(*mean - *tardata);
	    
	}

    };
    
    
    struct ComputeBPAccumVariance
    {
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
	real_t para;
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
	    int pos = totalTime*mixtureNum + timeStep*mixtureNum*featureDim;
	    const real_t *mean = mdnPara + pos + flag*featureDim + dimStep;
	    pos = totalTime*(mixtureNum+mixtureNum*featureDim) + timeStep*mixtureNum;

#ifdef ALTER_TIEVAR
	    const real_t *var = mdnPara + pos;
#else
	    const real_t *var = mdnPara + pos + flag;
#endif	    

	    pos = timeStep * layerSizeOut + startDOut + dimStep;
	    *(targets+pos) = (*var)*para*seed + (*mean);
	}
    };

    struct GetParameterMixture
    {
	int featureDim;
	int NNOutputSize;
	int mixtureNum;
	int totalTime;
	int startDimIn;
	real_t *targets;
	real_t *mdnPara;
	const char *patTypes;

	// from timesteps * featureDim
	__host__ __device__ void operator() (const int timeStep) const
	{
	    
	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;
	    
	    // pointer to the weight
	    int pos_weight = timeStep*mixtureNum;
	    int pos_mean   = totalTime*mixtureNum + timeStep*mixtureNum*featureDim;
	    int pos_var    = totalTime*(mixtureNum+mixtureNum*featureDim) + timeStep*mixtureNum;
	    int pos_output = timeStep*NNOutputSize;
	    
	    int bias = startDimIn;
	    for (int i = 0; i<mixtureNum; i++, bias++){
		*(targets + pos_output + bias) = *(mdnPara + pos_weight + i);
	    }
	    for (int i = 0; i<mixtureNum*featureDim; i++, bias++){
		*(targets + pos_output + bias) = *(mdnPara + pos_mean + i);
	    }
	    for (int i = 0; i<mixtureNum; i++, bias++){
		*(targets + pos_output + bias) = *(mdnPara + pos_var + i);
	    }	    
	}
    };
    
    struct copyMixtureWeightforEMGen
    {

	// copy the predicted parameter into the m_tmpPat
	int mixture_num;
	const char *patTypes;
	real_t *meanDis; 
	real_t *mdnPara;   // mean value of the mixture
	int totalTime;   //

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
		// save  w_i p_i
		*(meanDisPtr+i) = tmptmp; 
	    }
	    // save sum_i^mixture_num w_i p_i
	    meanDisPtr = meanDis + totalTime * mixture_num + timeStep;
	    *meanDisPtr= tmp;
	    //return -1*helpers::safeLog(tmp);
	}

    };
    
    struct initIterEMGen
    {
	// it seems that there is no need to differentiate the equation
	// for initialzation and iteration on the EM estimation
	// For initialization, just assume posterior probability is equal for all mixtures
	// For iteration, just plug in the posterior probability

	int featureDim;
	int mixtureNM;
	int totalTime;
	int outputSize;
	int startDOut;

	real_t *postP;
	real_t *mdnPara;
	real_t *targets;
	const char *patTypes;
	
	// u = (sum_i u_i/var_i^2) / (sum_i /var_i^2)
	__host__ __device__ void operator() (const int idx) const
	{
	    int timeStep = idx / (featureDim);
	    int dimOutput= idx % (featureDim);

	    if (patTypes[timeStep] == PATTYPE_NONE)
		return;
	    const real_t *m, *v, *p, *q;
	    /*const real_t *w;
	      int pos_w    = timeStep*mixtureNM;*/
	    // FATAL ERROR: remember to shift dimOutput
	    int pos_mean = totalTime*mixtureNM + timeStep*featureDim*mixtureNM + dimOutput;
	    int pos_var  = totalTime*(mixtureNM+featureDim*mixtureNM)+timeStep * mixtureNM;
	    int pos_postM = timeStep * mixtureNM;
	    int pos_postS = totalTime * mixtureNM + timeStep;

	    real_t tmp1=0.0;
	    real_t tmp2=0.0;
	    
	    /*real_t tmp3=0.0;
	    int widx = 0;
	    */
	    for(int i = 0; i < mixtureNM; i++){
		v = mdnPara + pos_var   + i;
		m = mdnPara + pos_mean  + (i*featureDim);
		p = postP   + pos_postM + i;
		q = postP   + pos_postS;
		
		/*
		  w = mdnPara + pos_w   + i;
		  if ((*w)>tmp3){
		      widx = i;
		      tmp3 = *w;
		  }
		*/

		tmp2 += helpers::safeExp((*p)-(*q))/((*v)*(*v));
		tmp1 += ((*m)*(helpers::safeExp((*p)-(*q))))/((*v)*(*v));
	    }

	    tmp1 = tmp1/tmp2;
	    // tmp1 = mdnPara[pos_mean+widx*featureDim];
	    int pos_tar = timeStep * outputSize + startDOut + dimOutput;
	    *(targets+pos_tar) = tmp1;
	}
    };

    
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
    
}
}


/********************************************************
 definition of the MDN Units

  for simplicity, let's define the output from previous output layer as a_i_*, i is the dimension
  a_i_g: input to the MDNUnit_sigmoid
  a_i_s: ... to MDNUnit_softmax
  a_i_mk: ... to MDNUnit_mixture, the mixture weight
  a_ij_mu: ... to MDNUnit_mixture, the mixture mean (i-th mixture, j-th dimension)
  a_i_ms:  ... to MDNUnit_mixture, the mixture variance (i-th mixture, shared by all dimension)  
 *******************************************************/

namespace layers {
    
    // virtual class of MDNUnit
    template <typename TDevice>
    MDNUnit<TDevice>::MDNUnit(int startDim, int endDim, int startDimOut, int endDimOut, 
			      int type, int paraDim, Layer<TDevice> &precedingLayer,
			      int outputSize)
	: m_startDim        (startDim)
	, m_endDim          (endDim)
	, m_type            (type)
	, m_paraDim         (paraDim)
	, m_precedingLayer  (precedingLayer)
	, m_startDimOut     (startDimOut)
	, m_endDimOut       (endDimOut)
	, m_layerSizeIn     (precedingLayer.size())
	, m_layerSizeTar    (outputSize)
    {
	// initilize the parameter vec
	int n = m_precedingLayer.patTypes().size();
	m_paraVec.resize(m_paraDim*n, 0.0);
	
	/*
	 here, we assume homogenous parameter should be adjacent to each other
	 for mixture Unit, 
	 [mixture_weight_1_time_1, mixture_weight_2_time_1, ... mixture_weight_k_time_1,
	  mixture_weight_1_time_2, mixture_weight_2_time_2, ... mixture_weight_k_time_2,
	  ...
	  mixture_weight_1_time_N, mixture_weight_2_time_N, ... mixture_weight_k_time_N,
	  mixture_mean_1_1_time_1, mixture_mean_1_2_time_1, ... mixture_mean_1_D_time_1,
	  ...]
	*/
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

    /********************************************************
     MDNUnit_sigmoid
    *******************************************************/
    template <typename TDevice>
    MDNUnit_sigmoid<TDevice>::MDNUnit_sigmoid(
	int startDim, int endDim, int startDimOut, int endDimOut, int type, 
	Layer<TDevice> &precedingLayer, int outputSize)
	: MDNUnit<TDevice>(startDim, endDim, startDimOut, endDimOut, 
			   type, (endDim - startDim), precedingLayer,
			   outputSize)
    {
	// nothing else to be initialized
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
	
	// sigmoid, o_i_g = sigmoid(a_i_g)
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
		    printf("SigForward: %f %f\n", tmp1[i], tmp2[i]);
		}
#endif

	}}	    
	
    }

    template <typename TDevice>
    void MDNUnit_sigmoid<TDevice>::getEMOutput(const real_t para, real_vector &targets)
    {	
	// no EM (no mixture at all), just sampling 
	this->getOutput(para, helpers::getRawPointer(targets));
    }
	
    template <typename TDevice>
    void MDNUnit_sigmoid<TDevice>::getOutput(const real_t para, real_t *targets)
    {
	// Here, probability is directly used as output
	// sampling output
	{{
		internal::SamplingSigmoid fn;
		fn.NNTargetSize = this->m_layerSizeTar;
		fn.startDTarget = this->m_startDimOut;
		fn.endDTarget   = this->m_endDimOut;
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
    void MDNUnit_sigmoid<TDevice>::getParameter(real_t *targets)
    {
	// STUPID MISTAKE
	/*{{
		internal::CopySimple fn;
		...
	}}*/
	{{
		// actually not sampling. but the method is the same
		// only the position of output is different
		// just borrow the function
		internal::SamplingSigmoid fn;
		fn.NNTargetSize = this->m_precedingLayer.size();
		fn.startDTarget = this->m_startDim;
		fn.endDTarget   = this->m_endDim;
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
		n = n*this->m_paraDim;
		
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
			
			printf("Back: %f\t", prop);
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
	Layer<TDevice> &precedingLayer, int outputSize)
        : MDNUnit<TDevice>(startDim, endDim, startDimOut, endDimOut, 
			   type, endDim-startDim, precedingLayer,
			   outputSize)
    {   
	// special strategy for vec is unecessary
	m_offset.resize(this->m_precedingLayer.patTypes().size(), 0.0);
	
	// assume ont softmax unit only corresponds to one dimension of the output
	if ((endDimOut - startDimOut) != 1){
	    throw std::runtime_error("Check MDN configure. SoftMax => one dimensional target");
	}
	
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
		fn.layerSize = this->m_paraDim;
		fn.outputs   = helpers::getRawPointer(this->m_paraVec);

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
    void MDNUnit_softmax<TDevice>::getOutput(const real_t para,real_t *targets)
    {
	{{    
	    internal::SamplingSoftmax fn;
	    fn.paradim = this->m_paraDim;
	    fn.startDOut = this->m_startDimOut;
	    fn.output    = targets; //helpers::getRawPointer(this->m_targets);
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
		
	    tmp = thrust::transform_reduce(
		  thrust::counting_iterator<int>(0),
		  thrust::counting_iterator<int>(0)+n,
		  fn,
		  (real_t)0,
		  thrust::plus<real_t>()
	  );

	}}
	return tmp;
    }
    template <typename TDevice>
    void MDNUnit_softmax<TDevice>::computeBackward(real_vector &targets)
    {
    }

    /********************************************************
     MDNUnit_mixture
    *******************************************************/
    template <typename TDevice>
    MDNUnit_mixture<TDevice>::MDNUnit_mixture(
	int startDim, int endDim, int startDimOut, int endDimOut, int type, 
	Layer<TDevice> &precedingLayer, int outputSize)
        : MDNUnit<TDevice>(startDim, endDim, startDimOut, endDimOut, 
			   type, (endDim-startDim), precedingLayer,
			   outputSize)
	, m_numMixture    (type)
	, m_featureDim    ((endDim - startDim - type*2)/type)
	, m_varFloor      (0.0)
    {                                                          
	
	// offset for the mixture weight
	m_offset.resize(this->m_precedingLayer.patTypes().size(), 0.0);	

	// intermediate matrix to store the \sum_dim (t-\mu)^2 and sum_mixture \sum_dim (t-\mu)^2
	m_tmpPat.resize(this->m_precedingLayer.patTypes().size()*(m_numMixture+1), 0.0);

	// for BP variance
	m_varBP.resize(this->m_precedingLayer.patTypes().size()*(endDim-startDim)*type, 0.0);

	// check the number of parameter
	int numParameter = m_numMixture * (m_featureDim + 2);
	if (numParameter != (endDim - startDim)){
	    printf("Parameter number: %d is not compatible", numParameter);
	    throw std::runtime_error("Parameter of mixture MDN is not correctly configured");
	}
	
    }
    template <typename TDevice>
    MDNUnit_mixture<TDevice>::~MDNUnit_mixture()
    {                                                           
    }
    
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
		
		
		// adjust the bias for mean
		real_t step = (config.getVarInitPara()*2)/(m_numMixture+1);
		real_t start= -1*config.getVarInitPara()+step;
		for (int i =0; i<m_numMixture; i++){
		    for (int j=0; j<m_featureDim; j++)
			biasInit[m_numMixture + i*m_featureDim + j] = 
			    mVecTmp[this->m_startDimOut+j] + 
			    (step * i + start) * vVecTmp[this->m_startDimOut+j];
		}		
		// set the same for variance
		for (int i = 0; i<m_numMixture; i++)
		    biasInit[m_numMixture*(m_featureDim+1) + i] = internal::safeLog(vVecAver);
		
		this->m_varFloor = config.getVFloorPara() * vVecAver;

		// set bias for mixture weight
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
	// softmax part for mixture weight
	// calculate the offset 
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

	// calculate the Exp
	{{
		internal::CalculateExpFn fn;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.startD    = this->m_startDim;
		fn.endD      = this->m_startDim + this->m_numMixture;
		fn.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNOutput  = helpers::getRawPointer(this->m_precedingLayer.outputs());
		fn.offset    = helpers::getRawPointer(this->m_offset);
		
		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		n = n*(this->m_numMixture);
		
		thrust::transform(
		   thrust::counting_iterator<int>(0),
		   thrust::counting_iterator<int>(0)+n,
		   this->m_paraVec.begin(),
		   fn);
	}}

	// sum up
	{{
		internal::SumUpOutputsFn fn;
		fn.layerSize = this->m_numMixture;
		fn.outputs   = helpers::getRawPointer(this->m_paraVec);

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
		fn.layerSize = this->m_numMixture;
		fn.normFacts = helpers::getRawPointer(this->m_offset);
		
		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences()*this->m_numMixture;

		thrust::for_each(
		    thrust::make_zip_iterator(
				thrust::make_tuple(this->m_paraVec.begin(),
						   thrust::counting_iterator<int>(0))),
		    thrust::make_zip_iterator(
				thrust::make_tuple(this->m_paraVec.begin()+n, 
						   thrust::counting_iterator<int>(0)+n)),
		    fn);

        }}

	// the mean part (unnessary to change anything. But need to copy)
	{{
		internal::CopyMean fn;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.featureDim   = this->m_numMixture*this->m_featureDim;
		fn.startD       = this->m_startDim + this->m_numMixture;
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNOutput     = helpers::getRawPointer(this->m_precedingLayer.outputs());

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		int n2 = n*this->m_numMixture*this->m_featureDim;

		thrust::for_each(
		    thrust::make_zip_iterator(
			  thrust::make_tuple(this->m_paraVec.begin()+n*this->m_numMixture,
					     thrust::counting_iterator<int>(0))),
		    thrust::make_zip_iterator(
                          thrust::make_tuple(this->m_paraVec.begin()+n2+n*this->m_numMixture, 
					     thrust::counting_iterator<int>(0)+n)),
		    fn);
	}}

	// the variance part
	// calculate the Exp
	{{
		internal::CalculateExpSimpleFnForVar fn;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.startD       = this->m_startDim + this->m_numMixture*(1+this->m_featureDim);
		fn.endD         = this->m_endDim;
		
		fn.varFloor     = this->m_varFloor;
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn.NNOutput     = helpers::getRawPointer(this->m_precedingLayer.outputs());

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		
		thrust::transform(
		   thrust::counting_iterator<int>(0),
		   thrust::counting_iterator<int>(0)+n*(this->m_numMixture),
		   this->m_paraVec.begin() + n*(this->m_numMixture+
						this->m_featureDim*this->m_numMixture),
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
		int pos_var  = n * this->m_numMixture * (1 + this->m_featureDim) +
		    timeStep*this->m_numMixture + mixIndex;
		
		for (int i = this->m_featureDim-10; i<this->m_featureDim; i++){
		    tmp += (temp_vec1[pos_var+i]);
			
		}
		}
		//printf("%f", tmp);
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

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		
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
	bool finish =false;
	int iter    =0;
	while(!finish)
	{   
	    
	    {{
		internal::initIterEMGen fn;
		fn.featureDim   = this->m_featureDim;
		fn.mixtureNM    = this->m_numMixture;
		fn.outputSize   = this->m_layerSizeTar;
		fn.startDOut    = this->m_startDimOut;
		fn.totalTime    = totalTime;

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
			pos_var += 1; // move to the next mixture
			pos_mean += fn.featureDim; // move to the next mixture
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
		    
		    for (int i = 0; i<fn.featureDim; i++){
			tmp += ((temp_vec2[pos_data+i]) - temp_vec1[pos_mean+i]) * 
			    ((temp_vec2[pos_data+i]) - temp_vec1[pos_mean+i])
			    /(temp_vec1[pos_var])
			    /(temp_vec1[pos_var])/2;
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
		fn.startD    = this->m_startDim;
		fn.endD      = this->m_endDim;
		fn.startDOut = this->m_startDimOut;
		fn.mixture_num  = this->m_numMixture;
		fn.featureDim   = this->m_featureDim;
		fn.layerSizeOut = this->m_layerSizeTar;
		
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
			t * fn.mixture_num;
		    
		    for (int i = 0; i<fn.mixture_num; i++){
			
			tmptmp = std::log(tvec1[mixPos+i])-(tvec2[meanPos+i]);
			tmptmp = tmptmp - fn.featureDim/2*std::log(2*PI_DEFINITION);
			// change this line according to ALTER_TIEVAR
			//tmptmp = tmptmp - fn.featureDim*std::log(tvec1[varPos]);
			tmptmp = tmptmp - fn.featureDim*std::log(tvec1[varPos+i]);
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
    void MDNUnit_mixture<TDevice>::getOutput(const real_t para,real_t *targets)
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

		
#ifdef DEBUG_LOCAL

#endif
	
	// copy to GPU
	temp2 = temp;
	
	{{
		internal::SamplingMixture fn;
		fn.featureDim = this->m_featureDim;
		fn.layerSizeOut = this->m_layerSizeTar;
		fn.startDOut    = this->m_startDimOut;
		fn.mixtureNum   = this->m_numMixture;
		fn.totalTime    = (int)(time/(this->m_featureDim));
		fn.para         = para;
		fn.targets      = targets;
		fn.mdnPara      = helpers::getRawPointer(this->m_paraVec);
		
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
		    pos = fn.totalTime*(fn.mixtureNum+fn.mixtureNum*fn.featureDim) + 
			timeStep*fn.mixtureNum;
#ifdef ALTER_TIEVAR
		    const real_t var = mdnPara[pos];
#else
		    const real_t var = mdnPara[pos + flag];
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

		
		fn.targets       = targets;
		fn.featureDim   = this->m_featureDim;
		fn.NNOutputSize = this->m_precedingLayer.size();
		fn.startDimIn   = this->m_startDim;
		fn.mixtureNum   = this->m_numMixture;
		fn.totalTime    = time;
		fn.targets      = targets;
		fn.mdnPara      = helpers::getRawPointer(this->m_paraVec);
		fn.patTypes     = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		
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
		fn.startDOut = this->m_startDimOut;
		fn.mixture_num = this->m_numMixture;
		fn.featureDim  = this->m_featureDim;
		fn.layerSizeOut= this->m_layerSizeTar;

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
		printf("MixtureDistance: data, mean, var, dis_over_dim\n");
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
			timeStep*fn.mixture_num;
		    
		    for (int i = 0; i<fn.featureDim; i++){
			tmp += (temp_vec2[pos_data+i] - temp_vec1[pos_mean+i]) * 
			    (temp_vec2[pos_data+i] - temp_vec1[pos_mean+i])
			    /(temp_vec1[pos_var+mixIndex])
			    /(temp_vec1[pos_var+mixIndex])/2;
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
		fn.startD    = this->m_startDim;
		fn.endD      = this->m_endDim;
		fn.startDOut = this->m_startDimOut;
		fn.mixture_num  = this->m_numMixture;
		fn.featureDim   = this->m_featureDim;
		fn.layerSizeOut = this->m_layerSizeTar;
		
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
		printf("Calculate Error: mixture, sum\n");
		for (int t = 0; t < n; t++){
		    real_t tmp=-1e30;
		    real_t tmptmp;
		    
		    int meanPos = t * fn.mixture_num;
		    int mixPos  = t * fn.mixture_num;
		    int varPos  = fn.totalTime*(fn.mixture_num+fn.mixture_num*fn.featureDim)+
			t * fn.mixture_num;
		    
		    for (int i = 0; i<fn.mixture_num; i++){
			
			tmptmp = std::log(tvec1[mixPos+i])-(tvec2[meanPos+i]);
			tmptmp = tmptmp - fn.featureDim/2*std::log(2*PI_DEFINITION);
			tmptmp = tmptmp - fn.featureDim*std::log(tvec1[varPos+i]);
			
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
		

		int n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();
		
		fn.totalTime = n;
		n = n * this->m_numMixture;
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
		fn.layerSize = this->m_precedingLayer.size();
		fn.startD    = this->m_startDim + this->m_numMixture;
		fn.startDOut = this->m_startDimOut;
		fn.layerSizeOut = this->m_layerSizeTar;
		fn.featureDim  = this->m_featureDim;
		fn.mixture_num = this->m_numMixture;

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
				 fn
				 );

		internal::ComputeBPAccumVariance fn2;
		fn2.layerSize = this->m_precedingLayer.size();
		fn2.startD    = this->m_startDim + this->m_numMixture;
		fn2.featureDim  = this->m_featureDim;
		fn2.mixture_num = this->m_numMixture;

		fn2.patTypes  = helpers::getRawPointer(this->m_precedingLayer.patTypes());
		fn2.errors    = helpers::getRawPointer(this->m_precedingLayer.outputErrors());
		fn2.varBuff   = helpers::getRawPointer(this->m_varBP);

		n =this->m_precedingLayer.curMaxSeqLength();
		n = n*this->m_precedingLayer.parallelSequences();		
 		thrust::for_each(thrust::counting_iterator<int>(0),
				 thrust::counting_iterator<int>(0)+n,
				 fn2
				 );

		
		
#ifdef DEBUG_LOCAL
		
		Cpu::real_vector errors = this->m_precedingLayer.outputErrors();
		Cpu::real_vector mdnPara= this->m_paraVec;
		Cpu::real_vector meanDis= this->m_tmpPat;
		Cpu::real_vector target = targets;
		Cpu::real_vector varBuf = this->m_varBP;
		Cpu::real_vector errorVBuf;
		real_t errorm(0.0), errorv(0.0);
		printf("Gradient for mean and variance of every mixture: time errorm errorv ...\n");
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

		    int varshift_error= timeStep*fn.layerSize +
			fn.startD + fn.mixture_num*fn.featureDim + mixtureI;
		    //real_t errorv = errors[varshift];
	    
		    // pointer to the target data y
		    const real_t tardata= target[timeStep*fn.layerSizeOut+fn.startDOut+featureI];

		    // pointer to the mean
		    int meanshift= fn.totalTime * fn.mixture_num + 
			timeStep * fn.mixture_num * fn.featureDim + 
			mixtureI * fn.featureDim + 
			featureI;
		    int varshift = fn.totalTime * fn.mixture_num * (1 + fn.featureDim) + 
			timeStep * fn.mixture_num + mixtureI;
	    
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




    /********************************************************
     MDNLayer
    *******************************************************/


    // definition of the MDN layer
    template <typename TDevice>
    MDNLayer<TDevice>::MDNLayer(const helpers::JsonValue &layerChild, 
				const helpers::JsonValue &weightsSection, 
				Layer<TDevice> &precedingLayer)
	: PostOutputLayer<TDevice>(layerChild, precedingLayer, -1)
    {
        const Configuration &config = Configuration::instance();
	
        // parse the MDN vector
	int numEle;
	int unitS, unitE, mdnType;
	int unitSOut, unitEOut;
	int outputSize = 0;
	m_mdnParaDim = 0;

	// build the MDN unit
	MDNUnit<TDevice> *mdnUnit;

	
	if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) {
	    const rapidjson::Value &weightsChild = (*weightsSection)[this->name().c_str()];
            if (!weightsChild.IsObject())
                throw std::runtime_error(std::string("Weights section for layer '") + 
					 this->name() + "' is not an object");
	    if (!weightsChild.HasMember("config") || !weightsChild["config"].IsArray())
                throw std::runtime_error(std::string("Missing array 'config/") + 
					 this->name() + "/config'");
            const rapidjson::Value &inputWeightsChild    = weightsChild["config"];
            m_mdnConfigVec.reserve(inputWeightsChild.Size());;
	    for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); 
		 it != inputWeightsChild.End(); ++it)
                m_mdnConfigVec.push_back(static_cast<real_t>(it->GetDouble()));
            numEle = m_mdnConfigVec[0];
        }else{
	
	    std::ifstream ifs(config.mdnFlagPath().c_str(), 
			      std::ifstream::binary | std::ifstream::in);
	    if (!ifs.good()){
		throw std::runtime_error(std::string("Fail to open "+config.mdnFlagPath()));
	    }
	    std::streampos numEleS, numEleE;
	    numEleS = ifs.tellg();
	    ifs.seekg(0, std::ios::end);
	    numEleE = ifs.tellg();
	    long int tmpnumEle  = (numEleE-numEleS)/sizeof(real_t);
	    ifs.seekg(0, std::ios::beg);
	
	    real_t tempVal;
	    ifs.read((char *)&tempVal, sizeof(real_t)); //
	    numEle = (long int)tempVal;                 // get the total number of parameter
	    
	    if (tmpnumEle != (numEle*5+1)){
		throw std::runtime_error("Invalid MDN config file.");
	    }
	    
	    // no use at all
	    m_mdnConfigVec.resize(1+numEle*5, 0.0);
	    m_mdnConfigVec[0] = (real_t)numEle;
	    for (int i=0; i<numEle; i++){
		ifs.read((char *)&tempVal, sizeof(real_t));
		m_mdnConfigVec[1+i*5] = tempVal;
		ifs.read((char *)&tempVal, sizeof(real_t));
		m_mdnConfigVec[2+i*5] = tempVal;
		ifs.read((char *)&tempVal, sizeof(real_t));
		m_mdnConfigVec[3+i*5] = tempVal;
		ifs.read((char *)&tempVal, sizeof(real_t));
		m_mdnConfigVec[4+i*5] = tempVal;
		ifs.read((char *)&tempVal, sizeof(real_t));
		m_mdnConfigVec[5+i*5] = tempVal;
	    }
	    ifs.close();
	}
	
	if (m_mdnConfigVec.size() != numEle*5+1){
	    throw std::runtime_error("Error in reading the configuration of MDN");
	}

	for (int i=0; i<numEle; i++){
	    unitS = (int)m_mdnConfigVec[1+i*5];
	    unitE = (int)m_mdnConfigVec[2+i*5];
	    unitSOut = (int)m_mdnConfigVec[3+i*5];
	    unitEOut = (int)m_mdnConfigVec[4+i*5];
	    mdnType  = (int)m_mdnConfigVec[5+i*5];
	    
	    if (mdnType==MDN_TYPE_SIGMOID){
		mdnUnit = new MDNUnit_sigmoid<TDevice>(unitS, unitE, unitSOut, unitEOut, mdnType, 
						       precedingLayer, this->size());
		m_mdnParaDim += (unitE - unitS);
		outputSize += (unitE - unitS);
	    }else if(mdnType==MDN_TYPE_SOFTMAX){
		mdnUnit = new MDNUnit_softmax<TDevice>(unitS, unitE, unitSOut, unitEOut, mdnType, 
						       precedingLayer, this->size());
		m_mdnParaDim += (unitE - unitS);
		outputSize += 1;
	    }else if(mdnType > 0){
		mdnUnit = new MDNUnit_mixture<TDevice>(unitS, unitE, unitSOut, unitEOut, mdnType, 
						       precedingLayer, this->size());
		// K mixture weight, K*Dim mean, K*1 variance. Variance is tied across dimension
		m_mdnParaDim += (unitE - unitS);
		outputSize += ((unitE - unitS) - 2*mdnType)/mdnType ;
	    }else{
		throw std::runtime_error("mdnUnit type invalid (>0, 0, -1)");
	    }
	    m_mdnUnits.push_back(boost::shared_ptr<MDNUnit<TDevice> >(mdnUnit));
	}
	
	// check
	printf("MDN layer parameter number: %d\n", m_mdnParaDim);
	if (m_mdnParaDim != precedingLayer.size()){
	    throw std::runtime_error("MDN parameter dim is not equal to NN output dimension");
	}
	if (outputSize != this->size()){
	    throw std::runtime_error("Mismatch between target dimension and MDN configuration");
	}
	
	
    }

    template <typename TDevice>
    MDNLayer<TDevice>::~MDNLayer()
    {
    }

    template <typename TDevice>
    const std::string& MDNLayer<TDevice>::type() const
    {
        static const std::string s("mdn");
        return s;
    }

    template <typename TDevice>
    real_t MDNLayer<TDevice>::calculateError()
    {
	real_t temp = 0.0;
	real_t temp2= 0.0;
	int i=0;
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    temp2 = mdnUnit->calculateError(this->_targets());
	    if (temp2 != temp2)
		printf("NaN: %d-th unit\t", i);
	    temp += temp2;
	    ++i;
	}
	return temp;
    }

    template <typename TDevice>
    typename MDNLayer<TDevice>::real_vector& MDNLayer<TDevice>::mdnParaVec()
    {
        return m_mdnParaVec;
    }
    
    template <typename TDevice>
    int MDNLayer<TDevice>::mdnParaDim()
    {
	return m_mdnParaDim;
    }

    template <typename TDevice>
    void MDNLayer<TDevice>::computeForwardPass()
    {
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    mdnUnit->computeForward();
	}
    }

    template <typename TDevice>
    void MDNLayer<TDevice>::computeBackwardPass()
    {
	// For updating the variance here, we accumulate the gradients.
	// Thus, need to reset outputErrors
	thrust::fill(this->_outputErrors().begin(),
		     this->_outputErrors().end(),
		     (real_t)0.0);
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    mdnUnit->computeBackward(this->_targets());
	}
    }
    
    template <typename TDevice>
    void MDNLayer<TDevice>::exportConfig(const helpers::JsonValue &weightsObject, 
					 const helpers::JsonAllocator &allocator) const
    {
	if (!weightsObject->IsObject())
            throw std::runtime_error("The JSON value is not an object");
	
	rapidjson::Value inputConfig(rapidjson::kArrayType);
	int inputConfigCount = this->m_mdnConfigVec.size();
	inputConfig.Reserve(inputConfigCount, allocator);
	for (int i = 0; i < inputConfigCount; i++)
	    inputConfig.PushBack(this->m_mdnConfigVec[i], allocator);
	rapidjson::Value weightsSection(rapidjson::kObjectType);
	weightsSection.AddMember("config", inputConfig, allocator);
	weightsObject->AddMember(this->name().c_str(), weightsSection, allocator);
    }
    
    template <typename TDevice>
    void MDNLayer<TDevice>::getOutput(const real_t para)
    {
	// Modify 05-24 Add support to EM-style generation
	if (para < -3.0){
	    throw std::runtime_error("Parameter to MDN->getOutput can't be less than -1.0");
	}else if (para >= 0.0){
	    BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
		mdnUnit->getOutput(para, helpers::getRawPointer(this->_targets()));
	    }
	    printf("sampling with variance scaled by %f", para);
	}else if (para > -1.50){
	    printf("generating the parameters of MDN");
	    this->m_mdnParaVec.resize(this->m_mdnParaDim*
				      this->precedingLayer().curMaxSeqLength()*
				      this->precedingLayer().parallelSequences(), 
				      0.0);
	    BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
		mdnUnit->getParameter(helpers::getRawPointer(this->m_mdnParaVec));
	    }
	}else{
	    printf("EM-style generation\n");
	    int i = 0;
	    BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
		printf("U%d",i++);
		mdnUnit->getEMOutput(para, this->_targets());
	    }
	}

#ifdef DEBUG_LOCAL
	Cpu::real_vector temp=this->_targets();
	printf("Sampling: %f \n", temp[0]);
#endif	
    }
    
    template <typename TDevice>
    Cpu::real_vector MDNLayer<TDevice>::getMdnConfigVec()
    {
	return m_mdnConfigVec;
    }
    
    template <typename TDevice>
    void MDNLayer<TDevice>::initPreOutput(const MDNLayer<TDevice>::cpu_real_vector &mVec, 
					  const MDNLayer<TDevice>::cpu_real_vector &vVec)
    {
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    mdnUnit->initPreOutput(mVec, vVec);
	}
    }

    template class MDNLayer<Cpu>;
    template class MDNLayer<Gpu>;
    template class MDNUnit<Cpu>;
    template class MDNUnit<Gpu>;
    template class MDNUnit_sigmoid<Cpu>;
    template class MDNUnit_sigmoid<Gpu>;
    template class MDNUnit_mixture<Cpu>;
    template class MDNUnit_mixture<Gpu>;
    template class MDNUnit_softmax<Cpu>;
    template class MDNUnit_softmax<Gpu>;

}
