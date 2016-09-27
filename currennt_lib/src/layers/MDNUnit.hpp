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
#ifndef LAYERS_MDNUNIT_HPP
#define LAYERS_MDNUNIT_HPP

#include "PostOutputLayer.hpp"


#define MDNUNIT_TYPE_0 0   // TYPE TAG for sigmoid, softmax, mixture
#define MDNUNIT_TYPE_1 1   // Type tag for mixture with trainable [w,b]  for the output layer
#define MDNUNIT_TYPE_2 2   // Type tag for mixture with [w,b] predicted by the NN

#define MDNUNIT_TYPE_1_DIRECT 0 // Type 1 dyn, time axis
#define MDNUNIT_TYPE_1_DIRECD 1 // Type 1 dyn, dimension axis
#define MDNUNIT_TYPE_1_DIRECB 2 // Type 1 dyn, both axes

namespace layers{
    
    // utilizes by MDNUnits
    int MixtureDynWeightNum(int featureDim, int mixNum, int backOrder, int dynDirection);

    /********************************************************
     MDNUnit: describes the distribution of the target data
     based on parameters given by NN
     
     SIGMOID MIXTURE SOFTMAX
       ^       ^       ^
       |       |       |
       -----------------
               |
            MDNUnits 
     However, SOFTMAX part is incomplete (the backpropagation part)
    ********************************************************/

    // virtual class of MDNUnit
    template <typename TDevice>
    class MDNUnit
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;

    protected:
	
	const int   m_startDim;            // 
	const int   m_endDim;              // [m_startDim, m_endDim) specifies the dimension range
	                                   // of the data embedded in the (input) data vector

	const int   m_layerSizeIn;         // dimension of the input vector (size of previous layer)

	const int   m_paraDim;             // dimension of the parameter set of this unit
	real_vector m_paraVec;             // the predicted parameter vector of this unit 
	real_t     *m_paraPtr;             // pointer to the parameter 

	Layer<TDevice> &m_precedingLayer;  // previous output layer (output of the network)

	/* w.r.t the target data (target data of NN) */
	const int   m_startDimOut;         // 
	const int   m_endDimOut;           // [m_startDimOut, m_endDimOut) specifies the dimension 
	                                   // range of the data in the output data vector
	const int   m_layerSizeTar;        // dimension of the target data vector (all)
	real_t     *m_targetPtr;           // pointer to the target data
	                                   // 
	// other
	const int   m_type;                // type of this unit
	real_vector m_mdnOutput;           // control the output method of processing (sampling)
	real_vector m_varScale;            // the ratio to scale each dimension of the variance 
	                                   // (used for mixture unit)
	// 
	const int  m_trainable;            // a type flag for trainable MDNUnit
	                                   // 0: sigmoid, softmax, mixture
	                                   // 1: mixture with dynamic link, tied
	                                   // 2: mixture with dynamic link, predicted by the NN

    public:
	MDNUnit(int startDim,    int endDim,  int startDimOut,                int endDimOut, 
		int type,        int paraDim, Layer<TDevice> &precedingLayer, int outputSize,
		const int trainable);

	virtual      ~MDNUnit();

	// methods
	const   int& paraDim() const; 
	
	// pure vitural function (must be over-written)
	
	// transform the previous output into MDN parameters
	virtual void computeForward() =0;  
	
	// sampling output from MDN
	virtual void getOutput(const real_t para, real_vector &targets) =0; 

	// vector of coefficients to scale the variance (maybe only for mixture unit)
	const   real_vector& varScale() const;	

	// EM MOPG output
	virtual void getEMOutput(const real_t para, real_vector &targets) =0; 

	// output the parmeter of this unit
	virtual void getParameter(real_t *targets) =0; 

	// the error(-log likelihood)
	virtual real_t calculateError(real_vector &targets) =0;
	
	// back ward computation
	virtual void computeBackward(real_vector &targets) =0; 

	// initialize the parameter set of the units
	virtual void initPreOutput(const cpu_real_vector &mVec, const cpu_real_vector &vVec) =0;
	
	// whether this unit contains trainable part ? Equivalent to a type tag
	virtual int flagTrainable() const;
	
	// whether this unit ties the variance ?
	virtual bool flagVariance() const;
	
	// link the wegith for trainable
	virtual void linkWeight(real_vector& weights, real_vector& weightsUpdate);

	// to validate the configuration this unit
	virtual bool flagValid() =0;
    };


    /********************************************************
     MDNUnit_sigmoid: elementwise binary distribution
       p(t^(n,t)_d | x^(n, t)) = sigmoid(NN(x^(n,t)))
       
        MDNUnits -> MDNUnit_sigmoid
    ********************************************************/    
    // MDN sigmoid unit
    template <typename TDevice>
    class MDNUnit_sigmoid : public MDNUnit<TDevice>
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;
	
    public:

	// methods
	MDNUnit_sigmoid(int startDim, int endDim, int startDimOut, int endDimOut, 
			int type, Layer<TDevice> &precedingLayer, int outputSize,
			const int trainable = MDNUNIT_TYPE_0);

	virtual ~MDNUnit_sigmoid();

	virtual void computeForward();
	
	virtual void getOutput(const real_t para,real_vector &targets);
	
	virtual void getEMOutput(const real_t para, real_vector &targets);

	virtual void getParameter(real_t *targets);

	virtual real_t calculateError(real_vector &targets);
	
	virtual void computeBackward(real_vector &targets);

	virtual void initPreOutput(const cpu_real_vector &mVec, const cpu_real_vector &vVec);
	
	virtual bool flagValid();
    };

    /********************************************************
     MDNUnit_softmax: 
        MDNUnits -> MDNUnit_softmax
     NOTE: I never test softmax unit
    ********************************************************/    
    // MDN softmax unit
    template <typename TDevice>
    class MDNUnit_softmax : public MDNUnit<TDevice>
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;

    protected:
	real_vector m_offset;
	
    public:
	MDNUnit_softmax(int startDim, int endDim, int startDimOut, int endDimOut, 
			int type, Layer<TDevice> &precedingLayer, int outputSize,
			const int trainable = MDNUNIT_TYPE_0);

	virtual ~MDNUnit_softmax();

	virtual void computeForward();
	
	virtual void getOutput(const real_t para,real_vector &targets);

	virtual void getEMOutput(const real_t para, real_vector &targets);

	virtual void getParameter(real_t *targets);

	virtual real_t calculateError(real_vector &targets);
	
	virtual void computeBackward(real_vector &targets);

	virtual void initPreOutput(const cpu_real_vector &mVec, const cpu_real_vector &vVec);
	
	virtual bool flagValid();
    };

    /********************************************************
     MDNUnit_mixture: 
        MDNUnits -> MDNUnit_mixture
    ********************************************************/    
    // MDN mixture unit
    template <typename TDevice>
    class MDNUnit_mixture : public MDNUnit<TDevice>
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;
	
    protected:
	// data
	const int   m_numMixture;
	const int   m_featureDim;
	
	real_t      m_varFloor;  // all dimensions and all mixtures share the same variance floor
	bool        m_tieVar;    // whether the variance should be tied across dimension ?

	real_vector m_offset;    // temporary, calculate the offset for mixture weight
	real_vector m_tmpPat;    // temporary, store the statistics for BP
	real_vector m_varBP;     // temporary, store the variance gradients for BP

    public:
	MDNUnit_mixture(int startDim, int endDim, int startDimOut, int endDimOut, 
			int type, int featureDim, Layer<TDevice> &precedingLayer, 
			int outputSize, const bool tieVar, const int trainable = MDNUNIT_TYPE_0);

	virtual ~MDNUnit_mixture();

	virtual void computeForward();
	
	virtual void getOutput(const real_t para, real_vector &targets);

	virtual void getEMOutput(const real_t para, real_vector &targets); // EM MOPG output from

	virtual void getParameter(real_t *targets);
	
	virtual real_t calculateError(real_vector &targets);
	
	virtual void computeBackward(real_vector &targets);

	virtual void initPreOutput(const cpu_real_vector &mVec, const cpu_real_vector &vVec);
	
	virtual bool flagVariance() const;
	
	virtual bool flagValid();
    };    


    /********************************************************
     MDNUnit_mixture_dyn: 
        mean_t_i^k = (mean_t_i^k + W^T_k o_(t-1))i). 
        Add one step dynamic features here
        MDNUnits -> MDNUnit_mixture -> MDNUnit_mixture_dyn
    ********************************************************/    
    // MDN mixture unit
    template <typename TDevice>
    class MDNUnit_mixture_dyn : public MDNUnit_mixture<TDevice>
    {	
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;

    protected:
	int          m_weightStart;         // where is the first parameter of this unit is 
	                                    //  the shared weight vector offered by MDNLayer ?
	int          m_weightNum;           // number of trainable weights
        real_t      *m_weightsPtr;          // pointer to the start of the weights 
	real_vector *m_weights;             // pointer to the shared weight vector
	real_t      *m_weightUpdatesPtr;    // pointer to the start of the gradident vector
	real_vector *m_weightUpdates;       // pointer to the shared gradient vector

	real_vector  m_dataBuff;            // temporary, store the time shifted target vector,
	                                    //  the transformed but W^T_k o(t-1) + b and statistics
	                                    //  for BP
	real_vector  m_wTransBuff;          // the buffer to store tanh(w) and tanh(w)*tanh(w)
	int          m_tanhReg;             // whether use tanh to transform the weight as tanh(w)


	real_vector  m_oneVec;              // temporary, a vector of 1.0, used in BP
	int          m_maxTime;             // maximum length of the data utterance
	int          m_paral;               // number of utterances in parallel 
	int          m_totalTime;           // m_curMaxLength * m_paral
	int          m_backOrder;           // y[t] - y[t-1] - ... - [y-m_backOrder]
	int          m_arrmdnLearning;      
	// Add 0822
	int          m_dynDirection;        // 0: along the time axis (default)
	                                    // 1: along the dimension axis
	                                    // 2: along the time and dimension axes
	int          m_linearPartLength;    // the length of the linear scale parameter
	int          m_biasPartLength;      // the length of the bias part
	int          m_weightShiftToDim;    // shift to the pointer of AR for dimension
	int          m_wTransBuffShiftToDim;// shift in wTransBuff

	
    public:
	MDNUnit_mixture_dyn(int startDim, int endDim, int startDimOut, int endDimOut, 
			    int type, Layer<TDevice> &precedingLayer, int outputSize,
			    const bool tieVar, int weightStart, int weightNum,
			    int backOrder,
			    const int trainable    = MDNUNIT_TYPE_1,
			    const int dynDirection = MDNUNIT_TYPE_1_DIRECT);

	virtual ~MDNUnit_mixture_dyn();
		
	virtual real_t calculateError(real_vector &targets);
	
	virtual void computeForward();
	
	virtual void computeBackward(real_vector &targets);
	
	virtual void getOutput(const real_t para, real_vector &targets);

	virtual void getEMOutput(const real_t para, real_vector &targets); // EM MOPG output from

	virtual void getParameter(real_t *targets);
	
	// link the wegith for trainable
	virtual void linkWeight(real_vector& weights, real_vector& weightsUpdate);     

	virtual bool flagValid();
    };    


    /********************************************************
     * MDNUnit_mixture_dyn: 
     *  mean_t_i^k = (mean_t_i^k + W^T_k o_(t-1))i). 
     *  different from the MDNUnit_mixture_dynSqr, the W and b
     *  are trainable and predicted by the previous NN
     *
     *  MDNUnits -> MDNUnit_mixture -> MDNUnit_mixture_dynSqr
    ********************************************************/    
    template <typename TDevice>
    class MDNUnit_mixture_dynSqr : public MDNUnit_mixture<TDevice>
    {	
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;

    protected:
	int        m_a_pos;                 // the start position of the trainable a
	int        m_b_pos;                 // the start position of the trainable b
	int        m_backOrder;           // y[t] - y[t-1] - ... - [y-m_backOrder]
	int        m_tanhReg;             // whether use tanh to transform the weight as tanh(w)

    public:

	MDNUnit_mixture_dynSqr(int startDim, int endDim, int startDimOut, int endDimOut, 
			       int type, Layer<TDevice> &precedingLayer, int outputSize,
			       const bool tieVar, const int backOrder, 
			       const int trainable = MDNUNIT_TYPE_2);
	
	virtual       ~MDNUnit_mixture_dynSqr();
	
	virtual void computeForward();
	
	virtual real_t calculateError(real_vector &targets);
	
	virtual void   computeBackward(real_vector &targets);
	
	virtual void   getOutput(const real_t para, real_vector &targets);

	virtual void   getEMOutput(const real_t para, real_vector &targets);

	virtual void   getParameter(real_t *targets);
	
	virtual bool   flagValid();

	virtual void   initPreOutput(const cpu_real_vector &mVec, const cpu_real_vector &vVec);
    };
    
}

#endif
