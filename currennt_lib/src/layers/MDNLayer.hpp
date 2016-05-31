
#ifndef LAYERS_MDNOUTPUTLAYER_HPP
#define LAYERS_MDNOUTPUTLAYER_HPP


#include "PostOutputLayer.hpp"
#include <boost/shared_ptr.hpp>


#define MDN_TYPE_SIGMOID 0
#define MDN_TYPE_SOFTMAX -1

namespace layers {
    
    /********************************************************
     MDNUnit: describes the distribution of the target data
     based on parameters given by NN
       
        MDNUnits 
            ^
            |
        MDNLayer
    ********************************************************/
    // virtual class of MDNUnit
    template <typename TDevice>
    class MDNUnit
    {
	typedef typename TDevice::real_vector real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;
    protected:
	// with respect to the input layer (last hidden layer of NN)
	const int   m_startDim;            // start dimension of data  in the input vector
	const int   m_endDim;              // 
                                           // input for this unit is between m_startDim and m_endDim
	const int   m_layerSizeIn;         // dimension of the input vector (size of previous layer)

	const int   m_paraDim;             // dimension of the parameter of this unit
	real_vector m_paraVec;             // the parameter of current unit (after conversion)
	real_t     *m_paraPtr;             // pointer to the parameter 
	Layer<TDevice> &m_precedingLayer;  // previous output layer (output of the network)

	// w.r.t the target data (target data of NN)
	const int   m_startDimOut;         // start dimension of the target data
	const int   m_endDimOut;           // 
	const int   m_layerSizeTar;        // dimension of the target data (in total)
	real_t     *m_targetPtr;           // pointer to the target data
	                                   // 
	// other
	const int   m_type;                // type of this unit
	real_vector m_mdnOutput;           // the output of processing (sampling)


    public:
	MDNUnit(int startDim, int endDim, int startDimOut, int endDimOut, 
		int type, int paraDim, Layer<TDevice> &precedingLayer, int outputSize);

	virtual ~MDNUnit();

	// methods
	const int&   paraDim() const; 
	
	// pure vitural function (must be over-written)
	virtual void computeForward() =0;  // transform the previous output into MDN parameters
	
	virtual void getOutput(const real_t para, real_t *targets) =0; // sampling output from MDN

	virtual void getEMOutput(const real_t para, real_vector &targets) =0; // EM MOPG output

	virtual void getParameter(real_t *targets) =0; // output the parmeter of this unit

	virtual real_t calculateError(real_vector &targets) =0;// the error(-log likelihood)
	
	virtual void computeBackward(real_vector &targets) =0; // back ward computation

	virtual void initPreOutput(const cpu_real_vector &mVec, const cpu_real_vector &vVec) =0;
	
    };

    /********************************************************
     MDNUnit_sigmoid: elementwise binary distribution
       p(t^(n,t)_d | x^(n, t)) = sigmoid(NN(x^(n,t)))
       
        MDNUnits -> MDNUnit_sigmoid
            ^
            |
        MDNLayer
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
			int type, Layer<TDevice> &precedingLayer, int outputSize);

	virtual ~MDNUnit_sigmoid();

	virtual void computeForward();
	
	virtual void getOutput(const real_t para,real_t *targets);
	
	virtual void getEMOutput(const real_t para, real_vector &targets); // EM MOPG output from  

	virtual void getParameter(real_t *targets);

	virtual real_t calculateError(real_vector &targets);
	
	virtual void computeBackward(real_vector &targets);

	virtual void initPreOutput(const cpu_real_vector &mVec, const cpu_real_vector &vVec);
    };

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
			int type, Layer<TDevice> &precedingLayer, int outputSize);

	virtual ~MDNUnit_softmax();

	virtual void computeForward();
	
	virtual void getOutput(const real_t para,real_t *targets);

	virtual void getEMOutput(const real_t para, real_vector &targets); // EM MOPG output from  

	virtual void getParameter(real_t *targets);

	virtual real_t calculateError(real_vector &targets);
	
	virtual void computeBackward(real_vector &targets);

	virtual void initPreOutput(const cpu_real_vector &mVec, const cpu_real_vector &vVec);
	
    };

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
	real_vector m_offset;
	real_vector m_tmpPat;
	real_vector m_varBP;     // vector to store the variance per featuredim/mixture/time
	real_t      m_varFloor;  // all mixture share the same variance floor

    public:
	MDNUnit_mixture(int startDim, int endDim, int startDimOut, int endDimOut, 
			int type, Layer<TDevice> &precedingLayer, int outputSize);

	virtual ~MDNUnit_mixture();

	virtual void computeForward();
	
	virtual void getOutput(const real_t para, real_t *targets);

	virtual void getEMOutput(const real_t para, real_vector &targets); // EM MOPG output from

	virtual void getParameter(real_t *targets);
	
	virtual real_t calculateError(real_vector &targets);
	
	virtual void computeBackward(real_vector &targets);

	virtual void initPreOutput(const cpu_real_vector &mVec, const cpu_real_vector &vVec);
	
    };    


    /********************************************************
     MDNLayer: 
        1. forward computation: transform the output of NN
           into statistical distribution
        2. calculate the likelihood of training data
        3. back-propagation
        4. sampling (predict) output

        MDNUnits 
            ^
            |
        MDNLayer
            ^
            |
     PostOutputLayer
    ********************************************************/    
    // MDN layer definition
    template <typename TDevice>
    class MDNLayer : public PostOutputLayer<TDevice>
    {

	typedef typename TDevice::real_vector    real_vector;
	typedef typename Cpu::real_vector cpu_real_vector;

    protected:
	cpu_real_vector m_mdnVec;        // the vector of mdnunit flag
	cpu_real_vector m_mdnConfigVec;  // vector of the mdn configuration
	real_vector m_mdnParaVec;        // vector of parameters of all MDNUnits
	// the vector of MDNUnit for computation
	std::vector<boost::shared_ptr<MDNUnit<TDevice> > > m_mdnUnits;  
	int m_mdnParaDim;
	


    public:
	MDNLayer(
		 const helpers::JsonValue &layerChild,
		 const helpers::JsonValue &weightsSection,
		 Layer<TDevice> &precedingLayer
		 );
	
	virtual ~MDNLayer();
	
	virtual const std::string& type() const;
	
	virtual real_t calculateError();
	
	virtual void computeForwardPass();
	
	virtual void computeBackwardPass();
	
	virtual real_vector& mdnParaVec();
	
	virtual int mdnParaDim();
	
	virtual void initPreOutput(const cpu_real_vector &mVec, const cpu_real_vector &vVec);
	
	virtual cpu_real_vector getMdnConfigVec();

	void getOutput(const real_t para);

	void exportConfig(const helpers::JsonValue &weightsObject, 
			  const helpers::JsonAllocator &allocator) const;
    };

} // namespace layers

#endif //
