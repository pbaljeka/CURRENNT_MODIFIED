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
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <fstream>
#include <cmath>


#define MIXTUREDYN_INITVARIANCE 0.01

namespace layers {
    

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
	
	// get the flag for variance tying
	m_tieVarMDNUnit = config.getTiedVariance();
	
	// tmp Buff to read the configuration
	Cpu::int_vector flagTieVariance;
	Cpu::int_vector flagTrainable;
	Cpu::int_vector flagTrainable_arg;
	flagTieVariance.clear();
	flagTrainable.clear();
	
	/******************* Read config ********************/
	// read in the configuration from .autosave 
	if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) {
	    const rapidjson::Value &weightsChild = (*weightsSection)[this->name().c_str()];
            if (!weightsChild.IsObject())
                throw std::runtime_error(std::string("Weights section for layer '") + 
					 this->name() + "' is not an object");
	    if (!weightsChild.HasMember("config") || !weightsChild["config"].IsArray())
                throw std::runtime_error(std::string("Missing array 'config/") + 
					 this->name() + "/config'");
	    
	    // read in the mdnConfig vector
            const rapidjson::Value &inputWeightsChild    = weightsChild["config"];
            m_mdnConfigVec.reserve(inputWeightsChild.Size());;
	    for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); 
		 it != inputWeightsChild.End(); ++it)
                m_mdnConfigVec.push_back(static_cast<real_t>(it->GetDouble()));
            numEle = m_mdnConfigVec[0];
	    
	    // read in the flagTieVariance vector
	    if (weightsChild.HasMember("tieVarianceFlag") && 
		weightsChild["tieVarianceFlag"].IsArray()){
		const rapidjson::Value &BuftieVarianceFlag = weightsChild["tieVarianceFlag"];
		for (rapidjson::Value::ConstValueIterator it = BuftieVarianceFlag.Begin(); 
		     it != BuftieVarianceFlag.End(); ++it)
		    flagTieVariance.push_back(static_cast<int>(it->GetInt()));
	    }

	    // read in the trainable type vector
	    if (weightsChild.HasMember("trainableFlag") && 
		weightsChild["trainableFlag"].IsArray()){
		const rapidjson::Value &BuftieVarianceFlag = weightsChild["trainableFlag"];
		for (rapidjson::Value::ConstValueIterator it = BuftieVarianceFlag.Begin(); 
		     it != BuftieVarianceFlag.End(); ++it)
		    flagTrainable.push_back(static_cast<int>(it->GetInt()));
	    }
	// read in the configuration from mdn_config (binary file)
        }else{
	    std::ifstream ifs(config.mdnFlagPath().c_str(), 
			      std::ifstream::binary | std::ifstream::in);
	    if (!ifs.good())
		throw std::runtime_error(std::string("Can't open MDNConfig:"+config.mdnFlagPath()));
	    
	    std::streampos numEleS, numEleE;
	    numEleS = ifs.tellg();
	    ifs.seekg(0, std::ios::end);
	    numEleE = ifs.tellg();
	    // get the total number of parameter
	    long int tmpnumEle  = (numEleE-numEleS)/sizeof(real_t);
	    ifs.seekg(0, std::ios::beg);
	
	    // get the total number of MDNUnit
	    real_t tempVal;
	    ifs.read((char *)&tempVal, sizeof(real_t));
	    numEle = (int)tempVal;                      
	    if (tmpnumEle != (numEle*5+1)){
		throw std::runtime_error("Number of parameter != 1st parameter * 5 + 1");
	    }	    

	    // get the configuration for each MDNUnit
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


	// check configuration
	if (m_mdnConfigVec.size() != numEle*5+1){
	    throw std::runtime_error("Error in reading the configuration of MDN");
	}
	if (flagTieVariance.size() != flagTrainable.size() ||
	    (flagTieVariance.size() >0 && flagTieVariance.size() != numEle)){
	    throw std::runtime_error("Error in tieVarianceFlag and trainableFlag (in model file)");
	}

	// read Trainable from input argument
	if (config.mdnDyn().size() > 0){
	    if (config.mdnDyn().size() != numEle){
		// num1_num2_num3 format
		std::vector<std::string> tempArgs;
		boost::split(tempArgs, config.mdnDyn(), boost::is_any_of("_"));
		if (tempArgs.size() != numEle){
		    printf("mdnDyn length: %d, MDNUnits %d\n", 
			   (int)tempArgs.size(), (int)numEle);
		    throw std::runtime_error("Error in mdnDyn");
		}
		flagTrainable_arg.resize(config.mdnDyn().size(), 0);
		for (int i=0; i < tempArgs.size(); i++)
		    flagTrainable_arg[i] = boost::lexical_cast<int>(tempArgs[i]);
		
	    }else{
		flagTrainable_arg.resize(config.mdnDyn().size(), 0);
		for (int i=0; i < config.mdnDyn().size(); i++)
		    flagTrainable_arg[i] = config.mdnDyn()[i] - '0';
	    }
	}else{
	    // default, make it all nontrainable unit
	    flagTrainable_arg.resize(numEle, 0);
	}

	printf("\n");	
	int weightsNum = 0;
	this->m_trainable = false;

	// create the MDNUnits
	for (int i=0; i<numEle; i++){
	    unitS    = (int)m_mdnConfigVec[1+i*5];  // start dimension in output of previous layer
	    unitE    = (int)m_mdnConfigVec[2+i*5];  // end dimension in output of previous layer
	    unitSOut = (int)m_mdnConfigVec[3+i*5];  // start dimension in output feature
	    unitEOut = (int)m_mdnConfigVec[4+i*5];  // end dimension in output feature
	    mdnType  = (int)m_mdnConfigVec[5+i*5];  // MDNUnit type
	    
	    // binomial distribution (parametrized as sigmoid function)
	    if (mdnType==MDN_TYPE_SIGMOID){
		mdnUnit = new MDNUnit_sigmoid<TDevice>(unitS, unitE, unitSOut, unitEOut, mdnType, 
						       precedingLayer, this->size());
		m_mdnParaDim += (unitE - unitS);
		outputSize += (unitE - unitS);
		printf("MDN sigmoid\n");
		
	    // multi-nomial distribution (parameterized by softmax function)
	    }else if(mdnType==MDN_TYPE_SOFTMAX){
		mdnUnit = new MDNUnit_softmax<TDevice>(unitS, unitE, unitSOut, unitEOut, mdnType, 
						       precedingLayer, this->size());
		m_mdnParaDim += (unitE - unitS);
		outputSize += 1;
		printf("MDN softmax\n");

	    // Gaussian mixture distribution
	    }else if(mdnType > 0){
		
		// the MDNUnit mixture dynamic unit (either specified by --mdnDyn or model option)
		// if the model (.autosave) has specified it, ignores the arguments
		bool tmpTieVarianceFlag = ((flagTieVariance.size()>0) ?
					   (flagTieVariance[i]>0)     :
					   (m_tieVarMDNUnit));

		// the Trainable flag, either by flagTrainable in model or arguments
		// if the model (.autosave) has specified it, ignores the arguments
		int  tmpTrainableFlag   = ((flagTrainable.size()>0)   ?
					   (flagTrainable[i])       :
					   (flagTrainable_arg[i]));
		
		printf("MDN mixture: trainable: %2d, tieVariance %d, #parameter ", 
		       tmpTrainableFlag, tmpTieVarianceFlag);
		int featureDim;   
		int thisWeightNum; 

		switch (tmpTrainableFlag)
		{	
		    // Due to historical reason, the coding here is quite complex
		    // the original scheme
		    // 0 : non-trainable mixture
		    // 1 : 1st AR, time axis
		    // 2 : dynamic AR, parameter predicted by network
		    // 3 : 2st AR, time axis
		    // 4-5: 1-2 AR, dimension axis
		    // 6-7: 1-2 AR, dimension and time axis
		    // else: AR, time axis
		    
		    // trainable unit with dynamic link (weight predicted by the network)
		    case MDNUNIT_TYPE_2:
			mdnUnit = new MDNUnit_mixture_dynSqr<TDevice>(
					unitS, unitE, unitSOut, unitEOut, mdnType, 
					precedingLayer, this->size(), 
					tmpTieVarianceFlag, 2);
			printf("%d, with dynamic link\n", 0);
			break;

		    // conventional non-trainable unit
		    case MDNUNIT_TYPE_0: 
			featureDim    = unitEOut - unitSOut;
			mdnUnit = new MDNUnit_mixture<TDevice>(
					unitS, unitE, unitSOut, unitEOut, mdnType, 
					featureDim, precedingLayer, this->size(), 
					tmpTieVarianceFlag);
			printf("%d\n", 0);
			break;

		    // MDN with time-invariant AR
		    default:
			int dynDirection;
			int lookBackStep;
			
			if (tmpTrainableFlag < 0){
			    printf("The value in --mdnDyn can only be [0-9]");
			    throw std::runtime_error("Error configuration --mdnDyn");
			}else if (tmpTrainableFlag <= 3){
			    // AR along the time axis
			    dynDirection = MDNUNIT_TYPE_1_DIRECT;
			    // tmpTrainableFlag = 1 or 2
			    lookBackStep = (tmpTrainableFlag==1)?(1):(tmpTrainableFlag-1);
			}else if (tmpTrainableFlag <= 5){
			    // AR along the dimension axis
			    dynDirection = MDNUNIT_TYPE_1_DIRECD;
			    lookBackStep = tmpTrainableFlag - 3;
			}else if (tmpTrainableFlag <= 7){
			    // AR long both time and dimension axes
			    dynDirection = MDNUNIT_TYPE_1_DIRECB;
			    lookBackStep = tmpTrainableFlag - 5;
			}else{
			    // AR along the time axis
			    dynDirection = MDNUNIT_TYPE_1_DIRECT;
			    lookBackStep = tmpTrainableFlag - 5;
			}
			
			// create the trainable unit
		        // case MDNUNIT_TYPE_1:	
			featureDim    = unitEOut - unitSOut;
			thisWeightNum = layers::MixtureDynWeightNum(featureDim, 
								    mdnType, 
								    lookBackStep,
								    dynDirection);
			mdnUnit = new MDNUnit_mixture_dyn<TDevice>(
					unitS, unitE, unitSOut, unitEOut, mdnType, 
					precedingLayer, this->size(), 
					tmpTieVarianceFlag, weightsNum, thisWeightNum,
					lookBackStep, tmpTrainableFlag, dynDirection);
			weightsNum += thisWeightNum;
			this->m_trainable = true;
			printf("%-8d, AR order and direction: %d %d", 
			       thisWeightNum, lookBackStep, dynDirection);
			if (config.tanhAutoregressive())
			    printf(", with tanh-based model");
			printf("\n");
			break;
		}
		m_mdnParaDim += (unitE - unitS);
		
		// with dynamic link (time-variant AR)
		if (tmpTrainableFlag == MDNUNIT_TYPE_2){
		    if (tmpTieVarianceFlag){
			// K mixture weight, K*Dim mean, K*1 variance, Dim a, Dim b
			outputSize += ((unitE - unitS)-2*mdnType-3*(unitEOut - unitSOut))/mdnType;
		    }else{
			// K mixture weight, K*Dim mean, K*Dim variance, Dim a, Dim b
			outputSize += ((unitE - unitS)-mdnType-3*(unitEOut - unitSOut))/(2*mdnType);
		    }
		}else{
		    if (tmpTieVarianceFlag){
			// K mixture weight, K*Dim mean, K*1 variance. 
			outputSize += ((unitE - unitS) - 2*mdnType)/mdnType;
		    }else{
			// K mixture weight, K*Dim mean, K*Dim variance. 
			outputSize += ((unitE - unitS) - mdnType) / (2*mdnType);
		    }
		}
		if (!mdnUnit->flagValid()){
		    throw std::runtime_error("Fail to initialize mdnUnit");
		}

	    }else{
		throw std::runtime_error("mdnUnit type invalid (>0, 0, -1)");
	    }
	    m_mdnUnits.push_back(boost::shared_ptr<MDNUnit<TDevice> >(mdnUnit));
	}

	/********************  check               ****************/
	printf("MDN layer parameter number: %d\n", m_mdnParaDim);
	if (m_mdnParaDim != precedingLayer.size()){
	    printf("MDN parameter dim %d is not equal to NN output dimension %d\n", 
		   m_mdnParaDim, precedingLayer.size());
	    throw std::runtime_error("");
	}
	if (outputSize != this->size()){
	    printf("Mismatch between target dimension %d and MDN configuration %d\n", 
		   outputSize, this->size());
	    printf("Did you use --tieVariance false for untied variance of MDN?\n");
	    throw std::runtime_error("");
	}
	
	

	
	/********************  Initialize the weight ****************/
	cpu_real_vector weights;
	if (this->m_trainable){
	    m_trainableNum = weightsNum;
	    
	    if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())){
		if (!weightsSection->HasMember(this->name().c_str()))
		    throw std::runtime_error(std::string("Missing weights section for layer '") + 
					     this->name() + "'");
		const rapidjson::Value &weightsChild = (*weightsSection)[this->name().c_str()];
		if (!weightsChild.IsObject())
		    throw std::runtime_error(std::string("Weights section for layer '") + 
					     this->name() + "' is not an object");

		if (!weightsChild.HasMember("input") || !weightsChild["input"].IsArray())
		    throw std::runtime_error(std::string("Missing array 'weights/") + 
					     this->name() + "/input'");
		if (!weightsChild.HasMember("bias") || !weightsChild["bias"].IsArray())
		    throw std::runtime_error(std::string("Missing array 'weights/") + 
					     this->name() + "/bias'");
		if (!weightsChild.HasMember("internal") || !weightsChild["internal"].IsArray())
		    throw std::runtime_error(std::string("Missing array 'weights/") + 
					     this->name() + "/internal'");
        
		const rapidjson::Value &inputWeightsChild    = weightsChild["input"];
		const rapidjson::Value &biasWeightsChild     = weightsChild["bias"];
		const rapidjson::Value &internalWeightsChild = weightsChild["internal"];

		if (inputWeightsChild.Size() != weightsNum)
		    throw std::runtime_error(std::string("Invalid number of weights for layer '") 
					 + this->name() + "'");

		if (biasWeightsChild.Size() != 0)
		    throw std::runtime_error(std::string("bias part should be void for layer '") 
					     + this->name() + "'");

		if (internalWeightsChild.Size() != 0)
		    throw std::runtime_error(std::string("internal weights should be void layer'")
					 + this->name() + "'");
		
		weights.reserve(inputWeightsChild.Size() + 
				biasWeightsChild.Size()  + 
				internalWeightsChild.Size());

		for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); 
		     it != inputWeightsChild.End(); ++it)
		    weights.push_back(static_cast<real_t>(it->GetDouble()));
		
	    }else {
		// No other initialization methods implemented yet
		// Add 0923, we need random initialization here
		// The problem is that, for high order filter, we need to break the symmetry of
		// of the parameter
		weights.resize(weightsNum, 0.0);	
		if(config.arRMDNInitVar() > 0.0){
		    static boost::mt19937 *gen = NULL;
		    if (!gen) {
			gen = new boost::mt19937;
			gen->seed(config.randomSeed()+101);
		    }
		    boost::random::normal_distribution<real_t> dist(0.0, config.arRMDNInitVar());
		    for (size_t i = 0; i < weights.size(); ++i)
			weights[i] = dist(*gen);
		    printf("\nARRMDN para initialized as Gaussian noise (var: %f)", 
			   config.arRMDNInitVar());
		}else{
		    printf("\nARRMDN initialized as zero");
		}
	    }
	    printf("\nMDN trainable mixture is used. The number of parameter is %d\n", weightsNum);
	    m_sharedWeights       = weights;
	    m_sharedWeightUpdates = weights;
	    
	    // link the shared weights to each trainable MDNUnits
	    BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
		mdnUnit->linkWeight(m_sharedWeights, m_sharedWeightUpdates);
	    }
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
	
	
        // do nothing if we don't have any weights
	if (m_sharedWeights.empty()){
	    //weightsObject->AddMember(this->name().c_str(), weightsSection, allocator);
            //return;
	}else{

	    // create and fill the weight arrays
	    rapidjson::Value inputWeightsArray(rapidjson::kArrayType);
	    int inputWeightsCount = this->m_sharedWeights.size();
	    inputWeightsArray.Reserve(inputWeightsCount, allocator);
	    for (int i = 0; i < inputWeightsCount; ++i)
		inputWeightsArray.PushBack(m_sharedWeights[i], allocator);

	    rapidjson::Value biasWeightsArray(rapidjson::kArrayType);
	    int biasWeightsCount = 0;
	    biasWeightsArray.Reserve(biasWeightsCount, allocator);

	    rapidjson::Value internalWeightsArray(rapidjson::kArrayType);
	    //int internalWeightsCount = 0; 

	    // create and fill the weights subsection
	    weightsSection.AddMember("input",    inputWeightsArray,    allocator);
	    weightsSection.AddMember("bias",     biasWeightsArray,     allocator);
	    weightsSection.AddMember("internal", internalWeightsArray, allocator);
	    
	    
	}
	
	// Add additional options for MDN
	int mdnUnitCounts = this->m_mdnUnits.size();
	rapidjson::Value tieVariance(rapidjson::kArrayType);
	tieVariance.Reserve(mdnUnitCounts, allocator);
	rapidjson::Value trainableType(rapidjson::kArrayType);
	trainableType.Reserve(mdnUnitCounts, allocator);

	BOOST_FOREACH (const boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    tieVariance.PushBack((int)mdnUnit->flagVariance(), allocator);
	    trainableType.PushBack(mdnUnit->flagTrainable(), allocator);
	}
	weightsSection.AddMember("tieVarianceFlag",  tieVariance, allocator);
	weightsSection.AddMember("trainableFlag",  trainableType, allocator);
	
	// add the weights section to the weights object
	weightsObject->AddMember(this->name().c_str(), weightsSection, allocator);
	return;
    }
    
    template <typename TDevice>
    void MDNLayer<TDevice>::exportWeights(const helpers::JsonValue &weightsObject, 
					  const helpers::JsonAllocator &allocator) const
    {
	// we use exportConfig above instead of exportWeights to dump the weight
        if (!weightsObject->IsObject())
            throw std::runtime_error("The JSON value is not an object");
    }

    template <typename TDevice>
    void MDNLayer<TDevice>::reReadWeight(const helpers::JsonValue &weightsSection, 
					 const int readCtrFlag)
    {
	Cpu::real_vector weights;
	if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())){
	    const rapidjson::Value &weightsChild = (*weightsSection)[this->name().c_str()];
            if (!weightsChild.IsObject())
                throw std::runtime_error(std::string("Weights section for layer '") + 
					 this->name() + "' is not an object");
            if (!weightsChild.HasMember("input") || !weightsChild["input"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + 
					 this->name() + "/input'");
            if (!weightsChild.HasMember("bias") || !weightsChild["bias"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + 
					 this->name() + "/bias'");
            if (!weightsChild.HasMember("internal") || !weightsChild["internal"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + 
					 this->name() + "/internal'");
	    const rapidjson::Value &inputWeightsChild    = weightsChild["input"];
            const rapidjson::Value &biasWeightsChild     = weightsChild["bias"];
            const rapidjson::Value &internalWeightsChild = weightsChild["internal"];

            if (inputWeightsChild.Size() != m_trainableNum)
                throw std::runtime_error(std::string("Invalid number of input weights for layer '") 
					 + this->name() + "'");
            if (biasWeightsChild.Size() != 0)
                throw std::runtime_error(std::string("Invalid number of bias weights for layer '") 
					 + this->name() + "'");
            if (internalWeightsChild.Size() != 0)
                throw std::runtime_error(std::string("Invalid number of internal for layer '") 
					 + this->name() + "'");

            weights.reserve(inputWeightsChild.Size());
            for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); 
		 it != inputWeightsChild.End(); 
		 ++it)
                weights.push_back(static_cast<real_t>(it->GetDouble()));	    
	    m_sharedWeights       = weights;
	    m_sharedWeightUpdates = weights;
	    
	}else{
	    throw std::runtime_error(std::string("Can't find layer:")+this->name().c_str());
	}
    }


    template <typename TDevice>
    void MDNLayer<TDevice>::getOutput(const real_t para)
    {
	// Modify 05-24 Add support to EM-style generation
	if (para < -3.0)
	{
	    throw std::runtime_error("Parameter to MDN->getOutput can't be less than -2.0");
	}
	else if (para >= 0.0)
	{
	    // sampling from the distribution
	    bool tmpFlag = true;
	    BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits)
	    {
		mdnUnit->getOutput(para, (this->_targets()));
		if(tmpFlag){
		    if (mdnUnit->varScale().size()>0)
			printf("sampling with variance scaled by varVector");
		    else
			printf("sampling with variance scaled by %f", para);
		    tmpFlag =false;
		}
	    }	     
	}
	else if (para > -1.50)
	{
	    // output the data parameter
	    printf("generating the parameters of MDN");
	    this->m_mdnParaVec.resize(this->m_mdnParaDim*
				      this->precedingLayer().curMaxSeqLength()*
				      this->precedingLayer().parallelSequences(), 
				      0.0);
	    BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits)
	    {
		mdnUnit->getParameter(helpers::getRawPointer(this->m_mdnParaVec));
	    }
	}
	else
	{
	    // EM generation
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
    
    template <typename TDevice>
    void MDNLayer<TDevice>::reInitWeight()
    {
	if (this->m_trainable){
	    thrust::fill(m_sharedWeights.begin(), m_sharedWeights.end(), 0.0);
	    thrust::fill(m_sharedWeightUpdates.begin(), m_sharedWeightUpdates.end(), 0.0);
	}
    }
    
    template <typename TDevice>
    MDNLayer<TDevice>::real_vector& MDNLayer<TDevice>::weights()
    {
	return m_sharedWeights;
    }

    template <typename TDevice>
    const MDNLayer<TDevice>::real_vector& MDNLayer<TDevice>::weights() const
    {
	return m_sharedWeights;
    }

    template <typename TDevice>
    MDNLayer<TDevice>::real_vector& MDNLayer<TDevice>::weightUpdates()
    {
	return m_sharedWeightUpdates;
    }
    
    template <typename TDevice>
    bool MDNLayer<TDevice>::flagTrainable() const
    {
	return m_trainable;
    }
    
    template <typename TDevice>
    void MDNLayer<TDevice>::setCurrTrainingEpoch(const int curTrainingEpoch)
    {
	Layer<TDevice>::setCurrTrainingEpoch(curTrainingEpoch);
	BOOST_FOREACH (boost::shared_ptr<MDNUnit<TDevice> > &mdnUnit, m_mdnUnits){
	    mdnUnit->setCurrTrainingEpoch(curTrainingEpoch);
	}
    }
    
    template <typename TDevice>
    int& MDNLayer<TDevice>::getCurrTrainingEpoch()
    {
	return Layer<TDevice>::getCurrTrainingEpoch();
    }

    template class MDNLayer<Cpu>;
    template class MDNLayer<Gpu>;

}
