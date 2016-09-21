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

#include "InputLayer.hpp"
#include "../Configuration.hpp"

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <boost/lexical_cast.hpp>
#include <thrust/transform.h>
#include <stdexcept>
#include <fstream>

namespace layers {

    template <typename TDevice>
    InputLayer<TDevice>::InputLayer(const helpers::JsonValue &layerChild, int parallelSequences, int maxSeqLength)
        : Layer<TDevice>(layerChild, parallelSequences, maxSeqLength)
	, m_weDim(-1)
	, m_flagWeUpdate(false)
    {
    }

    template <typename TDevice>
    InputLayer<TDevice>::~InputLayer()
    {
    }

    template <typename TDevice>
    const std::string& InputLayer<TDevice>::type() const
    {
        static const std::string s("input");
        return s;
    }

    template <typename TDevice>
    void InputLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
	if (m_flagWeUpdate){
	    if (m_weIDDim > fraction.inputPatternSize()){
		throw std::runtime_error("we dimension is larger than input data dimension");
	    }
	    if (this->size() != fraction.inputPatternSize()-1+m_weDim){
		throw std::runtime_error("input data pattern -1 + weDim != networkinput");
	    }
	}else if (fraction.inputPatternSize() != this->size()) {
            throw std::runtime_error(std::string("Input layer size of ") + 
				     boost::lexical_cast<std::string>(this->size()) + 
				     " != data input pattern size of " + 
				     boost::lexical_cast<std::string>(fraction.inputPatternSize()));
        }

        Layer<TDevice>::loadSequences(fraction);
	
	/* Add 16-02-22 Wang: for WE updating */
	// thrust::copy(fraction.inputs().begin(),fraction.inputs().end(),this->_outputs().begin());
	
	if (m_flagWeUpdate){
	    
	    // load in the embedded vectors from weBank
	    int weidx=0;
	    long unsigned int bias=0;
	    long unsigned int fracTime=(fraction.inputs().size()/fraction.inputPatternSize());
	    
	    if (fracTime>m_weIdx.size()){
		throw std::runtime_error("m_weIdx size is smaller than fracTime\n");
	    }
	    thrust::fill(m_weIdx.begin(), m_weIdx.end(), -1);

	    
	    Cpu::real_vector tempInput;
	    tempInput.resize(this->size(), 0.0);

	    for (int i=0; i<fracTime; i++){
		bias = i*fraction.inputPatternSize();
		
		// copy the original input data
		thrust::copy(fraction.inputs().begin()+bias, 
			     fraction.inputs().begin()+bias+fraction.inputPatternSize(), 
			     tempInput.begin());
		
		// retrieve the embedded vector idx and save m_weIdx
		weidx = (long unsigned int)(fraction.inputs()[i * fraction.inputPatternSize() + 
							      m_weIDDim]);
		if (weidx*m_weDim>m_weBank.size()){
		    printf("Vector idx: %d\t", weidx);
		    throw std::runtime_error("vector idx larger than weBank size");
		}
		// the number of valid m_weIdx is always equal to fracTime
		m_weIdx[i]=weidx;
		
		// copy the we data into the input data (output of the InputLayer)
		thrust::copy(m_weBank.begin()  + weidx     * m_weDim, 
			     m_weBank.begin()  + (weidx+1) * m_weDim, 
			     tempInput.begin() +  
			     fraction.inputPatternSize()  - 1);
		
		// Add 0902: add noise to the input
		// Note: this is different from the input_noise_sigma
		//       here, the noise will be added every epoch
		if (this->m_weNoiseStartDim > 0){		    
		    if (this->m_weNoiseStartDim >= this->size() ||
			this->m_weNoiseEndDim   >  this->size()){
			throw std::runtime_error("weNoiseDimenion error");
		    }
		    const Configuration &config = Configuration::instance();	    
		    static boost::mt19937 *gen = NULL;
		    if (!gen) {
			gen = new boost::mt19937;
			gen->seed(config.randomSeed()+100);
		    }
		    boost::random::normal_distribution<real_t> dist(0.0, this->m_weNoiseDev);
		    for (size_t j = this->m_weNoiseStartDim; j < this->m_weNoiseEndDim; ++j)
			tempInput[j] += dist(*gen);;
		}

		// copy the we data into the input data (output of the InputLayer)
		thrust::copy(tempInput.begin(), tempInput.end(),
			     this->_outputs().begin()+i*this->size());
		
	    }
	    // for debugging
	    if (0){
		Cpu::real_vector tempVec(this->_outputs());
		std::cout << tempVec.size() << std::endl;
	    }
	}else
	    // if no we is utilized, just copy the input
	    thrust::copy(fraction.inputs().begin(),
			 fraction.inputs().end(),
			 this->_outputs().begin());
    }

    template <typename TDevice>
    void InputLayer<TDevice>::computeForwardPass()
    {
    }

    template <typename TDevice>
    void InputLayer<TDevice>::computeBackwardPass()
    {
    }


    /* Add 16-02-22 Wang: for WE updating */
    // return the reference to m_weBank;
    template <typename TDevice>
    Cpu::real_vector& InputLayer<TDevice>::_weBank(){
	return m_weBank;
    }
    template <typename TDevice>
    Cpu::real_vector& InputLayer<TDevice>::_weIdx(){
	return m_weIdx;
    }

    // return the m_weDim
    template <typename TDevice>
    unsigned int& InputLayer<TDevice>::_weDim(){
	return m_weDim;
    }
    template <typename TDevice>
    unsigned int& InputLayer<TDevice>::_weIDDim(){
	return m_weIDDim;
    }
    
    // read the we data into m_weBank
    template <typename TDevice>
    bool InputLayer<TDevice>::readWeBank(const std::string weBankPath, 
					 const unsigned dim, const unsigned dimidx, 
					 const unsigned maxLength)
    {
	// 
	if (dim < 1){
	    throw std::runtime_error(std::string("Dimention of weBank below 1"));
	}	
	std::ifstream ifs(weBankPath.c_str(), std::ifstream::binary | std::ifstream::in);
	if (!ifs.good()){
	    throw std::runtime_error(std::string("Fail to open ")+weBankPath);
	}
	
	// save the information
	m_weDim                 = dim;
	m_flagWeUpdate          = true;
	m_weIDDim               = dimidx;
	
	// set the flag for We input
	this->_setInputWeUpdate(true);  
	
	// get the number of we data
	std::streampos numEleS, numEleE;
	long int numEle;
	numEleS = ifs.tellg();
	ifs.seekg(0, std::ios::end);
	numEleE = ifs.tellg();
	numEle  = (numEleE-numEleS)/sizeof(real_t);
	ifs.seekg(0, std::ios::beg);
	
	// read in the data
	m_weBank = Cpu::real_vector(numEle, 0);
	real_t tempVal;
	std::vector<real_t> tempVec;
	for (unsigned int i = 0; i<numEle; i++){
	    ifs.read ((char *)&tempVal, sizeof(real_t));
	    tempVec.push_back(tempVal);
	}
	thrust::copy(tempVec.begin(), tempVec.end(), m_weBank.begin());
	std::cout << "Read " << numEle/dim << " vectors" << std::endl;
	
	// to store the word vector sequences for each frame
	m_weIdx    = Cpu::real_vector(maxLength, -1);
	ifs.close();
	return true;
    }
    
    template <typename TDevice>
    bool InputLayer<TDevice>::flagInputWeUpdate()
    {
	return m_flagWeUpdate;
    }

    template <typename TDevice>
    bool InputLayer<TDevice>::saveWe(const std::string weFile)
    {
	if (m_flagWeUpdate && m_weBank.size()>0){
	    std::ofstream ofs(weFile.c_str(), std::ofstream::binary);
	    if (!ofs.good()){
		std::cout << "Fail to open " << weFile << std::endl;
		return false;
	    }
	    // we assume it is a CPU vector
	    std::vector<real_t> tempVec(m_weBank.begin(), m_weBank.end());
	    for(int i=0; i<tempVec.size(); i++){
		ofs.write((char *)&(tempVec[i]), sizeof(real_t));
	    }
	    ofs.close();
	}else{
	    std::cout << "No WeBank is available " << std::endl;
	    return false;
	}
	return true;
    }

    template <typename TDevice>
    void InputLayer<TDevice>::reInitWeight()
    {
	// nothing to be done here
    }
    
    template <typename TDevice>
    bool InputLayer<TDevice>::initWeNoiseOpt(const int weNoiseStartDim, const int weNoiseEndDim,
					     const real_t weNoiseDev)
    {
	this->m_weNoiseStartDim = weNoiseStartDim;
	this->m_weNoiseEndDim   = weNoiseEndDim;
	this->m_weNoiseDev      = weNoiseDev;
	if (this->m_weNoiseStartDim > this->m_weNoiseEndDim){
	    printf("Error: this->m_weNoiseStartDim > this->m_weNoiseEndDim\n");
	    return false;
	}
	if (this->m_weNoiseDev < 0.0){
	    printf("Error: this->m_weNoiseDev < 0.0 \n");
	    return false;
	}
	if (this->m_weNoiseStartDim > 0){
	    printf("WE noise: from %d to %d, %f\n", weNoiseStartDim, weNoiseEndDim, weNoiseDev);
	}
	return true;
    }
    
    // explicit template instantiations
    template class InputLayer<Cpu>;
    template class InputLayer<Gpu>;

} // namespace layers
