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

#include "PostOutputLayer.hpp"

#include <boost/lexical_cast.hpp>
#include <stdexcept>

#include <fstream>
namespace layers {

    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_targets()
    {
        return this->outputs();
    }

    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_actualOutputs()
    {
        return m_precedingLayer.outputs();
    }

    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_outputErrors()
    {
        return m_precedingLayer.outputErrors();
    }
    
    /* Add 16-04-01 return the vector for mseWeight  */
    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_mseWeight()
    {
	return m_outputMseWeights;
    }

    template <typename TDevice>
    PostOutputLayer<TDevice>::PostOutputLayer(
        const helpers::JsonValue &layerChild, 
        Layer<TDevice> &precedingLayer,
        int requiredSize,
        bool createOutputs)
        : Layer<TDevice>  (layerChild, precedingLayer.parallelSequences(), 
			   precedingLayer.maxSeqLength(), createOutputs)
        , m_precedingLayer(precedingLayer)
    {
	// Modify 0506. For MDN, requireSize = -1, no need to check here
	// if (this->size() != requiredSize)
        if (requiredSize > 0 && this->size() != requiredSize)
            throw std::runtime_error("Size mismatch: " + 
				     boost::lexical_cast<std::string>(this->size()) + " vs. " + 
				     boost::lexical_cast<std::string>(requiredSize));
	
	/* Add 0401 wang */
	// assign the vector to output weights for RMSE
	// m_outputMseWeights = TDevice::real_vector(this->size(), (real_t)1.0);
	m_flagMseWeight = false;
    }

    template <typename TDevice>
    PostOutputLayer<TDevice>::~PostOutputLayer()
    {
    }

    template <typename TDevice>
    void PostOutputLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
        if (fraction.outputPatternSize() != this->size()) {
            throw std::runtime_error(std::string("Output layer size of ") + 
				     boost::lexical_cast<std::string>(this->size()) +
				     " != data target pattern size of " + 
				     boost::lexical_cast<std::string>(fraction.outputPatternSize()));
        }

        Layer<TDevice>::loadSequences(fraction);

        if (!this->_outputs().empty())
        	thrust::copy(fraction.outputs().begin(), fraction.outputs().end(), 
			     this->_outputs().begin());
    }

    /* Add 0401 wang for weighted MSE*/
    // return flag
    template <typename TDevice>
    bool PostOutputLayer<TDevice>::flagMseWeight()
    {
	return this->m_flagMseWeight;
    }

    // initialize the weight for Mse calculation
    template <typename TDevice>
    bool PostOutputLayer<TDevice>::readMseWeight(const std::string mseWeightPath)
    {
	std::ifstream ifs(mseWeightPath.c_str(), std::ifstream::binary | std::ifstream::in);
	if (!ifs.good()){
	    throw std::runtime_error(std::string("Fail to open ")+mseWeightPath);
	}
	m_flagMseWeight         = true;

	// get the number of we data
	std::streampos numEleS, numEleE;
	long int numEle;
	numEleS = ifs.tellg();
	ifs.seekg(0, std::ios::end);
	numEleE = ifs.tellg();
	numEle  = (numEleE-numEleS)/sizeof(real_t);
	ifs.seekg(0, std::ios::beg);
	
	if (numEle != this->size()){
	    printf("MSE weight vector length incompatible: %d %d", (int)numEle, (int)this->size());
	    throw std::runtime_error("Error in MSE weight configuration");
	}
	
	// read in the data
	real_t tempVal;
	std::vector<real_t> tempVec;
	for (unsigned int i = 0; i<numEle; i++){
	    ifs.read ((char *)&tempVal, sizeof(real_t));
	    tempVec.push_back(tempVal);
	}
	Cpu::real_vector tempVec2(numEle, 1.0);
	thrust::copy(tempVec.begin(), tempVec.end(), tempVec2.begin());
	m_outputMseWeights = tempVec2;
	
	std::cout << "Read #dim" << numEle << " mse vector" << std::endl;
	
	ifs.close();
	return true;
    }
    
    template <typename TDevice>
    void PostOutputLayer<TDevice>::reInitWeight()
    {
	// nothing to be done here
    }
    
    template <typename TDevice>
    Layer<TDevice>& PostOutputLayer<TDevice>::precedingLayer()
    {
        return m_precedingLayer;
    }    

    template <typename TDevice>
    bool PostOutputLayer<TDevice>::flagTrainable() const
    {
	return false;
    }

    // explicit template instantiations
    template class PostOutputLayer<Cpu>;
    template class PostOutputLayer<Gpu>;

} // namespace layers
