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

#ifndef LAYERS_LAYER_HPP
#define LAYERS_LAYER_HPP

#include "../Types.hpp"
#include "../data_sets/DataSetFraction.hpp"
#include "../helpers/JsonClassesForward.hpp"

#include <string>


namespace layers {

    /******************************************************************************************//**
     * Represents a layer in the neural network
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class Layer
    {
        typedef typename TDevice::real_vector    real_vector;
        typedef typename TDevice::pattype_vector pattype_vector;
	typedef typename Cpu::real_vector cpu_real_vector;
    private:
        const std::string m_name;
        const int         m_size;
        const int         m_parallelSequences;
        const int         m_maxSeqLength;

        int               m_curMaxSeqLength;
        int               m_curMinSeqLength;
        int               m_curNumSeqs;
        real_vector       m_outputs;
        real_vector       m_outputErrors;
        pattype_vector    m_patTypes;
	
	/* Add 16-02-22 Wang: for WE updating */
	bool              m_InputWeUpdate;     // the whether layer is the input layer with WE 
	                                       // to be updated ?
	                                       // We can not define it in Trainablelayer 
	                                       // because input layer only inherits from layer
	cpu_real_vector   m_outputErrorsCopy;  // make a CPU copy

	/* Add 16-09-28 Wang: the current training epoch */
	int               m_currTrainingEpoch; // maybe useful

    protected:
        real_vector& _outputs();
	
	/* Add 16-02-22 Wang: for WE updating */
	bool         _setInputWeUpdate(const bool& flag);
	
    public:

	/* Add 16-02-22 Wang: for WE updating */
	bool&        inputWeUpdate();
	
        /**
         * Constructs the Layer
         *
         * @param layerChild        The layer child of the JSON configuration for this layer
         * @param parallelSequences The maximum number of sequences  computed in parallel
         * @param maxSeqLength      The maximum length of a sequence
         * @param createOutputs     If false, then the outputs vector will be left empty
         */
        Layer(const helpers::JsonValue &layerChild, 
	      int parallelSequences, int maxSeqLength, bool createOutputs = true);
	
	// overload for CNN
	Layer(const helpers::JsonValue &layerChild, int parallelSequences, int maxSeqLength, 
	      int outputSize, bool createOutputs = true);

        /**
         * Destructs the Layer
         */
        virtual ~Layer();

        /**
         * Returns the name of the layer
         *
         * @return The name of the layer
         */
        const std::string& name() const;

        /**
         * Returns the number of blocks in the layer
         * 
         * @return The number of blocks in the layer
         */
	// modify 0612: to virtual size so that LstmLayerChaW can be supported
        virtual int size() const;

        /**
         * Returns the maximum number of sequences that can be computed in parallel
         *
         * @return The maximum number of sequences that can be computed in parallel
         */
        int parallelSequences() const;

        /**
         * Returns the maximum length of a sequence
         *
         * @return The maximum length of a sequence
         */
        int maxSeqLength() const;

        /**
         * Returns the maximum length of the currently loaded sequences
         *
         * @return The maximum length of the currently loaded sequences
         */
        int curMaxSeqLength() const;

        /**
         * Returns the minimum length of the currently loaded sequences
         *
         * @return The minimum length of the currently loaded sequences
         */
        int curMinSeqLength() const;

        /**
         * Returns the number sequences in the current data set fraction
         *
         * @return The number sequences in the current data set fraction
         */
        int curNumSeqs() const;

        /**
         * Calculates the output errors of the layer
         *
         * @return The output error
         */
        virtual real_vector& outputErrors();

	
	cpu_real_vector& outputErrorsCpu();   

        /**
         * Returns the pattern types vector
         * 
         * @return The pattern types vector
         */
        const pattype_vector& patTypes() const;

        /**
         * Returns a string describing the layer type
         *
         * @return A string describing the layer type
         */
        virtual const std::string& type() const =0;

        /**
         * Returns the outputs of the layer
         *
         * @return The outputs
         */
        virtual real_vector& outputs();

        /**
         * Loads sequences from a data set
         *
         * @param fraction The fraction of the data set to load
         */
        virtual void loadSequences(const data_sets::DataSetFraction &fraction);

        /**
         * Computes the forward pass
         */
        virtual void computeForwardPass() =0;

        /**
         * Computes the backward pass, including the weight updates
         */
        virtual void computeBackwardPass() =0;
	
        /**
         * Stores the description of the layer in a JSON object
         *
         * @param layersArray The array of layers in the document
         * @param allocator   The allocator to use
         */
        virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;

	/**
	 * Re-initialize the network
	   only defines for Trainable Layers
	 */
	virtual void reInitWeight() = 0;
	
	
	/*
	  
	 */
	virtual void linkTargetLayer(Layer<TDevice> &targetLayer);
	    
	/**
	 * Set and read the m_currTrainingEpoch
	 */
	virtual void setCurrTrainingEpoch(const int curTrainingEpoch);
	
	virtual int& getCurrTrainingEpoch();
	

	/*
	 *  Provide additional information
	 */
	virtual const std::string& layerAddInfor(const int opt) const;
	
	/*
	 * Add 1221, 
	 * prepareStepGeneration: prepare the layer for generating the timeStep-th frame
	 * computeForwardPass: compute the output for the timeStep-th frame
	 */
	virtual void prepareStepGeneration(const int timeStep);
	
	virtual void computeForwardPass(const int timeStep)=0;

	virtual real_vector& secondOutputs();

    };

} // namespace layers


#endif // LAYERS_LAYER_HPP
