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

#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "layers/InputLayer.hpp"
#include "layers/TrainableLayer.hpp"
#include "layers/PostOutputLayer.hpp"
#include "layers/MDNLayer.hpp"
#include "layers/SkipParaLayer.hpp"
#include "layers/LstmLayerCharW.hpp"
#include "data_sets/DataSet.hpp"
#include "helpers/JsonClassesForward.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <boost/shared_ptr.hpp>

#include <vector>
#include <memory>


/*****************************************************************************************************************//**
 * Represents the neural network
 *
 * @param TDevice The computation device
 *********************************************************************************************************************/
template <typename TDevice>
class NeuralNetwork
{
private:
    std::vector<boost::shared_ptr<layers::Layer<TDevice> > > m_layers;
    
    /* Add 02-24 Wang Residual Network*/
    std::vector<layers::Layer<TDevice>*> m_skipAddLayers;

public:
    /**
     * Creates the neural network from the process configuration
     *
     * @param jsonDoc           The JSON document containing the network configuration
     * @param parallelSequences The maximum number of sequences that shall be computed in parallel
     * @param maxSeqLength      The maximum length of a sequence
     */
    NeuralNetwork(const helpers::JsonDocument &jsonDoc, int parallelSequences, 
		  int maxSeqLength, int chaDim, int maxTxtLength,
                  int inputSizeOverride, int outputSizeOverride);

    /**
     * Destructs the neural network
     */
    ~NeuralNetwork();

    /**
     * Returns the layers
     *
     * @return The layers
     */
    const std::vector<boost::shared_ptr<layers::Layer<TDevice> > >& layers() const;

    /**
     * Returns the input layer
     *
     * @return The input layer
     */
    layers::InputLayer<TDevice>& inputLayer();

    /**
     * Returns the output layer
     *
     * @return The output layer
     */
    /* Modify 04-08 Wang: to tap in the output of arbitary layer */
    //layers::TrainableLayer<TDevice>& outputLayer();
    layers::Layer<TDevice>& outputLayer(const int layerID=-1);

    layers::SkipLayer<TDevice>* outGateLayer(const int layerID);
    
    layers::MDNLayer<TDevice>* outMDNLayer();

    /**
     * Returns the post output layer
     *
     * @return The post output layer
     */
    layers::PostOutputLayer<TDevice>& postOutputLayer();

    /**
     * Loads sequences to the device
     *
     * @param fraction The data set fraction containing the sequences
     */
    void loadSequences(const data_sets::DataSetFraction &fraction);

    /**
     * Computes the forward pass
     */
    void computeForwardPass();

    /**
     * Computes the backward pass, including the weight updates
     *
     * The forward pass must be computed first!
     */
    void computeBackwardPass();

    /**
     * Calculates the error at the output layer
     *
     * The forward pass must be computed first!
     *
     * @return The computed error
     */
    real_t calculateError() const;

    /**
     * Stores the description of the layers in a JSON tree
     *
     * @param jsonDoc The JSON document
     */
    void exportLayers(const helpers::JsonDocument& jsonDoc) const;

    /**
     * Stores the weights in a JSON tree
     *
     * @param jsonDoc The JSON document
     */
    void exportWeights(const helpers::JsonDocument& jsonDoc) const;

    /**
     * Returns the outputs of the processed fraction
     *
     * ...[1][2][3] contains the activation of the 4th output neuron at the 3nd timestep
     * of the 2nd parallel sequence.
     *
     * @return Outputs of the processed fraction
     */
    std::vector<std::vector<std::vector<real_t> > > getOutputs(const int  layerID        = -1, 
							       const bool gateFromOutput = false,
							       const real_t  mdnoutput   = -4.0);
    
    /**
     * Read in the weight from trained_network.jsn or .autosave
     * 
     */
    void importWeights(const helpers::JsonDocument &jsonDoc, const std::string &ctrStr);
    
    /* Add 16-02-22 Wang: for WE updating */
    // repare for we updateing
    bool initWeUpdate(const std::string weBankPath, const unsigned weDim, 
		      const unsigned weIDDim, const unsigned maxLength);
    
    bool flagInputWeUpdate() const;


    bool saveWe(const std::string weFile) const;
    
    /* Add 04-01 Wang: for RMSE output mask */
    bool initMseWeight(const std::string mseWeightPath);

    /* Add 0413 Wang: for weight mask */
    bool initWeightMask(const std::string weightMaskPath);

    void maskWeight();
    
    /* Add 0511 Wang: re-initialize the weight*/
    void reInitWeight();


    /* Add 0514 Wang: initialize the output layer for MDN */
    void initOutputForMDN(const data_sets::DataSetMV &datamv);
    
    /* Add 0531 Wang: get the mdn config*/
    Cpu::real_vector getMdnConfigVec();

    /* Add 0630 Wang: print the binary weight matrix */
    void printWeightMatrix(const std::string weightPath);
    
};


#endif // NEURALNETWORK_HPP
