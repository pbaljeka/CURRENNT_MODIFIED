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

#include "../../currennt_lib/src/Configuration.hpp"
#include "../../currennt_lib/src/NeuralNetwork.hpp"
#include "../../currennt_lib/src/layers/LstmLayer.hpp"
#include "../../currennt_lib/src/layers/MDNLayer.hpp"
#include "../../currennt_lib/src/layers/BinaryClassificationLayer.hpp"
#include "../../currennt_lib/src/layers/MulticlassClassificationLayer.hpp"
#include "../../currennt_lib/src/optimizers/SteepestDescentOptimizer.hpp"
#include "../../currennt_lib/src/helpers/JsonClasses.hpp"
#include "../../currennt_lib/src/rapidjson/prettywriter.h"
#include "../../currennt_lib/src/rapidjson/filestream.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <stdarg.h>
#include <sstream>
#include <cstdlib>
#include <iomanip>

//
#define MAIN_BLOWED_THRESHOLD 5 // how many times after training errors blowed before terminating



void swap32 (uint32_t *p)
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 3 ); *( q + 3 ) = temp;
  temp = *( q + 1 ); *( q + 1 ) = *( q + 2 ); *( q + 2 ) = temp;
}

void swap16 (uint16_t *p) 
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 1 ); *( q + 1 ) = temp;
}

void swapFloat(float *p)
{
  uint8_t temp, *q;
  q = (uint8_t*) p;
  temp = *q; *q = *( q + 3 ); *( q + 3 ) = temp;
  temp = *( q + 1 ); *( q + 1 ) = *( q + 2 ); *( q + 2 ) = temp;
}

enum data_set_type
{
    DATA_SET_TRAINING,
    DATA_SET_VALIDATION,
    DATA_SET_TEST,
    DATA_SET_FEEDFORWARD
};

// helper functions (implementation below)
void readJsonFile(rapidjson::Document *doc, const std::string &filename);
boost::shared_ptr<data_sets::DataSet> loadDataSet(data_set_type dsType);
template <typename TDevice> void printLayers(const NeuralNetwork<TDevice> &nn);
template <typename TDevice> void printOptimizer(const optimizers::Optimizer<TDevice> &optimizer);
template <typename TDevice> void saveNetwork(const NeuralNetwork<TDevice> &nn, 
					     const std::string &filename, const real_t nnlr, 
					     const real_t welr);
void createModifiedTrainingSet(data_sets::DataSet *trainingSet, 
			       int parallelSequences, 
			       bool outputsToClasses, 
			       boost::mutex &swapTrainingSetsMutex);
template <typename TDevice> void saveState(const NeuralNetwork<TDevice> &nn, 
					   const optimizers::Optimizer<TDevice> &optimizer, 
					   const std::string &infoRows, const real_t nnlr, 
					   const real_t welr);
template <typename TDevice> void restoreState(NeuralNetwork<TDevice> *nn, 
					      optimizers::Optimizer<TDevice> *optimizer, 
					      std::string *infoRows);
std::string printfRow(const char *format, ...);


// main function
template <typename TDevice>
int trainerMain(const Configuration &config)
{
    try {
        // read the neural network description file 
        std::string networkFile = (config.continueFile().empty() ? 
				   config.networkFile() : config.continueFile());
        printf("Reading network from '%s'... ", networkFile.c_str());
        fflush(stdout);
	
	// Modify 0302 Wang: 
	// Add netDocPtr to select either netDocParameter or netDoc
	// netDocParameter: pointer to network parameter (trained_network.jsn or .autosave)
	// netDoc: the normal pointer to network.jsn
	rapidjson::Document *netDocPtr(0);
	rapidjson::Document netDocParameter;
        rapidjson::Document netDoc;

        readJsonFile(&netDoc, networkFile);
        printf("done.\n");
        printf("\n");
	
        // load data sets
        boost::shared_ptr<data_sets::DataSet> trainingSet    = 
	    boost::make_shared<data_sets::DataSet>();
        boost::shared_ptr<data_sets::DataSet> validationSet  = 
	    boost::make_shared<data_sets::DataSet>();
        boost::shared_ptr<data_sets::DataSet> testSet        = 
	    boost::make_shared<data_sets::DataSet>();
        boost::shared_ptr<data_sets::DataSet> feedForwardSet = 
	    boost::make_shared<data_sets::DataSet>();

        if (config.trainingMode()) {
            trainingSet = loadDataSet(DATA_SET_TRAINING);
            
            if (!config.validationFiles().empty())
                validationSet = loadDataSet(DATA_SET_VALIDATION);
            
            if (!config.testFiles().empty())
                testSet = loadDataSet(DATA_SET_TEST);
        }else if(config.printWeightPath().size()>0){
	    
        }else {
            feedForwardSet = loadDataSet(DATA_SET_FEEDFORWARD);
        }

        // calculate the maximum sequence length
        int maxSeqLength;
        if (config.trainingMode())
            maxSeqLength = std::max(trainingSet->maxSeqLength(), 
				    std::max(validationSet->maxSeqLength(), 
					     testSet->maxSeqLength()));
	else if(config.printWeightPath().size()>0)
	    maxSeqLength = 0;
        else
            maxSeqLength = feedForwardSet->maxSeqLength();

        int parallelSequences = config.parallelSequences();
	
        // modify input and output size in netDoc to match the training set size 
        // trainingSet->inputPatternSize
        // trainingSet->outputPatternSize

	
	// Add wang 0620: get the maxTxtDataLength and chaDim
	int chaDim = config.txtChaDim();
	

	int maxTxtLength;
	if (config.trainingMode()){
	    maxTxtLength = std::max(trainingSet->maxTxtLength(),
				    std::max(validationSet->maxTxtLength(),
					     testSet->maxTxtLength()));
	}else if(config.printWeightPath().size()>0){
	    maxTxtLength = 0;
	}else{
	    maxTxtLength = feedForwardSet->maxTxtLength();
	}

        // create the neural network
        printf("Creating the neural network...");
        fflush(stdout);
        int inputSize = -1;
        int outputSize = -1;
        inputSize = trainingSet->inputPatternSize();
        outputSize = trainingSet->outputPatternSize();

	/* Add 16-02-22 Wang: for WE updating */
	// NeuralNetwork<TDevice> neuralNetwork(netDoc, parallelSequences, 
	//    maxSeqLength, inputSize, outputSize);	
	if (config.weUpdate() && config.trainingMode()){
	    // change input size when external WE is used
	    // this inputSize will be checked against parameter set in neuralNetwork()
	    inputSize = inputSize-1+config.weDim();
	}

	// Re-Modify 03-02
	/* Don't need this anymore. Just directly use --network trained_network.jsn or .autosave
	if (!config.trainedParameterPath().empty()){
	    // just read in the network parameter (not any parameter else)
	    // this option can only be used through --trainedModel
	    readJsonFile(&netDocParameter, config.trainedParameterPath());
	    netDocPtr = &netDocParameter;
	}else{
	    netDocPtr = &netDoc;
	}
	*/
	
        netDocPtr = &netDoc;
	NeuralNetwork<TDevice> neuralNetwork(*netDocPtr, parallelSequences, 
					     maxSeqLength, chaDim, maxTxtLength,
					     inputSize, outputSize);

        if (!trainingSet->empty() && trainingSet->outputPatternSize() != 
	    neuralNetwork.postOutputLayer().size())
            throw std::runtime_error("Post output layer size != target size(training set)");
        if (!validationSet->empty() && validationSet->outputPatternSize() != 
	    neuralNetwork.postOutputLayer().size())
            throw std::runtime_error("Post output layer size != target size(validation set)");
        if (!testSet->empty() && testSet->outputPatternSize() != 
	    neuralNetwork.postOutputLayer().size())
            throw std::runtime_error("Post output layer size != target size(test set)");

	printf("\nNetwork construction done.\n\n");
        printf("Network summary:\n");
        printLayers(neuralNetwork);
        printf("\n");

	/* Add 16-02-22 Wang: for WE updating */
	if (config.weUpdate()){
	    // Call the method and ask neuralNetwork to load the input
	    neuralNetwork.initWeUpdate(config.weBankPath(), config.weDim(), 
				       config.weIDDim(), maxSeqLength*parallelSequences);
	    if (!neuralNetwork.initWeNoiseOpt(config.weNoiseStartDim(), config.weNoiseEndDim(),
					      config.weNoiseDev())){
		throw std::runtime_error("Error in configuration of weNoise");
	    }
	}
	
	/* Add 16-04-01 Wang: for MSE weight */
	if (config.mseWeightPath().size()>0){
	    neuralNetwork.initMseWeight(config.mseWeightPath());
	}
	
	/* Add 0413 Wang: for weight mask */
	if (config.weightMaskPath().size()>0){
	    neuralNetwork.initWeightMask(config.weightMaskPath());
	}
	
	/* Add 0514 Wang: read the data mean and variance (MV), and initialize MDN */
	// step1: read MV if provided. MV maybe not necessary
	boost::shared_ptr<data_sets::DataSetMV> dataMV=boost::make_shared<data_sets::DataSetMV>();
	if (config.datamvPath().size()>0){
	    dataMV = boost::make_shared<data_sets::DataSetMV>(config.datamvPath());
	    neuralNetwork.readMVForOutput(*dataMV);
	}
	
	// step2: initialize for MDN 
	/* As data has been normalized, no need to read MV for MDN
	  if (config.trainingMode() && config.continueFile().empty())
	  neuralNetwork.initOutputForMDN(*dataMV);
	*/
	// Note: config.continueFile().empty() make sure it is the first epoch


        // check if this is a classification task
        bool classificationTask = false;
        if (dynamic_cast<layers::BinaryClassificationLayer<TDevice>*>(
					&neuralNetwork.postOutputLayer())
            || dynamic_cast<layers::MulticlassClassificationLayer<TDevice>*>(
					&neuralNetwork.postOutputLayer())) {
                classificationTask = true;
        }
        printf("\n");

        // Training Mode: 
        if (config.trainingMode()) {
            printf("Creating the optimizer... ");
            fflush(stdout);
            boost::scoped_ptr<optimizers::Optimizer<TDevice> > optimizer;
            optimizers::SteepestDescentOptimizer<TDevice> *sdo;

            switch (config.optimizer()) {
            case Configuration::OPTIMIZER_STEEPESTDESCENT:
                sdo = new optimizers::SteepestDescentOptimizer<TDevice>(
                    neuralNetwork, *trainingSet, *validationSet, *testSet,
                    config.maxEpochs(), config.maxEpochsNoBest(), 
		    config.validateEvery(), config.testEvery(),
                    config.learningRate(), config.momentum(), config.weLearningRate(),
		    config.lrDecayRate(), config.lrDecayEpoch(),
		    config.optimizerOption()
                    );
                optimizer.reset(sdo);
                break;

            default:
                throw std::runtime_error("Unknown optimizer type");
            }

            printf("done.\n");
            printOptimizer(config, *optimizer);

            std::string infoRows;

            // continue from autosave?
            if (!config.continueFile().empty()) {
                printf("Restoring state from '%s'... ", config.continueFile().c_str());
                fflush(stdout);
                restoreState(&neuralNetwork, &*optimizer, &infoRows);
                printf("done.\n\n");
            }
	    
	    
	    // Add 05-27: Add the function to read in the weight
	    // Note: this part is only utilized in the training stage
	    if (config.continueFile().empty() && !config.trainedParameterPath().empty()){
		// Modify 05-29
		// Note: no need to re-initialize a .autosave
		//if (config.continueFile().empty()){
		//    printf("WARNING: Network parameter will be over-written by %s\n", 
		//	   config.trainedParameterPath().c_str());
		//}
		printf("Read NN parameter from %s\n", config.trainedParameterPath().c_str());
		readJsonFile(&netDocParameter, config.trainedParameterPath());
		neuralNetwork.importWeights(netDocParameter, config.trainedParameterCtr());
	    }else if (!config.continueFile().empty() && !config.trainedParameterPath().empty()){
		printf("Re-initialize .autosave using another network is unnecessary\n");
	    }else{
		// nothing
	    }


            // train the network
            printf("Starting training...\nPrint error per sequence / per timestep");
            printf("\n");
	    printf(" Epoch | Duration |           Training error  |");
	    printf("           Validation error|");
	    printf("           Test error      |");
	    printf("New best \n");
            printf("-------+----------+---------------------------+");
	    printf("---------------------------+");
	    printf("---------------------------+");
	    printf("---------\n");
	    std::cout << infoRows;
	    


	    // tranining loop
	    int  blowedTime = 0;
            bool finished   = false;
	    
            while (!finished) {
		
                const char *errFormat = (classificationTask ? 
					 "%6.2lf%%%10.3lf |" : "%14.3lf /%10.3lf |");
                const char *errSpace  = "                           |";

                // train for one epoch and measure the time
                infoRows += printfRow(" %5d | ", optimizer->currentEpoch() + 1);
                
                boost::posix_time::ptime startTime=boost::posix_time::microsec_clock::local_time();

                finished = optimizer->train();

		// Add 0511: if optimizer is blowed, decrease the learning_rate and start again
		if (optimizer->blowed()){
		    optimizer->reinit();          // reinitialize
		    optimizer->adjustLR(1);       // 
		    neuralNetwork.reInitWeight(); 
		    //neuralNetwork.initOutputForMDN(*dataMV);
		    if (++blowedTime > MAIN_BLOWED_THRESHOLD){
			printf("Learning rate tuning timeout\n");
			printf("Please change configuration and re-train\n");
			finished = true;
		    }
		    continue;
		}
		
                boost::posix_time::ptime endTime = boost::posix_time::microsec_clock::local_time();
                double duration = (double)(endTime - startTime).total_milliseconds() / 1000.0;
                infoRows += printfRow("%8.1lf |", duration);
		
		// print errors
                if (classificationTask)
                    infoRows += printfRow(errFormat, 
					  (double)optimizer->curTrainingClassError()*100.0, 
					  (double)optimizer->curTrainingError());
                else
                    infoRows += printfRow(errFormat, 
					  (double)optimizer->curTrainingError(),
					  (double)optimizer->curTrainingErrorPerFrame());
                
                if (!validationSet->empty() && 
		    optimizer->currentEpoch() % config.validateEvery() == 0) {
                    if (classificationTask)
                        infoRows += printfRow(errFormat, 
					      (double)optimizer->curValidationClassError()*100.0, 
					      (double)optimizer->curValidationError());
                    else
                        infoRows += printfRow(errFormat, 
					      (double)optimizer->curValidationError(),
					      (double)optimizer->curValidationErrorPerFrame());
                }
                else
                    infoRows += printfRow("%s", errSpace);

                if (!testSet->empty() && optimizer->currentEpoch() % config.testEvery() == 0) {
                    if (classificationTask)
                        infoRows += printfRow(errFormat, 
					      (double)optimizer->curTestClassError()*100.0, 
					      (double)optimizer->curTestError());
                    else
                        infoRows += printfRow(errFormat, 
					      (double)optimizer->curTestError(),
					      (double)optimizer->curTestErrorPerFrame());
                }
                else
                    infoRows += printfRow("%s", errSpace);
		
		// check whether to terminate training
                if (!validationSet->empty()&&optimizer->currentEpoch()%config.validateEvery()==0){
		    
                    if (optimizer->epochsSinceLowestValidationError() == 0) {
			
                        infoRows += printfRow("  yes %s\n", optimizer->optStatus().c_str());
                        if (config.autosaveBest()) {
                            std::stringstream saveFileS;
			    
                            if (config.autosavePrefix().empty()) {
                                size_t pos = config.networkFile().find_last_of('.');
                                if (pos != std::string::npos && pos > 0)
                                    saveFileS << config.networkFile().substr(0, pos);
                                else
                                    saveFileS << config.networkFile();
                            }else{
                                saveFileS << config.autosavePrefix();
			    }
			    
                            saveFileS << ".best.jsn";
                            saveNetwork(neuralNetwork,         saveFileS.str(), 
					config.learningRate(), config.weLearningRate());
                        }
                    }else{
			infoRows += printfRow("  no  %s\n", optimizer->optStatus().c_str());
		    }
                }else{
                    infoRows += printfRow("        \n");
		}
		
                // autosave
                if (config.autosave()){
                    saveState(neuralNetwork,  *optimizer, infoRows, 
			      config.learningRate(), 
			      config.weLearningRate());
		}
            }
	    

	    // Finish training
            printf("\n");

            if (optimizer->epochsSinceLowestValidationError() == config.maxEpochsNoBest())
                printf("No new lowest error since %d epochs. Training stopped.\n", 
		       config.maxEpochsNoBest());
            else
                printf("Maximum number of training epochs reached. Training stopped.\n");

            if (!validationSet->empty())
                printf("Lowest validation error: %lf\n", optimizer->lowestValidationError());
            else
                printf("Final training set error: %lf\n", optimizer->curTrainingError());
            printf("\n");

            // save the trained network to the output file
            printf("Storing the trained network in '%s'... ", config.trainedNetworkFile().c_str());
            saveNetwork(neuralNetwork, config.trainedNetworkFile(), 
			config.learningRate(), config.weLearningRate());
            printf("done.\n");

            std::cout << "Removing cache file(s) ..." << std::endl;
            if (trainingSet != boost::shared_ptr<data_sets::DataSet>())
                boost::filesystem::remove(trainingSet->cacheFileName());
            if (validationSet != boost::shared_ptr<data_sets::DataSet>())
                boost::filesystem::remove(validationSet->cacheFileName());
            if (testSet != boost::shared_ptr<data_sets::DataSet>())
                boost::filesystem::remove(testSet->cacheFileName());

	// Printing the weight into binary data
        }else if(config.printWeightPath().size()>0){
	    neuralNetwork.printWeightMatrix(config.printWeightPath());
	    
	 // Prediction mode
        }else {
            Cpu::real_vector outputMeans  = feedForwardSet->outputMeans();
            Cpu::real_vector outputStdevs = feedForwardSet->outputStdevs();
            assert (outputMeans.size()  == feedForwardSet->outputPatternSize());
            assert (outputStdevs.size() == feedForwardSet->outputPatternSize());
            
	   
            bool unstandardize = config.revertStd(); 
	    

	    /* Modify 04-08 */
	    if (unstandardize && config.outputFromWhichLayer()<0 && 
		(config.mdnPara() < -1.0 || config.mdnPara() > 0.0)){
		printf("Outputs will be scaled by mean and std  specified in NC file.\n");

		// when de-normalization is not used ?
		// 1. unstandardize
		// 2. output layer is not the last output
		// 3. MDN, output the distribution parameter
	       	// 4. MDN, output is from the sigmoid or softmax unit
		
		// If the outputMeans and outputStdevs are not in test.nc file
		// we can provide the data.mv through --datamv
		if (config.datamvPath().size()>0){
		    if (dataMV == NULL)
			throw std::runtime_error("Can't read datamv");
		    if (dataMV->outputM().size() != outputMeans.size())
			throw std::runtime_error("output dimension mismatch datamv");
		    for (int y = 0; y < outputMeans.size(); y++){
			outputMeans[y]  = dataMV->outputM()[y];
			outputStdevs[y] = dataMV->outputV()[y];
		    }
		}

		/* Add 05-31*/
		// escape the dimension corresponding to the sigmoid or softmax units
		Cpu::real_vector mdnConfigVec = neuralNetwork.getMdnConfigVec();
		if (!mdnConfigVec.empty()){
		    // if the unit is sigmoid or softmax, set the mean and std
		    for (int x = 0; x < (mdnConfigVec.size()-1)/5; x++){
			int mdnType  = (int)mdnConfigVec[5+x*5];
			if (mdnType == MDN_TYPE_SIGMOID || mdnType == MDN_TYPE_SOFTMAX){
			    int unitSOut = (int)mdnConfigVec[3+x*5];
			    int unitEOut = (int)mdnConfigVec[4+x*5];
			    for (int y = unitSOut; y < unitEOut; y++){
				outputMeans[y] = 0.0;
				outputStdevs[y] = 1.0;
			    }
			    printf("Output without de-normalization for dimension");
			    printf("from %d to %d\n", unitSOut+1, unitEOut);
			}else{
			    // nothing for GMM unit
			}
		    }
		}else{
		    // nothing for network without MDN
		}		

            }else{
		printf("Outputs will NOT be scaled by mean and std specified in NC file.\n");
	    }

	    
            int output_lag = config.outputTimeLag();
            if (config.feedForwardFormat() == Configuration::FORMAT_SINGLE_CSV) {
                // Block 20161111x01
		printf("WARNING: output only for HTK format");
            }else if (config.feedForwardFormat() == Configuration::FORMAT_CSV) {
                // Block 20161111x02
		printf("WARNING: output only for HTK format");
            }else if (config.feedForwardFormat() == Configuration::FORMAT_HTK) {
                // process all data set fractions
                int fracIdx = 0;
                boost::shared_ptr<data_sets::DataSetFraction> frac;
		if (config.outputFromGateLayer()){
		    printf("Outputs from layer %d gate output\n", config.outputFromWhichLayer());
		}else if(config.mdnPara()>0){
		    printf("Outputs from MDN with para=%f\n",config.mdnPara());
		}else{
		    printf("Outputs from layer %d \n", config.outputFromWhichLayer());
		}
		    
                while (((frac = feedForwardSet->getNextFraction()))) {
                    printf("Computing outputs for data fraction %d...", ++fracIdx);
                    fflush(stdout);
		    
                    // compute the forward pass for the current fraction and extract the outputs
		    // Note, mdnPara is the third argument.
		    //      when mdnVarScale is 1, this argument should be 1, instead of default -4
                    neuralNetwork.loadSequences(*frac);
                    neuralNetwork.computeForwardPass();
                    std::vector<std::vector<std::vector<real_t> > > outputs = 
			neuralNetwork.getOutputs(
				config.outputFromWhichLayer(), 
				config.outputFromGateLayer(),
				((config.mdnVarScaleGen().size()>0) ? 
				 ((config.mdnPara() > -1.5) ? config.mdnPara() : 1 ) : 
				 (config.mdnPara()))); 
		    // The third argument: 
		    //      if mdnVarScale is specified 
		    //          if config.mdnPara is -1, 
		    //               this is MDN parameter generation with mdnVarScale specified
		    //          else
		    //               this is sampling, scaled by mdnVarScake
		    //      else
		    //          directly use the mdnPara()

                    // write one output file per sequence
                    for (int psIdx = 0; psIdx < (int)outputs.size(); ++psIdx) {
                        if (outputs[psIdx].size() > 0) {
                            // replace_extension does not work in all Boost versions ...
                            //std::string seqTag = frac->seqInfo(psIdx).seqTag;
                            /*size_t dot_pos = seqTag.find_last_of('.');
                            if (dot_pos != std::string::npos && dot_pos > 0) {
                                seqTag = seqTag.substr(0, dot_pos);
                            }*/
                            //seqTag += ".htk";
                            //std::cout << seqTag << std::endl;
			    std::string seqTagSuf;
			    if (config.outputFromWhichLayer()<0){
				seqTagSuf = ".htk";
			    }else{
				seqTagSuf = ".bin";
			    }
                            boost::filesystem::path seqPath(frac->seqInfo(psIdx).seqTag+seqTagSuf);
                            std::string filename(seqPath.filename().string());
                            boost::filesystem::path oPath = 
				boost::filesystem::path(config.feedForwardOutputFile()) / 
				seqPath.relative_path().parent_path();
                            boost::filesystem::create_directories(oPath);
                            boost::filesystem::path filepath = oPath / filename;
                            std::ofstream file(filepath.string().c_str(), 
					       std::ofstream::out | std::ios::binary);

                            int nComps = outputs[psIdx][0].size();

                            // write header
			    if (config.outputFromWhichLayer()<0){
				unsigned tmp = (unsigned)outputs[psIdx].size();
				swap32(&tmp);
				file.write((const char*)&tmp, sizeof(unsigned));
				tmp = (unsigned)(config.featurePeriod() * 1e4);
				swap32(&tmp);
				file.write((const char*)&tmp, sizeof(unsigned));
				unsigned short tmp2 = (unsigned short)(nComps) * sizeof(float);
				swap16(&tmp2);
				file.write((const char*)&tmp2, sizeof(unsigned short));
				tmp2 = (unsigned short)(config.outputFeatureKind());
				swap16(&tmp2);
				file.write((const char*)&tmp2, sizeof(unsigned short));
			    }
			    

                            // write the patterns
                            for (int time=0; time<(int)outputs[psIdx].size(); ++time) 
			    {
                                for (int outIdx=0;outIdx<(int)outputs[psIdx][time].size();++outIdx)
				{
                                    float v;
				    v = (time < outputs[psIdx].size() - output_lag) ? 
					((float)outputs[psIdx][time+output_lag][outIdx]) :
					((float)outputs[psIdx][outputs[psIdx].size()-1][outIdx]);
                                    

                                    if (unstandardize && config.outputFromWhichLayer()<0 && 
					(config.mdnPara() < -1.0 || config.mdnPara() > 0.0)) {
                                        v *= outputStdevs[outIdx];
                                        v += outputMeans[outIdx];
                                    }
				    if (config.outputFromWhichLayer()<0){
					swapFloat(&v); 
				    }
                                    file.write((const char*)&v, sizeof(float));
                                }
                            }
                            file.close();
                        }
                    }

                    printf(" done.\n");
                }
            }
            if (feedForwardSet != boost::shared_ptr<data_sets::DataSet>()) 
                std::cout << "Removing cache file: "<<feedForwardSet->cacheFileName()<<std::endl;
            boost::filesystem::remove(feedForwardSet->cacheFileName());
        } // evaluation mode
    }
    catch (const std::exception &e) {
        printf("FAILED: %s\n", e.what());
        return 2;
    }

    return 0;
}


int main(int argc, const char *argv[])
{
    // load the configuration
    Configuration config(argc, argv);

    // run the execution device specific main function
    if (config.useCuda()) {
        int count;
        cudaError_t err;
        if (config.listDevices()) {
            if ((err = cudaGetDeviceCount(&count)) != cudaSuccess) {
                std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
                return err;
            }
            std::cout << count << " devices found" << std::endl;
            cudaDeviceProp prop;
            for (int i = 0; i < count; ++i) {
                if ((err = cudaGetDeviceProperties(&prop, i)) != cudaSuccess) {
                    std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
                    return err;
                }
                std::cout << i << ": " << prop.name << std::endl;
            }
            return 0;
        }
        int device = 0;
        char* dev = std::getenv("CURRENNT_CUDA_DEVICE");
        if (dev != NULL) {
            device = std::atoi(dev);
        }
        cudaDeviceProp prop;
        if ((err = cudaGetDeviceProperties(&prop, device)) != cudaSuccess) {
            std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        std::cout << "Using device #" << device << " (" << prop.name << ")" << std::endl;
        if ((err = cudaSetDevice(device)) != cudaSuccess) {
            std::cerr << "FAILED: " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        return trainerMain<Gpu>(config);
    }
    else
        return trainerMain<Cpu>(config);
}


void readJsonFile(rapidjson::Document *doc, const std::string &filename)
{
    // open the file
    std::ifstream ifs(filename.c_str(), std::ios::binary);
    if (!ifs.good())
        throw std::runtime_error("Cannot open file");
 
    // calculate the file size in bytes
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    // read the file into a buffer
    char *buffer = new char[size + 1];
    ifs.read(buffer, size);
    buffer[size] = '\0';

    std::string docStr(buffer);
    delete buffer;

    // extract the JSON tree
    if (doc->Parse<0>(docStr.c_str()).HasParseError())
        throw std::runtime_error(std::string("Parse error: ") + doc->GetParseError());
}


boost::shared_ptr<data_sets::DataSet> loadDataSet(data_set_type dsType)
{
    std::string type;
    std::vector<std::string> filenames;
    real_t fraction       = 1;
    bool fracShuf         = false;
    bool seqShuf          = false;
    real_t noiseDev       = 0;
    std::string cachePath = "";
    int truncSeqLength    = -1;
    
    std::string auxDataDir   = "";
    std::string auxDataExt   = "";
    int         auxDataDim   = -1;
    int         auxDataTyp   = -1;
    
    Configuration config = Configuration::instance();

    cachePath = Configuration::instance().cachePath();
    switch (dsType) {
    case DATA_SET_TRAINING:
        type           = "training set";
        filenames      = Configuration::instance().trainingFiles();
        fraction       = Configuration::instance().trainingFraction();
        fracShuf       = Configuration::instance().shuffleFractions();
        seqShuf        = Configuration::instance().shuffleSequences();
        noiseDev       = Configuration::instance().inputNoiseSigma();
        truncSeqLength = Configuration::instance().truncateSeqLength();
        break;

    case DATA_SET_VALIDATION:
        type           = "validation set";
        filenames      = Configuration::instance().validationFiles();
        fraction       = Configuration::instance().validationFraction();
        cachePath      = Configuration::instance().cachePath();
        break;

    case DATA_SET_TEST:
        type           = "test set";
        filenames      = Configuration::instance().testFiles();
        fraction       = Configuration::instance().testFraction();
        break;

    default:
        type           = "feed forward input set";
        filenames      = Configuration::instance().feedForwardInputFiles();
        noiseDev       = Configuration::instance().inputNoiseSigma();
        break;
    }

    auxDataDir         = Configuration::instance().auxillaryDataDir();
    auxDataExt         = Configuration::instance().auxillaryDataExt();
    auxDataDim         = Configuration::instance().auxillaryDataDim();
    auxDataTyp         = Configuration::instance().auxillaryDataTyp();
    
    
    printf("Loading %s ", type.c_str());
    for (std::vector<std::string>::const_iterator fn_itr = filenames.begin();
         fn_itr != filenames.end(); ++fn_itr){
        printf("'%s' ", fn_itr->c_str());
    }
    printf("...");
    fflush(stdout);

    //std::cout << "truncating to " << truncSeqLength << std::endl;
    boost::shared_ptr<data_sets::DataSet> ds 
	= boost::make_shared<data_sets::DataSet>(
		filenames,
		Configuration::instance().parallelSequences(), 
		fraction,   truncSeqLength, 
		fracShuf,   seqShuf,          noiseDev,    cachePath);

    printf("done.\n");
    printf("Loaded fraction:  %d%%\n",   (int)(fraction*100));
    printf("Sequences:        %d\n",     ds->totalSequences());
    printf("Sequence lengths: %d..%d\n", ds->minSeqLength(), ds->maxSeqLength());
    printf("Total timesteps:  %d\n",     ds->totalTimesteps());
    if (auxDataDir.size()>0){
	printf("Auxillary path:   %s\n", auxDataDir.c_str());
	printf("Auxillary ext :   %s\n", auxDataExt.c_str());
	printf("Auxillary type:   %d\n", auxDataTyp);
	printf("Auxillary dim:    %d\n", auxDataDim);
    }
    printf("\n");

    return ds;
}


template <typename TDevice>
void printLayers(const NeuralNetwork<TDevice> &nn)
{
    int weights = 0;

    for (int i = 0; i < (int)nn.layers().size(); ++i) {
        printf("(%d) %s ", i, nn.layers()[i]->type().c_str());
        printf("[size: %d", nn.layers()[i]->size());

        const layers::TrainableLayer<TDevice>* tl = 
	    dynamic_cast<const layers::TrainableLayer<TDevice>*>(nn.layers()[i].get());
        if (tl) {
            printf(", bias: %.1lf, weights: %d", (double)tl->bias(), (int)tl->weights().size());
            weights += (int)tl->weights().size();
        }else{
	    const layers::MDNLayer<TDevice>* mdnlayer = 
		dynamic_cast<const layers::MDNLayer<TDevice>*>(nn.layers()[i].get());
	    if (mdnlayer && mdnlayer -> flagTrainable()){
		printf(", weights: %d", (int)mdnlayer->weights().size());
		weights += (int)mdnlayer->weights().size();
	    }
	}

        printf("]\n");
    }

    printf("Total weights: %d\n", weights);
}


template <typename TDevice> 
void printOptimizer(const Configuration &config, const optimizers::Optimizer<TDevice> &optimizer)
{
    if (dynamic_cast<const optimizers::SteepestDescentOptimizer<TDevice>*>(&optimizer)) {
        printf("Optimizer type: Steepest descent with momentum\n");
        printf("Max training epochs:       %d\n", config.maxEpochs());
        printf("Max epochs until new best: %d\n", config.maxEpochsNoBest());
        printf("Validation error every:    %d\n", config.validateEvery());
        printf("Test error every:          %d\n", config.testEvery());
        printf("Learning rate:             %g\n", (double)config.learningRate());
        printf("Momentum:                  %g\n", (double)config.momentum());
	
	if (config.continueFile().empty() && !config.trainedParameterPath().empty()){
	    printf("Model Parameter:           %s\n", config.trainedParameterPath().c_str());
	}
	if (config.weUpdate()){
	    printf("\nParameter for WE:\n");
	    printf("WE learning_rate:          %g\n", (double)config.weLearningRate());
	    printf("WE Bank:                   %s\n", config.weBankPath().c_str());
	    printf("WE Dim:                    %d\n", config.weDim());
	    printf("WE Start index:            %d\n", config.weIDDim());
	}
	
        printf("\n");
    }
}


template <typename TDevice> 
void saveNetwork(const NeuralNetwork<TDevice> &nn, const std::string &filename,
		 const real_t nnlr, const real_t welr)
{

    if (nnlr > 0){
	rapidjson::Document jsonDoc;
	jsonDoc.SetObject();
	nn.exportLayers (&jsonDoc);
	nn.exportWeights(&jsonDoc);
	
	FILE *file = fopen(filename.c_str(), "w");
	if (!file)
	    throw std::runtime_error("Cannot open file");

	rapidjson::FileStream os(file);
	rapidjson::PrettyWriter<rapidjson::FileStream> writer(os);
	jsonDoc.Accept(writer);

	fclose(file);
    }

    if (welr > 0){
	/* Add 16-02-22 Wang: for WE updating */
	// save WE
	//autosaveFilename << ".we";
	if (nn.flagInputWeUpdate()){
	    if (!nn.saveWe(filename+".we")){
		throw std::runtime_error("Fail to save we data");
	    }
	}    
    }
}


template <typename TDevice> 
void saveState(const NeuralNetwork<TDevice> &nn, 
	       const optimizers::Optimizer<TDevice> &optimizer, 
	       const std::string &infoRows,
	       const real_t nnlr, const real_t welr)
{

    if (nnlr > 0){
	// create the JSON document
	rapidjson::Document jsonDoc;
	jsonDoc.SetObject();

	// add the configuration options
	jsonDoc.AddMember("configuration", 
			  Configuration::instance().serializedOptions().c_str(), 
			  jsonDoc.GetAllocator());

	// add the info rows
	std::string tmp = boost::replace_all_copy(infoRows, "\n", ";;;");
	jsonDoc.AddMember("info_rows", tmp.c_str(), jsonDoc.GetAllocator());

	// add the network structure and weights
	nn.exportLayers (&jsonDoc);
	nn.exportWeights(&jsonDoc);

	// add the state of the optimizer
	optimizer.exportState(&jsonDoc);
    
	// open the file
	std::stringstream autosaveFilename;
	std::string prefix = Configuration::instance().autosavePrefix(); 
	autosaveFilename << prefix;
	if (!prefix.empty())
	    autosaveFilename << '_';
	autosaveFilename << "epoch";
	autosaveFilename << std::setfill('0') << std::setw(3) << optimizer.currentEpoch();
	autosaveFilename << ".autosave";
	std::string autosaveFilename_str = autosaveFilename.str();
	FILE *file = fopen(autosaveFilename_str.c_str(), "w");
	if (!file)
	    throw std::runtime_error("Cannot open file");
	
	// write the file
	rapidjson::FileStream os(file);
	rapidjson::PrettyWriter<rapidjson::FileStream> writer(os);
	jsonDoc.Accept(writer);
	fclose(file);
    }

    if (welr > 0){
	/* Add 16-02-22 Wang: for WE updating */
	// save WE
	// open the file
	std::stringstream autosaveFilename;
	std::string prefix = Configuration::instance().autosavePrefix(); 
	autosaveFilename << prefix;
	if (!prefix.empty())
	    autosaveFilename << '_';
	autosaveFilename << "epoch";
	autosaveFilename << std::setfill('0') << std::setw(3) << optimizer.currentEpoch();
	autosaveFilename << ".autosave";
	autosaveFilename << ".we";
	if (nn.flagInputWeUpdate()){
	    if (!nn.saveWe(autosaveFilename.str())){
		throw std::runtime_error("Fail to save we data");
	    }
	}
    }
}


template <typename TDevice> 
void restoreState(NeuralNetwork<TDevice> *nn, optimizers::Optimizer<TDevice> *optimizer, 
		  std::string *infoRows)
{
    rapidjson::Document jsonDoc;
    readJsonFile(&jsonDoc, Configuration::instance().continueFile());

    // extract info rows
    if (!jsonDoc.HasMember("info_rows"))
        throw std::runtime_error("Missing value 'info_rows'");
    *infoRows = jsonDoc["info_rows"].GetString();
    boost::replace_all(*infoRows, ";;;", "\n");

    // extract the state of the optimizer
    optimizer->importState(jsonDoc);
}



std::string printfRow(const char *format, ...)
{
    // write to temporary buffer
    char buffer[100];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    // print on stdout
    std::cout << buffer;
    fflush(stdout);

    // return the same string
    return std::string(buffer);
}

