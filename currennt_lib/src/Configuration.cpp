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

#include "Configuration.hpp"
#include "rapidjson/document.h"
#include "rapidjson/filestream.h"

#include <limits>
#include <fstream>
#include <sstream>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/random/random_device.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

namespace po = boost::program_options;

#define DEFAULT_UINT_MAX std::numeric_limits<unsigned>::max(), "inf"

Configuration *Configuration::ms_instance = NULL;


namespace internal {

std::string serializeOptions(const po::variables_map &vm) 
{
    std::string s;

    for (po::variables_map::const_iterator it = vm.begin(); it != vm.end(); ++it) {
        if (it->second.value().type() == typeid(bool))
            s += it->first + '=' + boost::lexical_cast<std::string>(boost::any_cast<bool>(it->second.value()));
        else if (it->second.value().type() == typeid(unsigned))
            s += it->first + '=' + boost::lexical_cast<std::string>(boost::any_cast<unsigned>(it->second.value()));
        else if (it->second.value().type() == typeid(float))
            s += it->first + '=' + boost::lexical_cast<std::string>(boost::any_cast<float>(it->second.value()));
        else if (it->second.value().type() == typeid(double))
            s += it->first + '=' + boost::lexical_cast<std::string>(boost::any_cast<double>(it->second.value()));
        else if (it->second.value().type() == typeid(std::string))
            s += it->first + '=' + boost::any_cast<std::string>(it->second.value());
        else if (it->second.value().type() == typeid(int))
            s += it->first + '=' + boost::lexical_cast<std::string>(boost::any_cast<int>(it->second.value()));

        s += ";;;";
    }

    return s;
}

void deserializeOptions(const std::string &autosaveFile, std::stringstream *ss)
{
    // open the file
    std::ifstream ifs(autosaveFile.c_str(), std::ios::binary);
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

    // parse the JSON file
    rapidjson::Document jsonDoc;
    if (jsonDoc.Parse<0>(buffer).HasParseError())
        throw std::runtime_error(std::string("Parse error: ") + jsonDoc.GetParseError());

    // extract the options
    if (!jsonDoc.HasMember("configuration"))
        throw std::runtime_error("Missing string 'configuration'");

    std::string s = jsonDoc["configuration"].GetString();
    (*ss) << boost::replace_all_copy(s, ";;;", "\n");
}

} // namespace internal


Configuration::Configuration(int argc, const char *argv[])
{
    if (ms_instance)
        throw std::runtime_error("Static instance of class Configuration already created");
    else
        ms_instance = this;

    std::string optionsFile;
    std::string optimizerString;
    std::string weightsDistString;
    std::string feedForwardFormatString;

    std::string trainingFileList;
    std::string validationFileList;
    std::string testFileList;
    std::string feedForwardInputFileList;

    // create the command line options
    po::options_description commonOptions("Common options");
    commonOptions.add_options()
        ("help",                                                                              "shows this help message")
        ("options_file",       po::value(&optionsFile),                                       "reads the command line options from the file")
        ("network",            po::value(&m_networkFile)      ->default_value("network.jsn"), "sets the file containing the layout and weights of the neural network")
        ("cuda",               po::value(&m_useCuda)          ->default_value(true),          "use CUDA to accelerate the computations")
        ("list_devices",       po::value(&m_listDevices)      ->default_value(false),         "display list of CUDA devices and exit")
        ("parallel_sequences", po::value(&m_parallelSequences)->default_value(1),             "sets the number of parallel calculated sequences")
        ("random_seed",        po::value(&m_randomSeed)       ->default_value(0u),            "sets the seed for the random number generator (0 = auto)")
        ;

    po::options_description feedForwardOptions("Forward pass options");
    feedForwardOptions.add_options()
        ("ff_output_format", po::value(&feedForwardFormatString)->default_value("single_csv"),  "output format for output layer activations (htk, csv or single_csv)")
        ("ff_output_file", po::value(&m_feedForwardOutputFile)->default_value("ff_output.csv"), "sets the name of the output file / directory in forward pass mode (directory for htk / csv modes)")
        ("ff_output_kind", po::value(&m_outputFeatureKind)->default_value(9),                   "sets the parameter kind in case of HTK output (9: user, consult HTK book for details)")
        ("feature_period", po::value(&m_featurePeriod)->default_value(10),                      "sets the feature period in case of HTK output (in seconds)")
        ("ff_input_file",  po::value(&feedForwardInputFileList),                                "sets the name(s) of the input file(s) in forward pass mode")
        ("revert_std",     po::value(&m_revertStd)->default_value(true),                        "for regression task, de-normalize the generated data using mean and variance in data.nc")
	/* Add 16-04-08 to tap in the output of arbitary layer */
	("output_from",    po::value(&m_outputTapLayer)->default_value(-1),                     "from which layer to get the output ? (input layer is 0. Default from the output layer) ")
	("output_from_gate",po::value(&m_outputGateOut)->default_value(false),                  "if the output layer is a gate layer, get output from gate instead of transformation units? (default false)")
        ;

    po::options_description trainingOptions("Training options");
    trainingOptions.add_options()
        ("train",               po::value(&m_trainingMode)     ->default_value(false),                 "enables the training mode")
	("print_weight_to",        po::value(&m_printWeightPath)  ->default_value(""),                 "print the weight to binary file")
        ("stochastic", po::value(&m_hybridOnlineBatch)->default_value(false),                          "enables weight updates after every mini-batch of parallel calculated sequences")
        ("hybrid_online_batch", po::value(&m_hybridOnlineBatch)->default_value(false),                 "same as --stochastic (for compatibility)")
        ("shuffle_fractions",   po::value(&m_shuffleFractions) ->default_value(false),                 "shuffles mini-batches in stochastic gradient descent")
        ("shuffle_sequences",   po::value(&m_shuffleSequences) ->default_value(false),                 "shuffles sequences within and across mini-batches")
        ("max_epochs",          po::value(&m_maxEpochs)        ->default_value(DEFAULT_UINT_MAX),      "sets the maximum number of training epochs")
        ("max_epochs_no_best",  po::value(&m_maxEpochsNoBest)  ->default_value(20),                    "sets the maximum number of training epochs in which no new lowest error could be achieved")
        ("validate_every",      po::value(&m_validateEvery)    ->default_value(1),                     "sets the number of epochs until the validation error is computed")
        ("test_every",          po::value(&m_testEvery)        ->default_value(1),                     "sets the number of epochs until the test error is computed")
        ("optimizer",           po::value(&optimizerString)    ->default_value("steepest_descent"),    "sets the optimizer used for updating the weights")
        ("learning_rate",       po::value(&m_learningRate)     ->default_value((real_t)1e-5, "1e-5"),  "sets the learning rate for the steepest descent optimizer")
        ("momentum",            po::value(&m_momentum)         ->default_value((real_t)0.9,  "0.9"),   "sets the momentum for the steepest descent optimizer")
        ("weight_noise_sigma",  po::value(&m_weightNoiseSigma)  ->default_value((real_t)0),            "sets the standard deviation of the weight noise added for the gradient calculation on every batch")
        ("save_network",        po::value(&m_trainedNetwork)   ->default_value("trained_network.jsn"), "sets the file name of the trained network that will be produced")
	/* Add 16-02-22 Wang: for WE updating */
	("welearning_rate",     po::value(&m_weLearningRate) ->default_value((real_t)-1, "0"),         "sets the learning rate for we.")
	("mseWeight",           po::value(&m_mseWeightPath)  ->default_value(""),                      "path to the weight for calculating the SSE and back-propagation (binary float data)")
	("lr_decay_rate",       po::value(&m_lr_decay_rate)  ->default_value(0.1),                     "the rate to decay learning rate (0.1)")
	("lr_decay_epoch",      po::value(&m_lr_decay_epoch) ->default_value(-1),                      "ffter how many no-best epochs should the lr be decayed (-1, no use)")
	/* Add 04-13 Wang: for weight mask*/
	("weight_mask",         po::value(&m_weightMaskPath) ->default_value(""),                      "path to the network transformation matrix mask. The size of the file is identitcal to total number of parameters of the network (binary float data)")
	
	/* Add 0504 Wang: for MDN flag*/
	("mdn_config",          po::value(&m_mdnFlagPath)    ->default_value(""),                      "path to the MDN flag. ")
	("mdn_samplePara",      po::value(&m_mdnSamplingPara)->default_value((real_t)-4.0, "-4.0"),    "parameter for MDN sampling. mdn_samplePara > 0: sampling output from the distribution with the variance multiplied by mdn_samplePara. mdn_samplePara: -1.0, generate the parameter of the distribution. mdn_samplePara < -1.0: not use mdn and mdn generation.")
	("mdn_EMGenIter",       po::value(&m_EMGenIter)      ->default_value(5, "5"),                  "Number of iterations for EM generation in MDN (default 5). Iteration 1 is only initialization")
	("varInitPara",         po::value(&m_varInitPara)    ->default_value(0.5, "0.5"), "Parameter to initialize the bias of MDN mixture unit (default 0.5)")
	("vFloorPara",          po::value(&m_vFloorPara)     ->default_value(0.0001, "0.0001"), "Variance scale parameter for the variance floor (default 0.0001)")
	("wInitPara",           po::value(&m_wInitPara)     ->default_value(1.0, "1.0"), "The weight of output layer before MDN will be initialized ~u(-para/layer*size, para/layer*size)")
	("tieVariance",         po::value(&m_tiedVariance)  ->default_value(true,"true"), "Whether the variance should be tied across dimension for all mixture in MDN mixture unit? (default true) Note, this argument will be ignored if tieVarianceFlag is specified in the model (.autosave)")
	("mdn_sampleParaVec",   po::value(&m_mdnVarScaleGen)->default_value(""), "The binary vector of coefficients to scale each dimension of the variance of the mixture model. Dimension of the vector should be equal to the dimension of the target feature vector.")
	("mdnDyn",            po::value(&m_mdnDyn) ->default_value(""), "Type of MDN dynamic model. Please specify a string of digitals as num1_num2_num3..., where the number of num is equal to the number of MDN units in the output MDN layer. \n\t0: normal MDN;\n\t1: 1-order AR;\n\t2: context-dependent AR;\n\t3: 2-order AR (default 0)\n Note, this argument will be ignored if trainableFlag is specified in the model (.autosave). Also note, due to historical reason, 2 is reserved for context-dependent AR")
	("tanhAutoReg",       po::value(&m_tanhAutoregressive) ->default_value("1"), "What kind of strategy to learn AR model.\n\t0: plain\n\t1:tanh-based 1st order (real poles AR)\n\t2:tanh-based 2nd order filter (complex poles AR)")
	("ReserverZeroFilter", po::value(&m_setDynFilterZero) ->default_value(0), "Reserved option for MDN Mixture Dyn units. Don't use it if you don't know it.")
	("arrmdnLearning",     po::value(&m_arrmdnLearning)   ->default_value(0), "An option to set the learning rate for ARRMDN. Don't use it if you don't know the code")
	("arrmdnInitVar",      po::value(&m_ARRMDNInitVar)    ->default_value(0.01), "The variance of Gaussian distribution for initialization the AR parameter")
	("arrmdnUpdateInterval", po::value(&m_ARRMDNUpdateInterval)->default_value(-1), "Option for the classical form AR model learning. N+1 order AR can be estimated after estimating N order AR for this number of training epochs. (default not use) ")
	("clockRNNTimeResolution", po::value(&m_clockRNNTimeRes) ->default_value(""), "Options for ClockRNN, StartDim1_TimeResolution1_StartDim2_TimeResolution2...")
	("Optimizer",            po::value(&m_optimizerOption) ->default_value(0), "Optimization technique: \n\t0: normal gradient descent (default)\n\t1:AdaGrad (except the Trainable MDNLayer).")
	("OptimizerSecondLR",    po::value(&m_secondLearningRate) ->default_value(0.01), "Optimizer==3, it requirs additional learning rate for AdaGrad (0.01 default)")
	
        ;

    po::options_description autosaveOptions("Autosave options");
    autosaveOptions.add_options()
        ("autosave",        po::value(&m_autosave)->default_value(false), "enables autosave after every epoch")
        ("autosave_best",        po::value(&m_autosaveBest)->default_value(false), "enables autosave on best validation error")
        ("autosave_prefix", po::value(&m_autosavePrefix),                 "prefix for autosave files; e.g. 'abc/mynet-' will lead to file names like 'mynet-epoch005.autosave' in the directory 'abc'")
        ("continue",        po::value(&m_continueFile),                   "continues training from an autosave file")
        ;

    po::options_description dataFilesOptions("Data file options");
    dataFilesOptions.add_options()
        ("train_file",        po::value(&trainingFileList),                                 "sets the *.nc file(s) containing the training sequences")
        ("val_file",          po::value(&validationFileList),                               "sets the *.nc file(s) containing the validation sequences")
        ("test_file",         po::value(&testFileList),                                     "sets the *.nc file(s) containing the test sequences")
        ("train_fraction",    po::value(&m_trainingFraction)  ->default_value((real_t)1), "sets the fraction of the training set to use")
        ("val_fraction",      po::value(&m_validationFraction)->default_value((real_t)1), "sets the fraction of the validation set to use")
        ("test_fraction",     po::value(&m_testFraction)      ->default_value((real_t)1), "sets the fraction of the test set to use")
        ("truncate_seq",      po::value(&m_truncSeqLength)    ->default_value(0),         "enables training sequence truncation to given maximum length (0 to disable)")
        ("input_noise_sigma", po::value(&m_inputNoiseSigma)   ->default_value((real_t)0), "sets the standard deviation of the input noise for training sets")
        ("input_left_context", po::value(&m_inputLeftContext) ->default_value(0), "sets the number of left context frames (first frame is duplicated as necessary)")
        ("input_right_context", po::value(&m_inputRightContext)->default_value(0), "sets the number of right context frames (last frame is duplicated as necessary)")
        ("output_time_lag",   po::value(&m_outputTimeLag)->default_value(0),              "sets the time lag in the training targets (0 = predict current frame, 1 = predict previous frame, etc.)")
        ("cache_path",        po::value(&m_cachePath)         ->default_value(""),        "sets the cache path where the .nc data is cached for random access")
	/* Add 16-02-22 Wang: for WE updating */
	("weExternal",          po::value(&m_weUpdate) ->default_value(false),    "whether update the input word embedding vectors (false)")
	("weIDDim",           po::value(&m_weIDDim)    ->default_value(-1),       "the WE index is the ?-th dimension of the input vector? (-1)")
	("weDim",             po::value(&m_weDim)      ->default_value(0),        "the dimension of the word embedding vectors (0)")
	("weBank",            po::value(&m_weBank)     ->default_value(""),       "the path to the word vectors")
	("trainedModel",      po::value(&m_trainedParameter)    ->default_value(""), "the path to the trained model paratemeter")
	("trainedModelCtr",   po::value(&m_trainedParameterCtr) ->default_value(""), "the trainedModel controller. A string of 0/1/2/3 whose length is #layer of NN. \n 0: not read this layer\n 1: read this layer if number of weights matches\n 2: assume column number is the same\n 3: assume row numbe is the same \n (default: void, read in all parameters in option 1)")
	("datamv",            po::value(&m_datamvPath)          ->default_value(""), "the path to the data mv file. This file can be read in and initialize MDN parameter (now not in use)")
	("txtChaDim",         po::value(&m_chaDimLstmCharW)     ->default_value(0), "the dimension of the bag of character for LstmCharW")
	("txtBank",           po::value(&m_chaBankPath)         ->default_value(""),       "the path to the character vectors for LstmCharW")
	("weNoiseStartDim",   po::value(&m_weNoiseStartDim)     ->default_value(-2), "the first dimension that will be added with noise in the input layer (for Word embedding). Python-style index")
	("weNoiseEndDim", po::value(&m_weNoiseEndDim)           ->default_value(-1), "the next of the last dimension that will be addded with noise in the input layer (for Word embedding). Python-style index")
	("weNoiseDev",    po::value(&m_weNoiseDev)              ->default_value(0.1), "the standard deviation of the noise that will be added to the word vectors (default 0.1)")
	("targetDataType", po::value(&m_KLDOutputDataType)      ->default_value(-1),   "the type of the target data.\n\t1: linear domain, zero-mean, uni-var\n2: log domain, zero-mean, uni-var\n")
	("KLDLRfactor",    po::value(&m_lrFactor)               ->default_value(1),   "the factor to scale the training criterion and gradient for KLD. default 1.0")
        ;

    po::options_description weightsInitializationOptions("Weight initialization options");
    weightsInitializationOptions.add_options()
        ("weights_dist",         po::value(&weightsDistString)   ->default_value("uniform"),            "sets the distribution type of the initial weights (uniform or normal)")
        ("weights_uniform_min",  po::value(&m_weightsUniformMin) ->default_value((real_t)-0.1, "-0.1"), "sets the minimum value of the uniform distribution")
        ("weights_uniform_max",  po::value(&m_weightsUniformMax) ->default_value((real_t)+0.1, "0.1"),  "sets the maximum value of the uniform distribution")
        ("weights_normal_sigma", po::value(&m_weightsNormalSigma)->default_value((real_t)0.1, "0.1"),   "sets the standard deviation of the normal distribution")
        ("weights_normal_mean",  po::value(&m_weightsNormalMean) ->default_value((real_t)0.0, "0"),     "sets the mean of the normal distribution")
	/* Add 16-04-02 Wang: for initiliaizing the bias for gate of Highway network */
	("highway_gate_bias",    po::value(&m_highwayBias) -> default_value((real_t)-1.50, "-1.50"),    "sets the bias for the sigmoid function in the gate of highway block")
        ;

    po::positional_options_description positionalOptions;
    positionalOptions.add("options_file", 1);

    // parse the command line
    po::options_description visibleOptions;
    visibleOptions.add(commonOptions);
    visibleOptions.add(feedForwardOptions);
    visibleOptions.add(trainingOptions);
    visibleOptions.add(autosaveOptions);
    visibleOptions.add(dataFilesOptions);
    visibleOptions.add(weightsInitializationOptions);

    po::options_description allOptions;
    allOptions.add(visibleOptions);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(allOptions).positional(positionalOptions).run(), vm);
        if (vm.count("options_file")) {
            optionsFile = vm["options_file"].as<std::string>();
            std::ifstream file(optionsFile.c_str(), std::ifstream::in);
            if (!file.is_open())
                throw std::runtime_error(std::string("Could not open options file '") + optionsFile + "'");
            po::store(po::parse_config_file(file, allOptions), vm);
        }
        po::notify(vm);
    }
    catch (const std::exception &e) {
        if (!vm.count("help"))
            std::cout << "Error while parsing the command line and/or options file: " << e.what() << std::endl;

        std::cout << "Usage: currennt [options] [options-file]" << std::endl;
        std::cout << visibleOptions;

        exit(vm.count("help") ? 0 : 1);
    }

    if (vm.count("help")) {
        std::cout << "Usage: currennt [options] [options-file]" << std::endl;
        std::cout << visibleOptions;

        exit(0);
    }

    // load options from autosave
    if (!m_continueFile.empty()) {
        try {
            std::stringstream ss;
            internal::deserializeOptions(m_continueFile, &ss);
            vm = po::variables_map();
            po::store(po::parse_config_file(ss, allOptions), vm);
            po::notify(vm);
        }
        catch (const std::exception &e) {
            std::cout << "Error while restoring configuration from autosave file: " << e.what() << std::endl;

            exit(1);
        }
    }

    // store the options for autosave
    m_serializedOptions = internal::serializeOptions(vm);

    // split the training file options
    boost::algorithm::split(m_trainingFiles, trainingFileList, boost::algorithm::is_any_of(";,"), boost::algorithm::token_compress_on);
    if (!validationFileList.empty())
        boost::algorithm::split(m_validationFiles, validationFileList, boost::algorithm::is_any_of(";,"), boost::algorithm::token_compress_on);
    if (!testFileList.empty())
        boost::algorithm::split(m_testFiles, testFileList, boost::algorithm::is_any_of(";,"), boost::algorithm::token_compress_on);
    if (!feedForwardInputFileList.empty())
        boost::algorithm::split(m_feedForwardInputFiles, feedForwardInputFileList, boost::algorithm::is_any_of(";,"), boost::algorithm::token_compress_on);

    // check the optimizer string
    if (optimizerString == "rprop")
        m_optimizer = OPTIMIZER_RPROP;
    else if (optimizerString == "steepest_descent")
        m_optimizer = OPTIMIZER_STEEPESTDESCENT;
    else {
        std::cout << "ERROR: Invalid optimizer. Possible values: steepest_descent, rprop." << std::endl;
        exit(1);
    }

    // create a random seed
    if (!m_randomSeed)
        m_randomSeed = boost::random::random_device()();

    // check the weights distribution string
    if (weightsDistString == "normal")
        m_weightsDistribution = DISTRIBUTION_NORMAL;
    else if (weightsDistString == "uniform")
        m_weightsDistribution = DISTRIBUTION_UNIFORM;
    else if (weightsDistString == "uninorm")
	m_weightsDistribution = DISTRIBUTION_UNINORMALIZED;
    else {
        std::cout << "ERROR: Invalid initial weights distribution type. Possible values: normal, uniform." << std::endl;
        exit(1);
    }

    // check the feedforward format string
    if (feedForwardFormatString == "single_csv")
        m_feedForwardFormat = FORMAT_SINGLE_CSV;
    else if (feedForwardFormatString == "csv")
        m_feedForwardFormat = FORMAT_CSV;
    else if (feedForwardFormatString == "htk")
        m_feedForwardFormat = FORMAT_HTK;
    else {
        std::cout << "ERROR: Invalid feedforward format string. Possible values: single_csv, csv, htk." << std::endl;
        exit(1);
    }

    // check data sets fractions
    if (m_trainingFraction <= 0 || 1 < m_trainingFraction) {
        std::cout << "ERROR: Invalid training set fraction. Should be 0 < x <= 1" << std::endl;
        exit(1);
    }
    if (m_validationFraction <= 0 || 1 < m_validationFraction) {
        std::cout << "ERROR: Invalid validation set fraction. Should be 0 < x <= 1" << std::endl;
        exit(1);
    }
    if (m_testFraction <= 0 || 1 < m_testFraction) {
        std::cout << "ERROR: Invalid test set fraction. Should be 0 < x <= 1" << std::endl;
        exit(1);
    }

    // print information about active command line options
    std::cout << "Configuration Infor:" << std::endl;
    if (m_trainingMode) {
        std::cout << "\tTraining Mode: Started in ";
	std::cout << (m_hybridOnlineBatch ? "hybrid online/batch" : "batch") << std::endl;

        if (m_shuffleFractions){
            std::cout << "\t\tMini-batches (parallel " << m_parallelSequences << " sequences each)";
	    std::cout << " will be shuffled during training." << std::endl;
	}
        if (m_shuffleSequences){
            std::cout << "\t\tSequences shuffled within and across mini-batches.\n" << std::endl;
	}
        if (m_inputNoiseSigma != (real_t)0){
            std::cout << "\t\tUsing input noise with std. of " << m_inputNoiseSigma << std::endl;
	}
        std::cout << "\t\tWritting network  to '" << m_trainedNetwork << "'." << std::endl;
        if (boost::filesystem::exists(m_trainedNetwork))
            std::cout << "\t\tWARNING: overwriting '" << m_trainedNetwork << "'" << std::endl;
	
    }else if(m_printWeightPath.size()>0){
	std::cout << "\tStarted in printing mode. ";
	std::cout << "Weight will be print to " << m_printWeightPath << std::endl;
	
    }else {
        std::cout << "\tStarted in forward pass mode." << std::endl;
        std::cout << "\tWritting output to '" << m_feedForwardOutputFile << "'." << std::endl;
        if (boost::filesystem::exists(m_feedForwardOutputFile))
            std::cout << "\t\tWARNING: overwriting '" << m_feedForwardOutputFile << std::endl;
    }

    if (m_trainingMode && !m_validationFiles.empty())
        std::cout << "\tValidation every " << m_validateEvery << " epochs." << std::endl;
    if (m_trainingMode && !m_testFiles.empty())
        std::cout << "\tTest  every " << m_testEvery << " epochs." << std::endl;

    if (m_trainingMode) {
        std::cout << "\n\tTraining will be stopped";
        if (m_maxEpochs != std::numeric_limits<unsigned>::max())
            std::cout << " after " << m_maxEpochs << " epochs or";
        std::cout << " after no new lowest validation error for ";
	std::cout << m_maxEpochsNoBest << " epochs." << std::endl;
    }
    
    if (m_autosave) {
        std::cout << "\tAutosave after EVERY EPOCH enabled." << std::endl;
    }
    if (m_autosaveBest) {
        std::cout << "\tAutosave on BEST VALIDATION ERROR enabled." << std::endl;
    }

    if (m_useCuda){
        std::cout << "\tUtilizing the GPU on ";
	std::cout << m_parallelSequences << " sequences in parallel." << std::endl;
    }else
        std::cout << "\tWARNING: CUDA option not set. Computations on the CPU!" << std::endl;

    if (m_trainingMode) {
	std::cout << "\n\tInitialization method:" << std::endl;
        if (m_weightsDistribution == DISTRIBUTION_NORMAL){
            std::cout << "\t\tNormal dist. with mean, std:"; 
	    std::cout << m_weightsNormalMean << m_weightsNormalSigma;
        }else if (m_weightsDistribution == DISTRIBUTION_UNINORMALIZED)
	    std::cout << "\t\tUniform dist. with layer-wise range" << std::endl;
	else{
            std::cout << "\t\tUniform dist. with range [";
	    std::cout << m_weightsUniformMin << ", " << m_weightsUniformMax << "]";
	}
	std::cout << "\n\t\tRandom seed: " << m_randomSeed << std::endl;
	
    }
    
    /* Add 16-02-22 Wang: for WE updating */
    if (m_weUpdate){
	// for checking:
	if (m_inputNoiseSigma > 0.0){
	    std::cout << "\tWARNING: input vectors are used, input noise is turned off" << std::endl;
	    m_inputNoiseSigma = 0.0;
	}
	if (m_weIDDim < 0 || m_weDim < 1 || m_weBank.size()<1){
	    std::cout << "\tERROR: Invalid configuration for WE updating" << std::endl;
	    exit(1);
	}
    }

    if (m_mseWeightPath.size()>0){
	std::cout << "\tUsing MSE Weight: " << m_mseWeightPath  << std::endl;
    }

    
    
    std::cout << std::endl;
}

Configuration::~Configuration()
{
}

const Configuration& Configuration::instance()
{
    return *ms_instance;
}

const std::string& Configuration::serializedOptions() const
{
    return m_serializedOptions;
}

bool Configuration::trainingMode() const
{
    return m_trainingMode;
}

bool Configuration::hybridOnlineBatch() const
{
    return m_hybridOnlineBatch;
}

bool Configuration::shuffleFractions() const
{
    return m_shuffleFractions;
}

bool Configuration::shuffleSequences() const
{
    return m_shuffleSequences;
}

bool Configuration::useCuda() const
{
    return m_useCuda;
}

bool Configuration::listDevices() const
{
    return m_listDevices;
}

bool Configuration::autosave() const
{
    return m_autosave;
}

bool Configuration::autosaveBest() const
{
    return m_autosaveBest;
}

Configuration::optimizer_type_t Configuration::optimizer() const
{
    return m_optimizer;
}

int Configuration::parallelSequences() const
{
    return (int)m_parallelSequences;
}

int Configuration::maxEpochs() const
{
    return (int)m_maxEpochs;
}

int Configuration::maxEpochsNoBest() const
{
    return (int)m_maxEpochsNoBest;
}

int Configuration::validateEvery() const
{
    return (int)m_validateEvery;
}

int Configuration::testEvery() const
{
    return (int)m_testEvery;
}

real_t Configuration::learningRate() const
{
    return m_learningRate;
}

real_t Configuration::momentum() const
{
    return m_momentum;
}

const std::string& Configuration::networkFile() const
{
    return m_networkFile;
}

const std::vector<std::string>& Configuration::trainingFiles() const
{
    return m_trainingFiles;
}

const std::string& Configuration::cachePath() const
{
    return m_cachePath;
}


const std::vector<std::string>& Configuration::validationFiles() const
{
    return m_validationFiles;
}

const std::vector<std::string>& Configuration::testFiles() const
{
    return m_testFiles;
}

unsigned Configuration::randomSeed() const
{
    return m_randomSeed;
}

Configuration::distribution_type_t Configuration::weightsDistributionType() const
{
    return m_weightsDistribution;
}

real_t Configuration::weightsDistributionUniformMin() const
{
    return m_weightsUniformMin;
}

real_t Configuration::weightsDistributionUniformMax() const
{
    return m_weightsUniformMax;
}

real_t Configuration::weightsDistributionNormalSigma() const
{
    return m_weightsNormalSigma;
}

real_t Configuration::weightsDistributionNormalMean() const
{
    return m_weightsNormalMean;
}

real_t Configuration::inputNoiseSigma() const
{
    return m_inputNoiseSigma;
}

int Configuration::inputLeftContext() const
{
    return m_inputLeftContext;
}

int Configuration::inputRightContext() const
{
    return m_inputRightContext;
}

int Configuration::outputTimeLag() const
{   
    return m_outputTimeLag;
}

real_t Configuration::weightNoiseSigma() const
{
    return m_weightNoiseSigma;
}

real_t Configuration::trainingFraction() const
{
    return m_trainingFraction;
}

real_t Configuration::validationFraction() const
{
    return m_validationFraction;
}

real_t Configuration::testFraction() const
{
    return m_testFraction;
}

const std::string& Configuration::trainedNetworkFile() const
{
    return m_trainedNetwork;
}

Configuration::feedforwardformat_type_t Configuration::feedForwardFormat() const
{
    return m_feedForwardFormat;
}

real_t Configuration::featurePeriod() const
{
    return m_featurePeriod;
}

unsigned Configuration::outputFeatureKind() const
{
    return m_outputFeatureKind;
}

unsigned Configuration::truncateSeqLength() const
{
    return m_truncSeqLength;
}

const std::vector<std::string>& Configuration::feedForwardInputFiles() const
{
    return m_feedForwardInputFiles;

}

const std::string& Configuration::feedForwardOutputFile() const
{
    return m_feedForwardOutputFile;
}

const std::string& Configuration::autosavePrefix() const
{
    return m_autosavePrefix;
}

const std::string& Configuration::continueFile() const
{
    return m_continueFile;
}

/* Add 16-02-22 Wang: for WE updating */
const std::string& Configuration::weBankPath() const
{
    return m_weBank;
}

const std::string& Configuration::chaBankPath() const
{
    return m_chaBankPath;
}


const std::string& Configuration::mseWeightPath() const
{
    return m_mseWeightPath;
}

const std::string& Configuration::weightMaskPath() const
{
    return m_weightMaskPath;
}

const std::string& Configuration::trainedParameterPath() const
{
    return m_trainedParameter;
}

const std::string& Configuration::trainedParameterCtr() const
{
    return m_trainedParameterCtr;
}
    
const unsigned& Configuration::weIDDim() const
{
    return m_weIDDim;
}
const unsigned& Configuration::weDim() const
{
    return m_weDim;
}

const unsigned& Configuration::txtChaDim() const
{
    return m_chaDimLstmCharW;
}

bool Configuration::weUpdate() const
{
    return m_weUpdate;
}


real_t Configuration::weLearningRate() const
{
    return m_weLearningRate;
}

bool Configuration::revertStd() const
{
    return m_revertStd;
}

const real_t& Configuration::highwayGateBias() const
{
    return m_highwayBias;
}

const int& Configuration::outputFromWhichLayer() const
{
    return m_outputTapLayer;
}


const bool& Configuration::outputFromGateLayer() const
{
    return m_outputGateOut;
}

const int& Configuration::lrDecayEpoch() const
{
    return m_lr_decay_epoch;
}
const real_t& Configuration::lrDecayRate() const
{
    return m_lr_decay_rate;
}

const std::string& Configuration::mdnFlagPath() const
{
    return m_mdnFlagPath;
}

bool Configuration::mdnFlag() const
{
    return m_mdnFlagPath.length()>0;
}

const std::string& Configuration::mdnDyn() const
{
    return m_mdnDyn;
}


const std::string& Configuration::datamvPath() const
{
    return m_datamvPath;
}

const real_t& Configuration::mdnPara() const
{
    return m_mdnSamplingPara;
}

const int& Configuration::EMIterNM() const
{
    return m_EMGenIter;
}

const real_t& Configuration::getVarInitPara() const
{
    return m_varInitPara;
}

const real_t& Configuration::getVFloorPara() const
{
    return m_vFloorPara;
}

const real_t& Configuration::getWInitPara() const
{
    return m_wInitPara;
}

const bool& Configuration::getTiedVariance() const
{
    return m_tiedVariance;
}

const std::string& Configuration::printWeightPath() const
{
    return m_printWeightPath;
}

const std::string& Configuration::mdnVarScaleGen() const
{
    return m_mdnVarScaleGen;
}

const std::string& Configuration::tanhAutoregressive() const
{
    return m_tanhAutoregressive;
}

const int& Configuration::zeroFilter() const
{
    return m_setDynFilterZero;
}

const int& Configuration::arrmdnLearning() const
{
    return m_arrmdnLearning;
}

const int& Configuration::weNoiseStartDim() const
{
    return m_weNoiseStartDim;
}

const int& Configuration::weNoiseEndDim() const
{
    return m_weNoiseEndDim;
}

const real_t& Configuration::weNoiseDev() const
{
    return m_weNoiseDev;
}

const real_t& Configuration::arRMDNInitVar() const
{
    return m_ARRMDNInitVar;
}

const int& Configuration::arRMDNUpdateInterval() const
{
    return m_ARRMDNUpdateInterval;
}

const std::string& Configuration::clockRNNTimeRes() const
{
    return m_clockRNNTimeRes;
}

const int& Configuration::KLDOutputDataType() const
{
    return m_KLDOutputDataType;
}

const real_t& Configuration::lrFactor() const
{
    return m_lrFactor;
}


const unsigned& Configuration::optimizerOption() const
{
    return m_optimizerOption;
}


const real_t& Configuration::optimizerSecondLR() const
{
    return m_secondLearningRate;
}
