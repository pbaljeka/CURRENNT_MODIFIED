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

#ifndef DATA_SETS_DATASETFRACTION_HPP
#define DATA_SETS_DATASETFRACTION_HPP

#include "../Types.hpp"

#include <vector>
#include <string>


namespace data_sets {

    /******************************************************************************************//**
     * Contains a fraction of the data sequences in a DataSet that is small enough to be 
     * transferred completely to the GPU
     *********************************************************************************************/
    class DataSetFraction
    {
        friend class DataSet;

    public:
        struct seq_info_t {
            int         originalSeqIdx;
            int         length;
            std::string seqTag;
	    int         txtLength;
        };

    private:
        int m_inputPatternSize;
        int m_outputPatternSize;
        int m_maxSeqLength;
        int m_minSeqLength;

        std::vector<seq_info_t> m_seqInfo;

        Cpu::real_vector    m_inputs;
        Cpu::real_vector    m_outputs;
        Cpu::pattype_vector m_patTypes;
        Cpu::int_vector     m_targetClasses;
	
	// Add 0620 Wang: 
	int m_txtPatternSize;
	int m_maxTxtLength;                 // the maximum length of txt of this faction
	Cpu::int_vector   m_txtData;       // the txt input data of this fraction
	
	// Add 1024 
	int m_fracTotalLength;

	// Add 1111
	int m_auxDataDim;
	Cpu::pattype_vector m_auxPattypeData;
	Cpu::real_vector    m_auxRealData;
	Cpu::int_vector     m_auxIntData;
	
    private:
        /**
         * Creates the instance
         */
        DataSetFraction();

    public:
        /**
         * Destructor
         */
        ~DataSetFraction();

        /**
         * Returns the size of each input pattern
         *
         * @return The size of each input pattern
         */
        int inputPatternSize() const;

        /**
         * Returns the size of each output pattern
         *
         * @return The size of each output pattern
         */
        int outputPatternSize() const;

        /**
         * Returns the length of the longest sequence
         *
         * @return The length of the longest sequence
         */
        int maxSeqLength() const;

        /**
         * Returns the length of the shortest sequence
         *
         * @return The length of the shortest sequence
         */
        int minSeqLength() const;

        /**
         * Returns the number of sequences in the fraction
         *
         * @return The number of sequences in the fraction
         */
        int numSequences() const;

        /**
         * Returns information about a sequence
         *
         * @param seqIdx The index of the sequence
         */
        const seq_info_t& seqInfo(int seqIdx) const;

        /**
         * Returns the pattern types vector
         *
         * @return The pattern types vector
         */
        const Cpu::pattype_vector& patTypes() const;

        /**
         * Returns the input patterns vector
         *
         * @return The input patterns vector
         */
        const Cpu::real_vector& inputs() const;

        /**
         * Returns the output patterns vector
         *
         * @return The output patterns vector
         */
        const Cpu::real_vector& outputs() const;

	
	const Cpu::real_vector&    auxRealData() const;
	const Cpu::pattype_vector& auxPattypeData() const;
	const Cpu::int_vector&     auxIntData() const;
	const int& auxDataDim() const;

        /**
         * Returns the target classes vector
         *
         * @return The target classes vector
         */
        const Cpu::int_vector& targetClasses() const;

	// Add 0620 Wang: methods related to the txt data
	/**
         * Returns the txt data
         *
         * @return the txt data
         */
	const Cpu::int_vector& txtData() const;

	int maxTxtLength() const;

	/*
	 * Return the number of valid frames for this current fraction
	 */
	int fracTimeLength() const;
    };

} // namespace data_sets


#endif // DATA_SETS_DATASETFRACTION_HPP
