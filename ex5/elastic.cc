
#include <iostream>
#include <fstream> // For streams
#include <sys/stat.h> // For mkdir
#include <cfloat> // For FLT_MAX
#include <cmath>

#define BZ_DEBUG	// Add debugging information to blitz library
#include <blitz/array.h>
#include <blitz/tinyvec2.h>

#include <gsl/gsl_matrix.h> // For gsl_matrix
#include <gsl/gsl_linalg.h> // For lincomp

#include "./BlitzHDF5Helper.hh"
#include "./FlowToImage.hh"
#include "./Fast_PD/Fast_PD.h"



/* Compile:
 * g++ -Wall -O3 -L/usr/local/HDF_Group/HDF5/1.8.15/lib/ -I/usr/local/HDF_Group/HDF5/1.8.15/include/ -g elastic.cc Fast_PD/graph.cpp Fast_PD/LinkedBlockList.cpp Fast_PD/maxflow.cpp FlowToImage.hh -lblitz -o elastic -lhdf5 -ldl
 */


void printUsage()
{
	printf("elastic <filename1> <filename2> <mode> [lambda]\n");
	printf("\tfilename1: \tThe filename of the first h5 file\n");
	printf("\tfilename2: \tThe filename of the second h5 file\n");
	printf("\tmode:\t\tThe mode which should be used, either 0 for SSD or 1 for NCC (not implemented yet!).\n");
	printf("\tmode:\t\tThe lambda that should be used, default value is 0.1\n");

	// Leave the application
	exit(1);
}


/**
 * @brief
 * This method returns all control point coordinates of the given image.
 * 
 * @param 	nRows 	[in]	The number of rows of the image, this will give the height of the returned grid.
 * @param 	nCols 	[in]	The number of cols of the image, this will give the width of the returned grid.
 * @param 	nodes 	[out]	The nodes of the grid. Thereby the two components that should be stored for each node
 *							are the x and y location within the image.
 * @param	edges 	[out]	The edges of the grid. Thereby the two components are the indices of the nodes which
 *							are connected. A connection only exists if the length of the neighboring nodes is 1.
 *
 * @author
 * Markus Näther
 */
void computeDenseControlPointGraph(
	int nRows, 
	int nCols, 
	blitz::Array<blitz::TinyVector<int, 2>, 1> & nodes,
	blitz::Array<blitz::TinyVector<int, 2>, 1> & edges)
{
	int _nNodes = nRows*nCols;
    int _nEdges = (nRows-1)*(nCols-1)*2 + (nRows-1) + (nCols-1);
    nodes.resize(_nNodes);
    edges.resize(_nEdges);

    
    int _nCounter = 0;
    for(int i = 0; i < nRows; ++i)
    {
		for(int j = 0; j < nCols; ++j)
		{
		    int _i = i*nCols+j;
		    nodes(_i)=blitz::TinyVector<int,2>(i,j);	  

		    // Are we within the image? If yes we can add this node, otherwise we are
		    // at the border and shouldn't add it.
		    // Also tried to add the upper neighbors but this leads to much more errors
		    if (i < nRows-1) 
		    {
		      edges(_nCounter++) = blitz::TinyVector<int,2>(_i,_i+nCols);
		    }

		    if (j < nCols-1) 
		    {
		      edges(_nCounter++) = blitz::TinyVector<int,2>(_i,_i+1);
		    }
		}
    } 

	
}


/**
 * @brief
 * This method returns the displacement hypotheses in the one-dimensional array @see labels.
 * 
 * @param 	radius	[in]	Some radius?
 * @param 	labels	[out]	The labels?
 * 
 * @author
 * Markus Näther
 */
void computeDenseDisplacementHypotheses(
	int radius, 
	blitz::Array<blitz::TinyVector<int, 2>, 1> & labels)
{
	int _nWidth = 2*radius + 1;
	labels.resize(_nWidth*_nWidth);

	int _nCounter = 0;
	for (int i = -radius; i <= radius; ++i)
	{
		for (int j = -radius; j <= radius; ++j, ++_nCounter)
		{
			///TODO: Is this correct this way?
			labels(_nCounter) = blitz::TinyVector<int, 2>(i, j);
		}
	}
}


/**
 * @brief
 * This method evaluates the unary cost at each control point for each displacement hypthesis. SSD is used for 
 * similarity measurement.
 * 
 * @param 	nodes		[in]	All nodes which where precomputed
 * @param 	labels 		[in]	The labels??
 * @param 	srcIm 		[in]	The source image for measuring the costs.
 * @param 	trgIm		[in]	The target image for measuring the costs.
 * @param 	unaryCosts 	[out]	Array containing the unary costs, has the shape nLabels x nNodes.
 * 
 * author
 * Markus Näther
 */
void computeUnaryCostsSSD(
	const blitz::Array<blitz::TinyVector<int, 2>, 1> & nodes,
	const blitz::Array<blitz::TinyVector<int, 2>, 1> & labels, 
	const blitz::Array<float, 2> & srcIm, 
	const blitz::Array<float, 2> & trgIm,
	blitz::Array<float, 2> & unaryCosts)
{
	int _nRows = srcIm.extent(0);
    int _nCols = srcIm.extent(1);
    int _nNodes = nodes.extent(0);
    int _nLabels = labels.extent(0);
    
    unaryCosts.resize(_nLabels, _nNodes);
    
    // Loop through all nodes
    for(int n = 0; n < _nNodes; ++n)
    {
		blitz::TinyVector<int,2> _cSrcPos = nodes(n);
		
		// Loop through all labels 
		for(int l = 0; l < _nLabels; ++l)
		{
		    blitz::TinyVector<int,2> _cTrgPos = _cSrcPos + labels(l);

		    if(_cTrgPos(0) < 0 || _cTrgPos(0) >= _nRows) 
		    	_cTrgPos(0) = std::max(std::min(_nRows-1, _cTrgPos(0)), 0);
		    if(_cTrgPos(1) < 0 || _cTrgPos(1) >= _nCols) 
		    	_cTrgPos(1) = std::max(std::min(_nCols-1, _cTrgPos(1)), 0);

		    float temp = srcIm(_cSrcPos) - trgIm(_cTrgPos);

		    unaryCosts(l,n) = (temp*temp) / 255.0;
		}
	}
}

/**
 * @brief
 * This method evaluates the pairwise cost of each pair of displacement hypotheses. Thereby the L2 norm of
 * the difference of two vectors will be used to compare the pairwise costs.
 * 
 * @param [in]	labels 			This array contains all labels.
 * @param [out]	pairwiseCosts 	This array contains the pairwise costs of all labels, it has the shape
 * 								nLabels x nLabels.
 * 
 * @author
 * Markus Näther
 */
void computePairwiseCostsL2(
	const blitz::Array<blitz::TinyVector<int, 2>, 1> & labels,
	blitz::Array<float, 2> & pairwiseCosts)
{
	// Temporarily save the length
	int _nLength = labels.extent(0);
	// First resite pairwiseCosts
	pairwiseCosts.resize(_nLength, _nLength);

	for (int i = 0; i < _nLength; ++i)
	{

		for (int j = 0; j < _nLength; ++j)
		{
			blitz::TinyVector<int, 2> _cTemp = labels(i) - labels(j);
			pairwiseCosts(i, j) = std::sqrt(blitz::dot(_cTemp, _cTemp));
		}
	}
}



void computeUnaryCostsNCC(
	const blitz::Array<blitz::TinyVector<int, 2>, 1> & nodes,
	const blitz::Array<blitz::TinyVector<int, 2>, 1> & labels,
	blitz::Array<float, 2> & srcIm, 
	blitz::Array<float, 2> & trgIm,
	blitz::Array<float, 2> & unaryCosts)
{
	int _nRows = srcIm.extent(0);
    int _nCols = srcIm.extent(1);
    int _nNodes = nodes.extent(0);
    int _nLabels = labels.extent(0);
    int _nRadius = 3;
    int _nWidth = 2*_nRadius + 1;
    
    unaryCosts.resize(_nLabels, _nNodes);



    // Loop through all nodes
    for(int n = 0; n < _nNodes; ++n)
    {
		blitz::TinyVector<int,2> _cSrcPos = nodes(n);
		
		// Loop through all labels 
		for(int l = 0; l < _nLabels; ++l)
		{
			blitz::TinyVector<int,2> _cTrgPos = _cSrcPos + labels(l);

			// Okay, here's the meat

			// Determine the patch, including the determination of the borders!

			// Calculate the means of src and trg here
			float _fMeanSrc = 0, _fMeanTrg = 0;
			for (int i = -_nRadius; i < _nRadius; ++i)
			{
				for (int j = -_nRadius; j < _nRadius; ++j)
				{
					_fMeanSrc += 0.0f;
					_fMeanTrg += 0.0f;
				}
			}
			// Now normalize
			_fMeanSrc /= (_nRadius*_nRadius);
			_fMeanTrg /= (_nRadius*_nRadius);

			// Now: Demean the images and calculate the variance
		}

	}
}


/**
 * @brief
 * This method will perform an inverse transformation on the target image @see trgIm. Thereby the target image
 * @see trgIm will be transformed and saved within the out parameter @see warpedBackTrgIm.
 * 
 * @param [out]	warpedBackTrgIm 	
 * @param [in]	trgIm 				
 * @param [in]	deformationField 	
 * 
 * @author
 * Markus Näther
 */
void warpImage(
	blitz::Array<float, 2> & warpedBackTrgIm, 
	const blitz::Array<float, 2> & trgIm,
	const blitz::Array<blitz::TinyVector<int, 2>, 2> & deformationField)
{
	warpedBackTrgIm.resize(trgIm.shape());

	for (int x = 0; x < trgIm.extent(0); ++x)
	{
		for (int y = 0; y < trgIm.extent(1); ++y)
		{
			// Calculate the position
			blitz::TinyVector<int,2> _tPos = blitz::TinyVector<int,2>(x, y) +  deformationField(x, y);
			//int _y = y * deformationField(x, y)(1);

			if ((_tPos(0) >= 0 && _tPos(0) < trgIm.extent(0)) && (_tPos(1) >= 0 && _tPos(1) < trgIm.extent(1)))
			{
				warpedBackTrgIm(x, y) = trgIm(_tPos);
			}
			else 
			{
				warpedBackTrgIm(x, y) = 0.0;
			}
		}
	}
}



/**
 * @brief
 * Very simple method for creating a grid.
 *
 * @param [in] nRows The number of rows.
 * @param [in] nCols The number of cols.
 * @param [in] nDelta The space between two lines.
 */
blitz::Array<float, 2> generateGrid(int nRows, int nCols, int nDelta = 9)
{
	// Create the grid
	blitz::Array<float,2> _cResult(nRows,nCols);
	// Fill it with black
	_cResult = 0.0f;
	// And just highlight all lines, depending on the 'nDelta' between the lines of the grid to draw.
	_cResult(blitz::Range(0, nRows - 1, nDelta), blitz::Range::all()) = 255.0f;
	_cResult(blitz::Range::all(), blitz::Range(0, nCols - 1, nDelta)) = 255.0f;

	return _cResult;
}



int main(int argc, char ** argv)
{

	if (argc < 4)
		printUsage();


	std::string _sFilename1(argv[1]);
	std::string _sFilename2(argv[2]);
	int _nMode = atoi(argv[3]);
	float _fLambda = 0.100f;
	if (argc == 5)
		_fLambda = atof(argv[4]);

	std::string _sOutFilename = "out.h5";
	std::string _sLocation = "image";
	

	//
	// Read in the data of both given files
	//
	blitz::Array<float, 2> _cSrcIm;
	blitz::Array<float, 2> _cTrgIm;
	readHDF5toBlitz(_sFilename1, _sLocation, _cSrcIm);
	readHDF5toBlitz(_sFilename2, _sLocation, _cTrgIm);


	// Loaded the target image, so because we make a backward transformation, get the 
	// num of cols and rows from this image
	const int _nRows = _cTrgIm.extent(0);
	const int _nCols = _cTrgIm.extent(1);


	// 
	// Here we compute the dense control point graph
	// 
	blitz::Array< blitz::TinyVector<int,2>, 1> _cNodes;
	blitz::Array< blitz::TinyVector<int,2>, 1> _cEdges;
	computeDenseControlPointGraph( _nRows, _nCols, _cNodes, _cEdges);

	// 
	// Compute the displacement hyptheses here, will later be used
	// 
	int _nRadius = 10;
	blitz::Array<blitz::TinyVector<int, 2>, 1> _cLabels;
	computeDenseDisplacementHypotheses(_nRadius, _cLabels);



	///TODO: Compute costs here!
	blitz::Array<float, 2> _cUnaryCosts;
	switch (_nMode)
	{
		case 1:
		{
			computeUnaryCostsNCC(_cNodes, _cLabels, _cSrcIm, _cTrgIm, _cUnaryCosts);
		} 
		case 0:
		default:
		{
			computeUnaryCostsSSD(_cNodes, _cLabels, _cSrcIm, _cTrgIm, _cUnaryCosts);
		}
	};



	///TODO: Pairwise costs!
	blitz::Array< float, 2> _cPairwiseCosts;
	computePairwiseCostsL2(_cLabels, _cPairwiseCosts);
	_cPairwiseCosts *= _fLambda;


	// START
	// Part from exercise sheet
	// 

	const int _nMaxIterations = 1;
	int _nNumNodes = _cNodes.extent(0);
	int _nNumLabels = _cLabels.extent(0);
	int _nNumEdges = _cEdges.extent(0);

	// 
	// Create the edge weights an initialize them with 1
	// 
	blitz::Array<float, 1> _cEdgeWeights(_nNumEdges);
	_cEdgeWeights = 1.0;

	float * _nodes = reinterpret_cast<float*>(_cNodes.dataFirst());
	float * _labels = reinterpret_cast<float*>(_cLabels.dataFirst());
	float * _unaryCosts = reinterpret_cast<float*>(_cUnaryCosts.dataFirst());
	int * _edges = reinterpret_cast<int*>(_cEdges.dataFirst());
	float * _pairwiseCosts = reinterpret_cast<float*>(_cPairwiseCosts.dataFirst());
	float * _edgeWeights = reinterpret_cast<float*>(_cEdgeWeights.dataFirst());

	CV_Fast_PD _cPD(_nNumNodes, _nNumLabels, _unaryCosts, _nNumEdges, _edges, _pairwiseCosts, _nMaxIterations, _edgeWeights);
	_cPD.run();

	blitz::Array<int, 1> optimalLabels(_nNumNodes);
	for (int i = 0; i < _nNumNodes; ++i)
	{
		optimalLabels(i) = _cPD._pinfo[i].label;
	}
	blitz::Array<blitz::TinyVector<int, 2>, 2> deformationField(_cSrcIm.shape());
	for (int i = 0; i < _nNumNodes; ++i)
	{
		deformationField(_cNodes(i)) = _cLabels(optimalLabels(i));
	}

	// END
	// Part from exercise sheet
	// 

	blitz::Array< blitz::TinyVector<float,3>, 2> _cDeformField;
	float _fScaling = 1.0f/sqrt(2.0f*(_nRadius*_nRadius));	// maximum displacement
	flowToImage( deformationField, _cDeformField, _fScaling); 


	// 
	blitz::Array<float,2> _cWarpedTrgIm(_cTrgIm.extent(0), _cTrgIm.extent(1));
	blitz::Array<float,2> _cWarpedGrid(_cTrgIm.extent(0), _cTrgIm.extent(1));
	warpImage(_cWarpedTrgIm, _cTrgIm, deformationField);
	blitz::Array<float,2> _cGrid(generateGrid(_cTrgIm.extent(0), _cTrgIm.extent(1)));
	warpImage(_cWarpedGrid, _cGrid, deformationField);


	// Now save the results
	writeBlitzToHDF5(_cSrcIm, "_cSrcIm", _sOutFilename);
	writeBlitzToHDF5(_cGrid, "_cGrid", _sOutFilename);
	writeBlitzToHDF5(_cWarpedGrid, "_cWarpedGrid", _sOutFilename);
	writeBlitzToHDF5(_cTrgIm, "_cTrgIm", _sOutFilename);
	writeBlitzToHDF5(_cWarpedTrgIm, "_cWarpedTrgIm", _sOutFilename);
	writeBlitzToHDF5(_cDeformField, "_cDeformField", _sOutFilename);

	return 0;
}