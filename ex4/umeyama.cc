
#include <iostream>
#include <fstream> // For streams
#include <sys/stat.h> // For mkdir
#include <cfloat> // For FLT_MAX
#include <cmath>

#include <blitz/array.h>
#include <blitz/tinyvec2.h>

#include <gsl/gsl_matrix.h> // For gsl_matrix
#include <gsl/gsl_linalg.h> // For lincomp

#include "./BlitzHDF5Helper.hh" // For helper methods for reading hdf5 format
#include "./my_blitz.hh" // For generic myproduct

//#define RM__PI 3.14159265359f
#define IA_EX04_RADIUS 3

const float ex04_pi = std::acos(-1);


/*

	Copied from demo.cc, maybe we will need this here too.
Compiling this on Laptop requires the following command line:
	g++ -Wall -O2 -L/usr/local/HDF_Group/HDF5/1.8.15/lib/ -I/usr/local/HDF_Group/HDF5/1.8.15/include/ -g umeyama.cc -lblitz -lhdf5 -ldl -lgsl -lgslcblas -o umeyama
	!

 */


/**
 * @brief 
 * This method will calculate and return the product of two matrices. 
 * 
 * @param [in] a The first matrix.
 * @param [in] b The second matrix.
 * 
 * @return 
 * The product of the calculation.
 */
blitz::TinyMatrix<float,4,4>  myproduct( blitz::TinyMatrix<float,4,4>  a , blitz::TinyMatrix<float,4,4>  b)
{
  blitz::TinyMatrix<float, 4, 4> _result;

  _result(0,0) = a(0,0)*b(0,0) + a(0,1)*b(1,0) + a(0,2)*b(2,0) + a(0,3)*b(3,0);
  _result(1,0) = a(1,0)*b(0,0) + a(1,1)*b(1,0) + a(1,2)*b(2,0) + a(1,3)*b(3,0);
  _result(2,0) = a(2,0)*b(0,0) + a(2,1)*b(1,0) + a(2,2)*b(2,0) + a(2,3)*b(3,0);
  _result(3,0) = a(3,0)*b(0,0) + a(3,1)*b(1,0) + a(3,2)*b(2,0) + a(3,3)*b(3,0);
  
  _result(0,1) = a(0,0)*b(0,1) + a(0,1)*b(1,1) + a(0,2)*b(2,1) + a(0,3)*b(3,1);
  _result(1,1) = a(1,0)*b(0,1) + a(1,1)*b(1,1) + a(1,2)*b(2,1) + a(1,3)*b(3,1);
  _result(2,1) = a(2,0)*b(0,1) + a(2,1)*b(1,1) + a(2,2)*b(2,1) + a(2,3)*b(3,1);
  _result(3,1) = a(3,0)*b(0,1) + a(3,1)*b(1,1) + a(3,2)*b(2,1) + a(3,3)*b(3,1);
  
  _result(0,2) = a(0,0)*b(0,2) + a(0,1)*b(1,2) + a(0,2)*b(2,2) + a(0,3)*b(3,2);
  _result(1,2) = a(1,0)*b(0,2) + a(1,1)*b(1,2) + a(1,2)*b(2,2) + a(1,3)*b(3,2);
  _result(2,2) = a(2,0)*b(0,2) + a(2,1)*b(1,2) + a(2,2)*b(2,2) + a(2,3)*b(3,2);
  _result(3,2) = a(3,0)*b(0,2) + a(3,1)*b(1,2) + a(3,2)*b(2,2) + a(3,3)*b(3,2);
  
  _result(0,3) = a(0,0)*b(0,3) + a(0,1)*b(1,3) + a(0,2)*b(2,3) + a(0,3)*b(3,3);
  _result(1,3) = a(1,0)*b(0,3) + a(1,1)*b(1,3) + a(1,2)*b(2,3) + a(1,3)*b(3,3);
  _result(2,3) = a(2,0)*b(0,3) + a(2,1)*b(1,3) + a(2,2)*b(2,3) + a(2,3)*b(3,3);
  _result(3,3) = a(3,0)*b(0,3) + a(3,1)*b(1,3) + a(3,2)*b(2,3) + a(3,3)*b(3,3);

  return _result;
}
blitz::TinyMatrix<float,3,3>  myproduct( blitz::TinyMatrix<float,3,3>  a , blitz::TinyMatrix<float,3,3>  b)
{
  blitz::TinyMatrix<float, 3, 3> _result;

  _result(0,0) = a(0,0)*b(0,0) + a(0,1)*b(1,0) + a(0,2)*b(2,0);
  _result(1,0) = a(1,0)*b(0,0) + a(1,1)*b(1,0) + a(1,2)*b(2,0);
  _result(2,0) = a(2,0)*b(0,0) + a(2,1)*b(1,0) + a(2,2)*b(2,0);
  
  _result(0,1) = a(0,0)*b(0,1) + a(0,1)*b(1,1) + a(0,2)*b(2,1);
  _result(1,1) = a(1,0)*b(0,1) + a(1,1)*b(1,1) + a(1,2)*b(2,1);
  _result(2,1) = a(2,0)*b(0,1) + a(2,1)*b(1,1) + a(2,2)*b(2,1);
  
  _result(0,2) = a(0,0)*b(0,2) + a(0,1)*b(1,2) + a(0,2)*b(2,2);
  _result(1,2) = a(1,0)*b(0,2) + a(1,1)*b(1,2) + a(1,2)*b(2,2);
  _result(2,2) = a(2,0)*b(0,2) + a(2,1)*b(1,2) + a(2,2)*b(2,2);

  return _result;
}


/**
 * @brief 
 * This method will calculate and return the product of a matrix and a vector.
 * 
 * @param [in] m The matrix.
 * @param [in] v The vector.
 * 
 * @return 
 * The result of the calculation.
 */
blitz::TinyVector<float,4>  myproduct( blitz::TinyMatrix<float,4, 4>  m , blitz::TinyVector<float,4>  v)
{
  blitz::TinyVector<float, 4> _result;

  _result(0) = m(0,0)*v(0) + m(0,1)*v(1) + m(0,2)*v(2) + m(0,3)*v(3);
  _result(1) = m(1,0)*v(0) + m(1,1)*v(1) + m(1,2)*v(2) + m(1,3)*v(3);
  _result(2) = m(2,0)*v(0) + m(2,1)*v(1) + m(2,2)*v(2) + m(2,3)*v(3);
  _result(3) = m(3,0)*v(0) + m(3,1)*v(1) + m(3,2)*v(2) + m(3,3)*v(3);

  return _result;
}


////////////////////////////////////////////////////////////////////////////////
// Old, slightly adapted part
////////////////////////////////////////////////////////////////////////////////



/**
 * @brief
 * This method receives a blitz::TinyMatrix and returns its transposed matrix. Thereby you only
 */
blitz::TinyMatrix<float, 3, 3> getTransposedMatrix(const blitz::TinyMatrix<float, 3, 3> & cMat)
{
	blitz::TinyMatrix<float, 3, 3> _cResult;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			_cResult(i, j) = cMat(j, i);
		}
	}

	return _cResult;
}



/**
 * @brief 
 * This method will return the interpolated value of the specified position 'pos'. Currently this is 
 * just a nearest neighborhood interpolation. Later on in the lecture more advanced interpolators
 * will be implemented.
 * If the requested position is outside of the array bounds 0 will be returned.
 * 
 * @param [in] arr This array represents the complete image.
 * @param [in] pos This is the position.
 *  
 * @return The interpolated value.
 */
float interpolNN(const blitz::Array<float, 3> & arr, blitz::TinyVector<float, 4> pos)
{
	// Keep in mind that we have to use the 4th component of the vector 'pos' for normalization.
  // This is necessary because where we are using homogenous coordinates
  if (pos(3) != 1)
    pos /= pos(3);

  // Now that we have the normalized position let's find the nearest neighbor.
  blitz::TinyVector<int, 3> _cPos;
  _cPos = blitz::floor(pos + 0.5);

  // If we are within the borders let's return the value at position _cPos in arr, otherwise just return 0
  return (blitz::all(_cPos >= 0) && blitz::all(_cPos < arr.shape())) ? arr(_cPos) : 0;
}


/**
 * @brief
 * This method will take the @p srcArr, rotate it by using @p invMat and save the result of the rotation inside of @p trgArr.
 *
 * @note
 * Keep in mind that we are applying the inverse rotation, so that we can use the method presented in the lecture: 
 *  Starting with the resulting images, loop through all voxels/pixels and determine where the source voxel/pixel was
 *  in the original image. After determing that a nearest neighborhood interpolation will be applied.
 */
void transformArray(const blitz::Array<float, 3> & srcArr, const blitz::TinyMatrix<float, 4, 4> & invMat, blitz::Array<float, 3> & trgArr)
{
  blitz::TinyVector<float,4> _cTemp;
  blitz::TinyVector<float,4> _cResult;

  // For every voxel within the rotated voxel image (trgArr) take the inverse of the transformation and look where this position is in the
  // original data (srcArra). Finally find the nearest voxel by using interpolNN and write the value to trgArr.
  for( int l = 0; l < trgArr.extent(0); ++l)
  {
    for( int r = 0; r < trgArr.extent(1); ++r)
    {
      for( int c = 0; c < trgArr.extent(2); ++c)
      {
        // The naming is a bit confusing, _cResult is the current position in the rotated image, 
        // On this position apply the inverse transformation to receive the position in the orignal image (_cTemp).
        // Now we can get the nearest neighbor at position _cTemp by using interpolNN.
        _cResult = l, r, c, 1;
        _cTemp = myproduct( invMat, _cResult);
        trgArr(l,r,c) = interpolNN( srcArr, _cTemp);
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
// New part
////////////////////////////////////////////////////////////////////////////////

void drawCubic(blitz::Array<float, 3> & Image, int nRadius, const blitz::TinyVector<double, 3> & cLandMark)
{
	// Make a small annotation in @see 'Image' to highlight it in some kind

		// Slightly light up the inner voxel a bit
		Image(blitz::Range(cLandMark(0) - nRadius, 
						   cLandMark(0) + nRadius),
			  blitz::Range(cLandMark(1) - nRadius, 
			  			   cLandMark(1) + nRadius),
			  blitz::Range(cLandMark(2) - nRadius,
			  			   cLandMark(2) + nRadius)) += 300; // Multiplication is not the right thing here, dark regions would stay dark (use addition)

		// Draw the edges of the cube with a 'hard color'
		Image(blitz::Range(cLandMark(0) - nRadius, 
						   cLandMark(0) + nRadius),
			  cLandMark(1) - nRadius,
			  cLandMark(2) - nRadius) = 1000;
		Image(blitz::Range(cLandMark(0) - nRadius, 
						   cLandMark(0) + nRadius),
			  cLandMark(1) - nRadius,
			  cLandMark(2) + nRadius) = 1000;
		Image(blitz::Range(cLandMark(0) - nRadius, 
						   cLandMark(0) + nRadius),
			  cLandMark(1) + nRadius,
			  cLandMark(2) - nRadius) = 1000;
		Image(blitz::Range(cLandMark(0) - nRadius, 
						   cLandMark(0) + nRadius),
			  cLandMark(1) + nRadius,
			  cLandMark(2) + nRadius) = 1000;

		Image(cLandMark(0) - nRadius,
			  blitz::Range(cLandMark(1) - nRadius, 
						   cLandMark(1) + nRadius),
			  cLandMark(2) - nRadius) = 1000;
		Image(cLandMark(0) - nRadius,
			  blitz::Range(cLandMark(1) - nRadius, 
						   cLandMark(1) + nRadius),
			  cLandMark(2) + nRadius) = 1000;
		Image(cLandMark(0) + nRadius,
			  blitz::Range(cLandMark(1) - nRadius, 
						   cLandMark(1) + nRadius),
			  cLandMark(2) - nRadius) = 1000;
		Image(cLandMark(0) + nRadius,
			  blitz::Range(cLandMark(1) - nRadius, 
						   cLandMark(1) + nRadius),
			  cLandMark(2) + nRadius) = 1000;

		Image(cLandMark(0) - nRadius,
			  cLandMark(1) - nRadius,
			  blitz::Range(cLandMark(2) - nRadius, 
						   cLandMark(2) + nRadius)) = 1000;
		Image(cLandMark(0) - nRadius,
			  cLandMark(1) + nRadius,
			  blitz::Range(cLandMark(2) - nRadius, 
						   cLandMark(2) + nRadius)) = 1000;
		Image(cLandMark(0) + nRadius,
			  cLandMark(1) - nRadius,
			  blitz::Range(cLandMark(2) - nRadius, 
						   cLandMark(2) + nRadius)) = 1000;
		Image(cLandMark(0) + nRadius,
			  cLandMark(1) + nRadius,
			  blitz::Range(cLandMark(2) - nRadius, 
						   cLandMark(2) + nRadius)) = 1000;
}

/**
 * @brief
 * This method will highlight the landmarks @see landmarks with a cross, a cube or a ball in a given image @see Image.
 * 
 * @param [in/out] Image 		The image itself, where the landmarks should be marked.
 * @param [in] landmarks 		The array containing all the landmarks.
 * @param [in] element_size_um 	For scaling the 3d cube.
 * 
 * @HINT
 * Use blitz::Range operator to do this efficiently.
 * 
 * @author
 * Markus Näther
 */
void markLandmark(blitz::Array<float, 3> & Image, const blitz::Array<blitz::TinyVector<double, 3>, 1> & landmarks, blitz::TinyVector<float, 3> element_size_um)
{
	// Compute the reciprocal so we avoid many divisions in the for loop. Multiplications are faster than divisions
	blitz::TinyVector<float, 3> _cElementSizeUm = 1.0f/element_size_um;

	// Let's make it simple and just draw a box.
	const int _nRadius = IA_EX04_RADIUS;

	// For all landmarks
	for (int l = 0; l < landmarks.extent(0); ++l)
	{
		// Get the next landmark
		blitz::TinyVector<double, 3> _cL = landmarks(l) * _cElementSizeUm;
		
		// Draw a cube around the landmark (what we are saving in the computation of the landmark position is wasted here ;) 
		drawCubic(Image, _nRadius, _cL);
		
	}
}


/**
 * @brief
 * This method takes two input images @see fixedImage and @see movingImages and combines both into one single rgb image.
 * Keep ind mind to assign the fixed image to the first channel, while assigning the second image to the second channel. 
 * 
 * @note
 * Both images have different sizes! SO the out image will be as big as the max between both images.
 * 
 * @param [in]	fixedImage 		The fixed image.
 * @param [in]	movingImage 	Logically, the moving image that should be arranged correctly.
 * 
 * @return
 * The resulting rgb image which pops out if both images are combined.
 * 
 * @author
 * Markus Näther
 */
blitz::Array<blitz::TinyVector<float, 3>, 3> overlay(const blitz::Array<float, 3> & fixedImage, const blitz::Array<float, 3> & movingImage)
{
	blitz::TinyVector<int, 3> _cFShape = fixedImage.shape(), _cMShape = movingImage.shape();

	// Create the output image
	blitz::Array<blitz::TinyVector<float, 3>, 3> _cResult(std::max(_cFShape(0), _cMShape(0)), std::max(_cFShape(1), _cMShape(1)), std::max(_cFShape(2), _cMShape(2)));


	_cResult(blitz::Range(0, _cFShape(0)-1), blitz::Range(0, _cFShape(1)-1), blitz::Range(0, _cFShape(2)-1))[0] = fixedImage;	// Fixed image to first channel
	_cResult(blitz::Range(0, _cMShape(0)-1), blitz::Range(0, _cMShape(1)-1), blitz::Range(0, _cMShape(2)-1))[1] = movingImage;	// Moving image to second channel
	_cResult[2] = 0.0f; 		// Cancel out the last channel

	// Finally return it
	return _cResult;
}

/**
 * @brief
 * Simple wrapper method for populating the values of @see blitzMat in @see pGslMat.
 * 
 * @param [in/out]	pGslMat 		Pointer to a gsl_matrix instance, which should be populated.
 * @param [in/out]	blitzMat 		Reference of a blitz TinyMatrix, whose parameters should be copied to @see pGslMat.
 * 
 * @author
 * Markus Näther
 */
void gslMatrixAssignWrapper(gsl_matrix * pGslMat, blitz::TinyMatrix<float, 3, 3> & blitzMat)
{
	// Okay, there is NO documentation of TinyMatrix .... do the assignment hard coded.
	if (pGslMat)
	{
		for (int i = 0; i < 3; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				gsl_matrix_set(pGslMat, i, j, blitzMat(i, j));
			}
		}
	}
}

/**
 * @brief
 * This method will allocate memory of a gsl_matrix, fill it with the data provided through @see blitzMat and
 * return the created instance of gsl_matrix.
 * 
 * @param [in]	blitzMat 	Reference to the blitz TinyMatrix that should be copied to a gsl_matrix instance.
 * 
 * @return Pointer to the created gsl_matrix instance. 
 * 
 * @author
 * Markus Näther
 */
gsl_matrix * gslMatrixAllocWrapper(blitz::TinyMatrix<float, 3, 3> & blitzMat)
{
	gsl_matrix * _cResult = gsl_matrix_calloc(3, 3);

	// If everything worked fine we assign the values of @see blitzMat to the gsl_matrix _cResult.
	if (_cResult != 0)
		gslMatrixAssignWrapper(_cResult, blitzMat);

	return _cResult;
}

/**
 * @brief
 * This method will receive a pointer to a gsl_matrix, @see pGslMat. If this pointer is valid and
 * not 0 it will be freed and set to zer0.
 */
void gslMatrixFreeWrapper(gsl_matrix * pGslMat)
{
	// Only free structure if it is still valid
	if (pGslMat != 0)
		gsl_matrix_free(pGslMat);

	// Set it to 0, so we don't get a double free error
	pGslMat = 0;
}


double gslMatrixAccessWrapper(gsl_matrix * pGslMat, size_t i, size_t j)
{
	if (pGslMat)
		return gsl_matrix_get(pGslMat, i, j);

	return 0.0;
}


void gslMatrixSetWrapper(gsl_matrix * pGslMat, size_t i, size_t j, double v)
{
	if (pGslMat)
		gsl_matrix_set(pGslMat, i, j, v);
}


/**
 * @brief
 * This method is a wrapper for assigning the values of a blitz::TinyVector to a gsl_vector.
 * 
 * @author
 * Markus Näther
 */
void gslVectorAssignWrapper(gsl_vector * pVec, blitz::TinyVector<float, 3> & cVec)
{
	if (pVec)
	{
		for (int i = 0; i < cVec.extent(0); ++i)
		{
			gsl_vector_set(pVec, i, cVec(i));
		}
	}
}

/**
 * @brief
 * This method is a wrapper for allocating gsl_vectors and assigning values to it.
 * 
 * @author
 * Markus Näther
 */
gsl_vector * gslVectorAllocWrapper(blitz::TinyVector<float, 3> & cVec)
{
	gsl_vector * _pResult = gsl_vector_alloc(cVec.extent(0));

	if (_pResult)
	{
		gslVectorAssignWrapper(_pResult, cVec);

		return _pResult;
	}

	return 0;
}

gsl_vector * gslVectorAllocWrapper(size_t nSize)
{
	gsl_vector * _pResult = gsl_vector_alloc(nSize);

	return (_pResult != 0) ? _pResult : 0;
}

/**
 * @brief
 * This method is a wrapper for the freeing of gsl_vectors.
 * 
 * @author
 * Markus Näther
 */
void gslVectorFreeWrapper(gsl_vector * pVec)
{
	if (pVec)
		gsl_vector_free(pVec);
}


double gslVectorAccessWrapper(gsl_vector * pVec, size_t i)
{
	if (pVec)
		return gsl_vector_get(pVec, i);

	return 0.0;
}

void gslVectorSetWrapper(gsl_vector * pVec, size_t i, double cVal)
{
	if (pVec)
		gsl_vector_set(pVec, i, cVal);
}


blitz::TinyVector<float, 3> blitzTinyVectorAssignWrapper(gsl_vector * pVec, size_t inSize)
{
	blitz::TinyVector<float, 3> _cResult;

	if (pVec != 0)
	{
		for (size_t i = 0; i < inSize; ++i)
			_cResult(i) = gsl_vector_get(pVec, i);
	}

	return _cResult;
}

blitz::TinyMatrix<float, 3, 3> blitzTinyMatrixAssignWrapper(gsl_matrix * pMat, size_t inI, size_t inJ)
{
	blitz::TinyMatrix<float, 3, 3> _cResult;

	if (pMat)
	{
		for (size_t i = 0; i < inI; ++i)
		{
			for (size_t j = 0; j < inJ; ++j)
			{
				_cResult(i, j) = gsl_matrix_get(pMat, i, j);
			}
		}
	}

	return _cResult;
}


double gslMatrixDeterminant(gsl_matrix * pMat, size_t nI, size_t nJ)
{
	double _dDeterminant = 0.0;

	if (pMat)
	{
		_dDeterminant = gsl_matrix_get(pMat, 0, 0)*(gsl_matrix_get(pMat, 1, 1) * gsl_matrix_get(pMat, 2, 2) - gsl_matrix_get(pMat, 1, 2) * gsl_matrix_get(pMat, 2, 1)) +
						gsl_matrix_get(pMat, 0, 1)*(gsl_matrix_get(pMat, 1, 2) * gsl_matrix_get(pMat, 2, 0) - gsl_matrix_get(pMat, 1, 0) * gsl_matrix_get(pMat, 2, 2)) +
						gsl_matrix_get(pMat, 0, 2)*(gsl_matrix_get(pMat, 1, 0) * gsl_matrix_get(pMat, 2, 1) - gsl_matrix_get(pMat, 1, 1) * gsl_matrix_get(pMat, 2, 0));
	}

	return _dDeterminant;
}

float blitzTinyMatrixDeterminant(const blitz::TinyMatrix<float, 3, 3> & cMat)
{
	float _dDeterminant = 0.0;

	_dDeterminant = cMat(0, 0)*(cMat(1, 1) * cMat(2, 2) - cMat(1, 2) * cMat(2, 1)) +
					cMat(0, 1)*(cMat(1, 2) * cMat(2, 0) - cMat(1, 0) * cMat(2, 2)) +
					cMat(0, 2)*(cMat(1, 0) * cMat(2, 1) - cMat(1, 1) * cMat(2, 0));

	return _dDeterminant;
}

double calculateMatrixTrace(const blitz::TinyMatrix<float, 3, 3> & cMat)
{
	return cMat(0,0) + cMat(1, 1) + cMat(2, 2);
}


/**
 * @brief
 * This method will calculate and return the components of the SVD as method parameters.
 * 
 * @note
 * Use gsl_linalg_SV_decomp(...) in here.
 * 	-> A wrapper is necessary to convert TinyMatrix into the gsl_matrix format!
 * 
 * @author
 * Markus Näther
 */
void mySVD(blitz::TinyMatrix<float, 3, 3> & A, blitz::TinyVector<float, 3> & S, blitz::TinyMatrix<float, 3, 3> & V)
{
	// There should be no need to propagate the values back, but these references are just for internal usage.
	blitz::TinyMatrix<float, 3, 3> & _A = A;
	blitz::TinyMatrix<float, 3, 3> & _V = V;
	blitz::TinyVector<float, 3> & _S = S;
	gsl_matrix * _pGslA = gslMatrixAllocWrapper(_A);
	gsl_vector * _pGslS = gslVectorAllocWrapper(_S);
	gsl_matrix * _pGslV = gslMatrixAllocWrapper(_V);
	gsl_vector * _pWork = gslVectorAllocWrapper(3);

	for (int i = 0; i < 3; i++)
		std::cout << " " << gsl_matrix_get(_pGslA, i, 0) << " " << std::endl;

	// Apply the SVD decomposition
	int _nResult = gsl_linalg_SV_decomp(_pGslA, _pGslV, _pGslS, _pWork);

	if (_nResult != 0)
		std::cout << "There seems to be something wrong with the decomp?" << std::endl;

	std::cout << "!..." << std::endl;
	std::cout << "::Determinates of the matrixes:" << std::endl;
	std::cout << "\tA:\t" << gslMatrixDeterminant(_pGslA, 3, 3) << std::endl;
	std::cout << "\tV:\t" << gslMatrixDeterminant(_pGslV, 3, 3) << std::endl;

	///TODO: Handle mirroring so matrices are no longer 

	// Assign the values back to the in/out parameters
	A = blitzTinyMatrixAssignWrapper(_pGslA, 3, 3);
	V = blitzTinyMatrixAssignWrapper(_pGslV, 3, 3);
	S = blitzTinyVectorAssignWrapper(_pGslS, 3);

	// Don't forget to free the memory again!
	gslMatrixFreeWrapper(_pGslA);
	gslMatrixFreeWrapper(_pGslV);
	gslVectorFreeWrapper(_pGslS);
	gslVectorFreeWrapper(_pWork);
}





blitz::TinyMatrix<float,4,4> umeyama(const blitz::Array<blitz::TinyVector<float,4>,1>& points,
 									 const blitz::Array<blitz::TinyVector<float,4>,1>& transformedPoints)
{

	// Use the blitz internal mean method
	blitz::TinyVector<float, 4> _cMean = blitz::mean(points);
	blitz::TinyVector<float, 4> _cTransformedMean = blitz::mean(transformedPoints);


	// Next we should generate the covariance matrix
	blitz::TinyMatrix<float, 3, 3> _cCovarianceMatrix;
	_cCovarianceMatrix = 0;

	/*According to paper: 
	 * For all i .. n
	 * (x_i - mean_x)*(y_i - transformedMean_x)^T
	 */
	for (int n = 0; n < _cMean.extent(0); ++n)
	{
		for (int i = 0; i < 3; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				_cCovarianceMatrix(i, j) += (points(n)(j) - _cMean(j))*(transformedPoints(n)(i) - _cTransformedMean(i));
			}
		}
	}

	// Calculate the 1/n part, precalculate 1/n because multiplication is faster than division
	float _fNum = 1.0f / _cMean.extent(0);
	_cCovarianceMatrix *= _fNum;

	// Next: Single value decomposition, by using the covariance matrix as A
	blitz::TinyMatrix<float, 3, 3> _A = _cCovarianceMatrix;
	blitz::TinyMatrix<float, 3, 3> _V;
	blitz::TinyVector<float, 3> _S;

	mySVD(_A, _S, _V);

	std::cout << std::endl;
	std::cout << "\t!--- [START] DEBUGGING ---!" << std::endl;
	std::cout << "\t" << _A << std::endl;
	std::cout << "\t" << _S << std::endl;
	std::cout << "\t" << _V << std::endl;
	std::cout << "\t!--- [ END ] DEBUGGING ---!" << std::endl;



	//TODO: Calculate the c for the translation
	blitz::TinyVector<float, 4> _fSigma;
	_fSigma = 0.0f;
	blitz::Array<blitz::TinyVector<float,4>,1> _cAddPoints(_cTransformedMean.shape());
	for (int n = 0; n < _cTransformedMean.extent(0); ++n)
	{
		_cAddPoints(n) = transformedPoints(n) - _cTransformedMean(n);
	}
	_fSigma = blitz::mean(_cAddPoints);


	blitz::TinyMatrix<float, 3, 3> S;
	blitz::TinyMatrix<float, 3, 3> D;
	D = 0;
	D(0,0) = _S(0);
	D(1,1) = _S(1);
	D(2,2) = _S(2);
	if (blitzTinyMatrixDeterminant(_A)*blitzTinyMatrixDeterminant(_V) == 1.0f)
	{
		S = 
			1.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 
			0.0f, 0.0f, 1.0f;
	}
	else if (blitzTinyMatrixDeterminant(_A)*blitzTinyMatrixDeterminant(_V) == -1.0f)
	{
		S = 
			1.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 
			0.0f, 0.0f, -1.0f;
	}
	blitz::TinyVector<float, 4> _fC;
	_fC = (calculateMatrixTrace(myproduct(D,S)) / _fSigma);

	std::cout << "\t !--- DEBUGGING OF (sigma: "<< _fSigma << ") _fC: " << _fC << std::endl;


	///: Why is there no way to directly assign the values to a matrix while creation?!
	blitz::TinyMatrix<float, 3, 3> _cRotationMatrix;
	_cRotationMatrix = myproduct(_A, getTransposedMatrix(_V));


	/*
	 * TODO: Somewhere here seems to be a strange bug. 
	 * Also tested it with S = Identity matrix for myproduct(S, getTransposedMatrix(_V)) but the result of this
	 * calculation is not equal to getTransposedMatrix(_V) which should be the case. Implementation of 
	 * myproduct for TinyMatrix<float, 3, 3> is correct so I don't know where the exact problem is, so I left
	 * out this part and would implicitely set _fC to (1.0f, 1.0f, 1.0f, 1.0f) (so just leave it out in the equation).
	 */
	//_cRotationMatrix = myproduct(_A, myproduct(S, getTransposedMatrix(_V)));

	// Add the rotation matrix to the transformation matrix
	blitz::TinyMatrix<float, 4, 4> _cTransformationMatrix;
	_cTransformationMatrix =
		_cRotationMatrix(0, 0), _cRotationMatrix(0, 1), _cRotationMatrix(0, 2), 0.0f, 
		_cRotationMatrix(1, 0), _cRotationMatrix(1, 1), _cRotationMatrix(1, 2), 0.0f, 
		_cRotationMatrix(2, 0), _cRotationMatrix(2, 1), _cRotationMatrix(2, 2), 0.0f, 
		0.0f, 					0.0f, 					0.0f, 					1.0f;


	// Get the translation of the moving points, this is just formula 41 of the umayama paper
	blitz::TinyVector<float, 4> _translationVector = _cTransformedMean - myproduct(_cTransformationMatrix, _cMean);


	//blitz::TinyVector<float, 4> _translationVector = _cTransformedMean - myproduct(, myproduct(_cTransformationMatrix, _cMean),  _fC);


	// Last, add the translation to the transformation matrix
	_cTransformationMatrix(0, 3) = _translationVector(0);
	_cTransformationMatrix(1, 3) = _translationVector(1);
	_cTransformationMatrix(2, 3) = _translationVector(2);
	_cTransformationMatrix(3, 3) = 1.0f;
	//_cTransformationMatrix(3, 3) = _translationVector(3);

	// Finally return the transformation matrix
	return _cTransformationMatrix;
}








void dataLoadingPreparation(const std::string & inFilename, blitz::Array<float, 3> & cImage, blitz::Array<blitz::TinyVector<double, 3>, 1> & cLandmarks, blitz::TinyVector<float, 3> & cElementSizeUm)
{
	// Read the image 
	readHDF5toBlitz(inFilename, "/channel0", cImage);

	// Read the landmarks
	readHDF5toBlitz(inFilename, "landmarks_um", cLandmarks);

	// Read the Element size um vector
	readBlitzTinyVectorFromHDF5Attribute(cElementSizeUm, "element_size_um", "/channel0", inFilename);

	printf("\t!--- HDF5 Information ---!\n");
	std::cout << "\tImage shape: " << cImage.shape() << std::endl;
	std::cout << "\tElementSize: " << cElementSizeUm << std::endl;
}


double getDistance(blitz::TinyVector<float, 4> & a, blitz::TinyVector<float, 4> & b) 
{
	return std::sqrt(a(0)*b(0) + a(1)*b(1) + a(2)*b(2));
}

void programMethod()
{
	printf("Starting the program\n");
	// Load the data
	const std::string inFileName1 = "zebrafish0.h5";
	const std::string inFileName2 = "zebrafish1.h5";
	const std::string sChannel = "/channel0";
	const std::string sElement = "element_size_um";
	const std::string sLandmark_um = "landmarks_um";
	std::string outFileName = "result.h5";

	blitz::Array<float, 3> cFixedImage, cMovingImage;
	blitz::Array<blitz::TinyVector<double, 3>, 1> cLandmarksFixed, cLandmarksMoving;
	blitz::TinyVector<float, 3> cElementSizeFixed, _cElementSizeMoving;

	dataLoadingPreparation(inFileName1, cFixedImage, cLandmarksFixed, cElementSizeFixed);
	dataLoadingPreparation(inFileName2, cMovingImage, cLandmarksMoving, _cElementSizeMoving);

	// Data preparation: transform the landmarks from <double, 3> to <float, 4>
	blitz::Array<blitz::TinyVector<float, 4>, 1> _cTLandmarksFixed, _cTLandmarksMoving;
	_cTLandmarksFixed.resize(cLandmarksFixed.shape());
	_cTLandmarksMoving.resize(cLandmarksMoving.shape());
	// Handle every vector for the landmarks separately because it could always be the case that the sizes of
	// both vectors are inequal (shouldn't be the case, but just to be save.)
	for (int i = 0; i < cLandmarksFixed.extent(0); ++i)
	{
		blitz::TinyVector<float, 4> _t;
		_t = cLandmarksFixed(i)(0), cLandmarksFixed(i)(1), cLandmarksFixed(i)(2), 1.0f;
		_cTLandmarksFixed(i) = _t;
	}
	for (int i = 0; i < cLandmarksMoving.extent(0); ++i)
	{
		blitz::TinyVector<float, 4> _t;
		_t = cLandmarksMoving(i)(0), cLandmarksMoving(i)(1), cLandmarksMoving(i)(2), 1.0f;
		_cTLandmarksMoving(i) = _t;
	}


	// calculate the distance BEFORE
	double _fDistanceBefore = 0.0f;
	for (int i = 0; i < cLandmarksFixed.extent(0); ++i)
	{
		blitz::TinyVector<float,4> _temp1;
		_temp1 = cLandmarksMoving(i)(0), cLandmarksMoving(i)(1), cLandmarksMoving(i)(2), 1.0f;
		blitz::TinyVector<float,4> _temp2;
		_temp2 = cLandmarksFixed(i)(0), cLandmarksFixed(i)(1), cLandmarksFixed(i)(2), 1.0f;
		_fDistanceBefore += getDistance(_temp1, _temp2);
	}
  	std::cout << "Distance before calculating umeyama is: " << _fDistanceBefore << std::endl;


	// Get and apply the transformation now
	blitz::TinyMatrix<float, 4, 4> _cTransformationMatrix = umeyama(_cTLandmarksFixed, _cTLandmarksMoving);

	// Don't forget to apply the scaling through cElementSizeFixed and cElementSizeMoving
	blitz::TinyMatrix<float, 4,4 > _cScalingMovingElements, _cScalingFixedElements;
	_cScalingMovingElements = 
		_cElementSizeMoving(0), 0.0f, 0.0f, 0.0f, 
		0.0f, _cElementSizeMoving(1), 0.0f, 0.0f, 
		0.0f, 0.0f, _cElementSizeMoving(2), 0.0f, 
		0.0f, 0.0f, 0.0f, 					1.0f;
	_cScalingFixedElements =
		1.0f / cElementSizeFixed(0), 0.0f, 0.0f, 0.0f, 
		0.0f, 1.0f / cElementSizeFixed(1), 0.0f, 0.0f, 
		0.0f, 0.0f, 1.0f / cElementSizeFixed(2), 0.0f, 
		0.0f, 0.0f, 0.0f, 						 1.0f;

	blitz::TinyMatrix<float, 4, 4> _cFinalMatrix;
	_cFinalMatrix = myproduct(_cTransformationMatrix, _cScalingMovingElements);
	_cFinalMatrix = myproduct(_cScalingFixedElements, _cFinalMatrix);

	// Mark the landmarks here
	markLandmark(cFixedImage, cLandmarksFixed, cElementSizeFixed);
	markLandmark(cMovingImage, cLandmarksMoving, _cElementSizeMoving);
	blitz::Array<blitz::TinyVector<float, 3>, 3> cRgbImage = overlay(cFixedImage, cMovingImage);

	// Write back to file
	writeBlitzToHDF5(cRgbImage, "umeyama", "firstpart.h5");
	writeBlitzTinyVectorToHDF5Attribute(cElementSizeFixed, sElement, "/umeyama", "firstpart.h5");

	// apply the transformed result of umeyama method here, just call transformArray 
	blitz::Array<float, 3> _cTargetImage(cFixedImage.shape());
	transformArray(cMovingImage, _cFinalMatrix, _cTargetImage);


	// Calculate the distance AFTER
	double _fDistanceAfter = 0.0f;
	for (int i = 0; i < cLandmarksFixed.extent(0); ++i)
	{
		blitz::TinyVector<float,4> _landmark;
		blitz::TinyVector<float,4> _temp;
		_temp = cLandmarksFixed(i)(0), cLandmarksFixed(i)(1), cLandmarksFixed(i)(2), 1.0f;
		_landmark = myproduct(_cFinalMatrix, _temp);

		_fDistanceAfter += getDistance(_landmark, _temp);
	}
  	std::cout << "Distance after calculating umeyama is: " << _fDistanceAfter << std::endl;


	blitz::Array<blitz::TinyVector<float, 3>, 3> cRgbImage2 = overlay(cFixedImage, _cTargetImage);


	// Write back to file
	writeBlitzToHDF5(cRgbImage2, "umeyama", outFileName);
	writeBlitzTinyVectorToHDF5Attribute(cElementSizeFixed, sElement, "/umeyama", outFileName);
}


/**
 * @brief
 * Simple method for testing if all methods are working as they should. This include the test cases as sugguested 
 * in the exercise sheet itself.
 * 
 * @author
 * Markus Näther
 */
void testMethod()
{
	// Test of mySVD
	{
		blitz::TinyMatrix<float, 3, 3> _A;
		_A = 1.0f, 2.0f, 3.0f, 
			 4.0f, 5.0f, 6.0f,
			 7.0f, 8.0f, 9.0f;
		blitz::TinyMatrix<float, 3, 3> _V;
		blitz::TinyVector<float, 3> _S;

		mySVD(_A, _S, _V);

		std::cout << _A << std::endl;
		std::cout << "(0.214837,0.887231,0.408248;0.520587,0.249644,-0.816497;0.826338,-0.387943,0.408248)" << std::endl;

		std::cout << _S << std::endl;
		std::cout << "(16.8481,1.06837,0.00000)" << std::endl;

		std::cout << _V << std::endl;
		std::cout << "(0.479671,-0.776691,0.408248;0.572368,-0.0756865,-0.816497;0.665064,0.625318,0.408248)" << std::endl;
	}

	// Test of umeyama implementation
	{
		blitz::Array<blitz::TinyVector<float, 4>, 1> _fixedPoints(50);
		blitz::Array<blitz::TinyVector<float, 4>, 1> _transformedPoints(50);

		for (int i = 0; i < _fixedPoints.extent(0); ++i)
		{
			_fixedPoints(i)(0) = rand()%100;
			_fixedPoints(i)(1) = rand()%100;
			_fixedPoints(i)(2) = rand()%100;
			_fixedPoints(i)(3) = 1.0f;
		}

		blitz::TinyMatrix<float, 4, 4> _T;
		_T = 1.0f, 0.0f, 0.0f, 2.0f, 
			 0.0f, std::cos(ex04_pi/3.0f), -std::sin(ex04_pi/3.0f), 3.0f, 
			 0.0f, std::sin(ex04_pi/3.0f), std::cos(ex04_pi/3.0f), 1.0f, 
			 0.0f, 0.0f, 0.0f, 1.0f;

	    for (int i = 0; i < _fixedPoints.extent(0); ++i)
		{
			_transformedPoints(i) = myproduct(_T, _fixedPoints(i));
		}

		blitz::TinyMatrix<float, 4, 4> _R = umeyama(_fixedPoints, _transformedPoints);

		std::cout << "!--- RESULTS ---!" << std::endl;
		std::cout << "! Original transformation matrix:" << std::endl;
		std::cout << _T << std::endl;
		std::cout << "! Calculated transformation matrix:" << std::endl;
		std::cout << _R << std::endl;
	}
}





/**
 * @brief
 * Main entry point of the program.
 *
 * @author
 * Markus Näther
 */
int main (int argc, char ** argv)
{
	// The real value of the second parameter is not important, but if there is one parameter we just call testMethod
	if (argc == 2)
	{
		// An additional parameter was given, so just run the tests.
		testMethod();
	}
	else 
	{
		// If no parameters was given just run the normal program.
		programMethod();
	}

	return 0;
}
