// Copyright Markus Näther

#include <iostream>
#include <fstream> // For streams
#include <sys/stat.h> // For mkdir
#include <cfloat> // For FLT_MAX

#include <blitz/array.h>
#include <blitz/tinyvec2.h>


#ifdef RM__PI
#undef RM__PI
#endif

#define RM__PI 3.14159265359f

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
unsigned char  interpolNN ( const blitz::Array<unsigned char, 3>&  arr, blitz::TinyVector<float,4>  pos)
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
 * @brief [brief description]
 * 
 * @param _cLevelViewarr [description]
 * @param r [description]
 * 
 * @return [description]
 */
blitz::Array<unsigned char,2> orthoMips( const blitz::Array<unsigned char,3>& arr)
{
	blitz::Array<unsigned char, 2> _cLevelView(arr.extent(1), arr.extent(2)); _cLevelView = 0; // View of the levels
	blitz::Array<unsigned char, 2> _cRowView(arr.extent(0), arr.extent(2)); _cRowView = 0; // View of the rows
	blitz::Array<unsigned char, 2> _cColView(arr.extent(1), arr.extent(0)); _cColView = 0; // View of the cols
  

  // Generating the final image:
  //
  // _________________
  // |               |   /
  // |               |  /
  // |      ROWS     | /
  // |_______________|/_____
  // |               |    |
  // |               |    |
  // |               |    |
  // |               | C  |
  // |    LEVELS     | O  |
  // |               | L  |
  // |               | S  |
  // |               |    |
  // |               |    |
  // |_______________|____|
  // NOTE: I've added + 1 in each direction because there is a border between each view direction in the original image
  blitz::Array<unsigned char, 2> _cResult(arr.extent(0) + arr.extent(1) + 1, arr.extent(0) + arr.extent(2) + 1);
  _cResult = 255; // Default to black


  // Calculate the MIP for all three views here, through this we also have the advantage to iterate just once through
  // the array.
	for(int l = 0; l < arr.extent(0); ++l)
	{
		for(int r = 0; r < arr.extent(1); ++r)
		{
			for(int c = 0; c < arr.extent(2); ++c)
			{
				unsigned char _cValue = arr(l, r, c);

				if (_cValue > _cLevelView(r, c))
					_cLevelView(r, c) = _cValue;
				if (_cValue > _cRowView(l, c))
					_cRowView(l, c) = _cValue;
				if (_cValue > _cColView(r, l))
					_cColView(r, l) = _cValue;
			}
		}
	}

  // Now we just have to copy over the arrays to the image itself; start with the top view (row), go to the level view and
  // finally the cols view

  // First copy over the rows view
  _cResult(blitz::Range(0, _cRowView.extent(0) - 1), 
           blitz::Range(0, _cRowView.extent(1) - 1)) 
    = _cRowView;
  _cResult(blitz::Range(_cRowView.extent(0) + 1, _cRowView.extent(0) + _cLevelView.extent(0)), 
           blitz::Range(0, _cLevelView.extent(1) - 1)) 
    = _cLevelView;
  _cResult(blitz::Range(_cRowView.extent(0) + 1, _cRowView.extent(0) + _cColView.extent(0)), 
           blitz::Range(_cLevelView.extent(1) + 1, _cLevelView.extent(1) + _cColView.extent(1))) 
    = _cColView;


	return _cResult;
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
void  transformArray(  const blitz::Array<unsigned char, 3>&  srcArr,  const blitz::TinyMatrix<float,4,4>&   invMat,  blitz::Array<unsigned char, 3>&  trgArr)
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


/**
 * @brief
 * This method is nearly the same as in the last exercise, but with the addition that we have the params vector, where only the first 3 
 * parameters are of interest because they contain the shift in x, y and z direction.
 * This shift should occur after the scaling.
 */
blitz::TinyMatrix<float,4,4>  createInverseRigidTransMatrix( 
                                                             const blitz::TinyVector<size_t, 3>&  srcArrShape,
                                                             const blitz::TinyVector<size_t, 3>&  trgArrShape,
                                                             const blitz::TinyVector<float,3>&  src_element_size_um,
                                                             const blitz::TinyVector<float,3>&  trg_element_size_um,
                                                             blitz::TinyVector<float,6>  params  )
{
  // As mention in the execise sheet we should use seperate matrices for every step of the calculation, the last matrix is the result after concatenating all matrices
  blitz::TinyMatrix<float, 4, 4> shiftTrgCenterToOrigin, scaleToMicrometer, rotateAroundLev, rotateAroundRow, rotateAroundCol, scaleToVoxel, transform, shiftOriginToSrcCenter, _cResult;


  // Transformation of target center to the origin
  // Have to use float explicitely (don't know why exactly, because when deviding an integer by a float the result will automatically be interpreted as a float)
  shiftTrgCenterToOrigin = 1.0f, 0.0f, 0.0f, -float(trgArrShape(0))/2.0f,
                           0.0f, 1.0f, 0.0f, -float(trgArrShape(1))/2.0f,
                           0.0f, 0.0f, 1.0f, -float(trgArrShape(2))/2.0f,
                           0.0f, 0.0f, 0.0f, 1.0f;

  // First matrix can just be copied
  _cResult = shiftTrgCenterToOrigin;                         

  // Scale to micrometer
  scaleToMicrometer = trg_element_size_um(0), 0.0f, 0.0f, 0.0f,
                      0.0f, trg_element_size_um(1), 0.0f, 0.0f,
                      0.0f, 0.0f, trg_element_size_um(2), 0.0f,
                      0.0f, 0.0f, 0.0f,                   1.0f;

  // Multiply with new calculated matrix
  _cResult = myproduct(scaleToMicrometer, _cResult);


  // Apply the translation here, before any other calculations
  transform = 1.0f, 0.0f, 0.0f, -params(0),
              0.0f, 1.0f, 0.0f, -params(1),
              0.0f, 0.0f, 1.0f, -params(2),
              0.0f, 0.0f, 0.0f, 1.0f;

  // Multiply with new calculated matrix
  _cResult = myproduct(transform, _cResult);

  // Calculate the rotation around the level
  float _fSin = sin( -params(3) / 180.0f * RM__PI);
  float _fCos = cos( -params(3) / 180.0f * RM__PI);
  rotateAroundLev = 1.0f, 0.0f,   0.0f,   0.0f, 
                    0.0f, _fCos,  -_fSin, 0.0f, 
                    0.0f, _fSin,   _fCos,  0.0f, 
                    0.0f, 0.0f,   0.0f,   1.0f;

  // Multiply with new calculated matrix
  _cResult = myproduct(rotateAroundLev, _cResult);

  // Calculate the rotation around the row
  _fSin = sin( -params(4) / 180 * RM__PI);
  _fCos = cos( -params(4) / 180 * RM__PI);
  rotateAroundRow = _fCos,  0.0f, _fSin,  0.0f,
                    0.0f,   1.0f, 0.0f,   0.0f,
                    -_fSin, 0.0f, _fCos,  0.0f, 
                    0.0f,   0.0f, 0.0f,   1.0f;

  // Multiply with new calculated matrix
  _cResult = myproduct(rotateAroundRow, _cResult);

  // Calculate the rotation around the col
  _fSin = sin( -params(5) / 180 * RM__PI);
  _fCos = cos( -params(5) / 180 * RM__PI);
  rotateAroundCol = _fCos, -_fSin,  0.0f, 0.0f,
                    _fSin, _fCos,   0.0f, 0.0f,
                    0.0f, 0.0f,     1.0f, 0.0f, 
                    0.0f, 0.0f,     0.0f, 1.0f;

  // Multiply with new calculated matrix
  _cResult = myproduct(rotateAroundCol, _cResult);

  // Scale to voxel
  scaleToVoxel = 1.0f/src_element_size_um(0), 0.0f, 0.0f, 0.0f,
                 0.0f, 1.0f/src_element_size_um(1), 0.0f, 0.0f,
                 0.0f, 0.0f, 1.0f/src_element_size_um(2), 0.0f,
                 0.0f, 0.0f, 0.0f,                        1.0f;

  // Multiply with new calculated matrix
  _cResult = myproduct(scaleToVoxel, _cResult);


  // Transformation of the origin back to the center
  shiftOriginToSrcCenter = 1.0f, 0.0f, 0.0f, float(srcArrShape(0))/2.0f,
                           0.0f, 1.0f, 0.0f, float(srcArrShape(1))/2.0f,
                           0.0f, 0.0f, 1.0f, float(srcArrShape(2))/2.0f,
                           0.0f, 0.0f, 0.0f, 1.0f;

  // Multiply with new calculated matrix
  _cResult = myproduct(shiftOriginToSrcCenter, _cResult);

  // Return the resultung matrix
  return _cResult;
}




/**
 * @brief
 * This method will compute the SSD (sum of squared differences) of the fixed and the moving image ('fixedIm' and 'movingIm'), 
 * using the provided transformation matrix 'invMat'
 */
float  ssdOfFixedImAndTransformedMovingIm( const blitz::Array<unsigned char, 3>&  movingIm,
                                           const blitz::TinyMatrix<float,4,4>&  invMat,
                                           const blitz::Array<unsigned char, 3>&  fixedIm,
                                           int  step )
{
  // Make this a vector with 4 components because the matrix is a 4x4 matrix.
  blitz::TinyVector<float, 4> _tTemp, _tCurrentFixedPos;

  float _fSSD = 0.0f;

  for (int l = 0; l < fixedIm.extent(0);++l)
  {
    for (int r = 0; r < fixedIm.extent(1);++r)
    {
      for (int c = 0; c < fixedIm.extent(2);++c)
      {
	      // Assign the values and make the last component 1.
        _tCurrentFixedPos = l, r, c, 1;

	      // Apply the transformation, here we transform the fixed image points, because it should
	      // be an inverse transformation
        _tTemp = myproduct(invMat, _tCurrentFixedPos);

	      // Interpolate the nearest neighbor
        float _fValue = interpolNN(movingIm, _tTemp);

	      // And sum up the squared distance of both images
        _fSSD += blitz::pow2(fixedIm(l, r, c) - _fValue);
      }
    }
  }

  // Return the result
  return _fSSD;
}



/**
 * @brief
 * This method will compute the center of mass of an image 'inImage'. The result will be a vector of the position
 * of the center of mass in the 'normal' order (x, y, z).
 * 
 * @return
 * A vector containing the position of the center of mass as (x, y, z) vector.
 */
blitz::TinyVector<float, 3> computeCenterOfMass(const blitz::Array<unsigned char, 3> & inImage)
{
  float _Nx = 0.0f, _Ny = 0.0f, _Nz = 0.0f;
  float _Sx = 0.0f, _Sy = 0.0f, _Sz = 0.0f;

  for (int l = 0; l < inImage.extent(0); ++l)
  {
    for (int r = 0; r < inImage.extent(1); ++r)
    {
      for (int c = 0; c < inImage.extent(2); ++c)
      {
      	unsigned char _nValue = inImage(l, r, c);

      	// Weight and write to variables
      	_Nx += c * _nValue;
      	_Sx += _nValue;
      	_Ny += r * _nValue;
      	_Sy += _nValue;
      	_Nz += l * _nValue;
      	_Sz += _nValue;
      }
    }
  }

  // Normalize the result
  _Nx /= _Sx;
  _Ny /= _Sy;
  _Nz /= _Sz;

  blitz::TinyVector<float, 3> _result;
  _result(0) = _Nx;
  _result(1) = _Ny;
  _result(2) = _Nz;
  return _result;
}




/**
 * @brief 
 * This method will save the provided 'image' in a file called 'fileName'.
 * Errors are thrown if anything unexpected happens, like that the file cannot be created.
 * 
 * @param [in] image The image that should be saved.
 * @param [in] fileName The filename of the file that should be created.
 */
void savePPMImage( const blitz::Array<blitz::TinyVector<unsigned char,3>, 2>&  image , const std::string&  fileName)
{
  // Just use the code of the last exercise sheet and adapt it a bit
  std::ofstream _cFile(fileName.c_str(), std::ofstream::binary);

  // Was the file opened correctly?
  if (_cFile.is_open())
  {
    // Write everything to the file
    _cFile << "P6\n";
    // Keep in mind that the entries for width and height are flipped!
    _cFile << image.extent(1) << " " << image.extent(0) << " " << "255\n";

    // Now write the content of the image
    _cFile.write(reinterpret_cast<const char*>(image.dataFirst()), image.size() * sizeof(blitz::TinyVector<unsigned char, 3>));

    // Close the file
    _cFile.close();
  }
  else 
  {
    // The file cannot be created/opened, write this to the console
    std::cerr << "Cannot open the file '" << fileName << "'."  << std::endl;

    // Leave the application
    exit(1);
  }
}


/**
 * @brief 
 * Main method.
 */
int main(int argc, char ** argv)
{
	const int _nLevel =150;
	const int _nCols =396;
	const int _nRows =521;

  std::string _sFirstFileName, _sSecondFileName;

  if (argc == 3) // Kind of nonsense, because we should somehow retrieve the dimensions of the volume too
  {
    _sFirstFileName = argv[1];
    _sSecondFileName = argv[2];
  }
  else 
  {
    _sFirstFileName = "leaf_t5_150x521x396_8bit.raw";
    _sSecondFileName = "leaf_t6_150x521x396_8bit.raw";
  }

	// I'm assuming that the first image will be the fixed one, while the second one is the moving image
	blitz::Array<unsigned char, 3> _cFixedImage(_nLevel, _nRows, _nCols);
	blitz::Array<unsigned char, 3> _cMovingImage(_nLevel, _nRows, _nCols);
	std::ifstream _cFile(_sFirstFileName.c_str(), std::ifstream::binary);
	_cFile.read(reinterpret_cast<char*>(_cFixedImage.dataFirst()), _cFixedImage.size() * sizeof(unsigned char));
	_cFile.close();
	std::ifstream _cFile2(_sSecondFileName.c_str(), std::ifstream::binary);
	_cFile2.read(reinterpret_cast<char*>(_cMovingImage.dataFirst()), _cMovingImage.size() * sizeof(unsigned char));
	_cFile2.close();


  blitz::TinyVector<float, 3> _vConverterSize(2.0f, 1.46484f , 1.46484f);


  blitz::Array<unsigned char, 2> _debugImage1 = orthoMips(_cFixedImage);
  blitz::Array<unsigned char, 2> _debugImage2 = orthoMips(_cMovingImage);


  blitz::Array<blitz::TinyVector<unsigned char, 3>, 2> _cResultImage(_debugImage1.shape());
  _cResultImage[0] = _debugImage1;
  _cResultImage[1] = _debugImage2;
  _cResultImage[2] = 0;

  // Save the first frame in this directory
  savePPMImage(_cResultImage, "iter_000.ppm");



  // best neighbor search
  blitz::Array<blitz::TinyVector<float, 6>, 1> _vNeighbors(12);
  _vNeighbors = 0.0f;
  for (int i = 0; i < 6; ++i)
  {
    _vNeighbors(2*i)(i) 	= 1.0f;
    _vNeighbors(2*i + 1)(i) 	= -1.0f;
  }

  // Compute the parameters for the rigid transformation matrix
  // Thereby _vBestParams contains the new best parameters and _bBackupParams will hold the params of last time
  blitz::TinyVector<float, 6> _vBestParams = 0.0f, _vBackupParams = 0.0f;

  // Here the center of mass will be computed, so it can be used to make a good initial guess
  blitz::TinyVector<float, 3> _vCenterOfMassFixing = computeCenterOfMass(_cFixedImage);
  blitz::TinyVector<float, 3> _vCenterOfMassMoving = computeCenterOfMass(_cMovingImage);

  // Keep in mind that the first element is the x component in our case, but the data is in reverse order. 
  // So we have x, y, z but in the params vector it is saved as (level, row, col)
  _vBackupParams(0) = (_vCenterOfMassFixing(2) - _vCenterOfMassMoving(2)) * _vConverterSize(0);
  _vBackupParams(1) = (_vCenterOfMassFixing(1) - _vCenterOfMassMoving(1)) * _vConverterSize(1);
  _vBackupParams(2) = (_vCenterOfMassFixing(0) - _vCenterOfMassMoving(0)) * _vConverterSize(2);


  // Because we want to test with different step sizes create a directory, where we save all of them
  mkdir("stepSize", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  float s = 16.0f;
  //for (float s = 1.0f; s < 32.0f; s += 2.0f)
  {
    // Create new directory for different start stepsize
    std::ostringstream ss;
    ss << "stepSize/" << s;
    std::string _sDir(ss.str());
    mkdir(_sDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  // Also save the 0th iteration inside the directory, just for completion
  savePPMImage(_cResultImage, _sDir + "/iter_000.ppm");

  float _fStepSize = s;
  int _nIteration = 1;
  float _fBestCosts = FLT_MAX; // Initialize with max value so we can decrease it in the while loop
  float _fOldCosts;

  // Repeat the iteration while the step size is smaller than 0.125f
  while (0.125f <= _fStepSize)
  {
    blitz::Array<unsigned char, 3> _vTransformedImage(_cMovingImage.shape());
    // Helper variable which indicates whether we've found a good neighbor assignment.
    bool _bFoundBestNeighbor = false;

    printf("[Debug] Current iteration %d\n", _nIteration);

    _fOldCosts = _fBestCosts;
    // Search in all directions for best neighbor which minimizes the costs
    for (int i = 0; i < _vNeighbors.size(); ++i)
    {
      blitz::TinyVector<float, 6> _vTempParams = _fStepSize * _vNeighbors(i) + _vBackupParams;

      blitz::TinyMatrix<float, 4, 4> _mNewTransMat = createInverseRigidTransMatrix(_cMovingImage.shape(), _cFixedImage.shape(), _vConverterSize, _vConverterSize, _vTempParams);

      float _fCosts = ssdOfFixedImAndTransformedMovingIm(_cMovingImage, _mNewTransMat, _cFixedImage, 4);

      if (_fCosts < _fBestCosts)
      {
        _fBestCosts = _fCosts;
        _vBestParams = _vTempParams; // Save the parameters
        _bFoundBestNeighbor = true;
      }
    }

    // Did we found a good neighbor?
    if (_bFoundBestNeighbor)
    {
      printf("Costs: %g\n", _fBestCosts);
      printf("Diff: %g\n", _fOldCosts - _fBestCosts);
      // If we found a neighbor compute the transformation, apply it and save the iteration step to disk
      blitz::TinyMatrix<float, 4, 4> _mBestMat = createInverseRigidTransMatrix(_cMovingImage.shape(), _vTransformedImage.shape(), _vConverterSize, _vConverterSize, _vBestParams);

      transformArray(_cMovingImage, _mBestMat, _vTransformedImage);

      // Create the colored image and save it.
      blitz::Array<blitz::TinyVector<unsigned char, 3>, 2> rgbImage(_debugImage1.shape());
      rgbImage[0] = _debugImage1;
      rgbImage[1] = orthoMips(_vTransformedImage);
      rgbImage[2] = 0;


      printf("[Debug] Found new transformation, will save it now\n");
      // Save this image
      std::ostringstream _cFilename;
      _cFilename << _sDir << "/iter_" << std::setw(3) << std::setfill('0') << _nIteration << ".ppm";
      savePPMImage(rgbImage, _cFilename.str());

      _vBackupParams = _vBestParams;
    }
    else 
    {
      // No, then change the step size and repeat
      _fStepSize /= 2.0f;
    }


    // Increase iteration level
    ++_nIteration;
  }
  }


	return 0;
}
