// Copyright Markus Näther

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>

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
 * @brief
 * This method will take the @p srcArr, rotate it by using @p invMat and save the result of the rotation inside of @p trgArr.
 *
 * @note
 * Keep in mind that we are applying the inverse rotation, so that we can use the method presented in the lecture: 
 * 	Starting with the resulting images, loop through all voxels/pixels and determine where the source voxel/pixel was
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
 * This method will create an inverse rotation matrix. Keep in mind that internally seven matrices should be created:
 * shiftTrgCenterToOrigin, scaleToMicrometer, rotateAroundLev, rotateAroundRow, rotateAroundCol, scaleToVoxel, shiftOriginToSrcCenter.
 * 
 * At the end these matrices should be combines to one single matrix.
 * 
 * @param trg_element_size_um [description]
 * @param angleAroundLev [description]
 * @param angleAroundRow [description]
 * @param angleArounCol [description]
 * @return [description]
 */
blitz::TinyMatrix<float,4,4>  createInverseRotationMatrix(const blitz::TinyVector<size_t, 3>&  srcArrShape, const blitz::TinyVector<size_t, 3>&  trgArrShape , const blitz::TinyVector<float,3>&  src_element_size_um , const blitz::TinyVector<float,3>&  trg_element_size_um , float  angleAroundLev,  float  angleAroundRow,  float angleAroundCol)
{
  // As mention in the execise sheet we should use seperate matrices for every step of the calculation, the last matrix is the result after concatenating all matrices
  blitz::TinyMatrix<float, 4, 4> shiftTrgCenterToOrigin, scaleToMicrometer, rotateAroundLev, rotateAroundRow, rotateAroundCol, scaleToVoxel, shiftOriginToSrcCenter, _cResult;


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

  // Calculate the rotation around the level
  float _fSin = sin( -angleAroundLev / 180.0f * RM__PI);
  float _fCos = cos( -angleAroundLev / 180.0f * RM__PI);
  rotateAroundLev = 1.0f, 0.0f,   0.0f,   0.0f, 
                    0.0f, _fCos,  -_fSin, 0.0f, 
                    0.0f, _fSin,   _fCos,  0.0f, 
                    0.0f, 0.0f,   0.0f,   1.0f;

	// Multiply with new calculated matrix
  _cResult = myproduct(rotateAroundLev, _cResult);

  // Calculate the rotation around the row
  _fSin = sin( -angleAroundRow / 180 * RM__PI);
  _fCos = cos( -angleAroundRow / 180 * RM__PI);
  rotateAroundRow = _fCos,  0.0f, _fSin,  0.0f,
                    0.0f,   1.0f, 0.0f,   0.0f,
                    -_fSin, 0.0f, _fCos,  0.0f, 
                    0.0f,   0.0f, 0.0f,   1.0f;

	// Multiply with new calculated matrix
  _cResult = myproduct(rotateAroundRow, _cResult);

  // Calculate the rotation around the col
  _fSin = sin( -angleAroundCol / 180 * RM__PI);
  _fCos = cos( -angleAroundCol / 180 * RM__PI);
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
 * This method will save the provided 'image' in a file called 'fileName'.
 * Errors are thrown if anything unexpected happens, like that the file cannot be created.
 * 
 * @param [in] image The image that should be saved.
 * @param [in] fileName The filename of the file that should be created.
 */
void savePGMImage( const blitz::Array<unsigned char,2>&  image , const std::string&  fileName)
{
	// Just use the code of the last exercise sheet and adapt it a bit
  std::ofstream _cFile(fileName.c_str(), std::ofstream::binary);

	// Was the file opened correctly?
  if (_cFile.is_open())
	{
		// Write everything to the file
		_cFile << "P5\n";
		// Keep in mind that the entries for width and height are flipped!
		_cFile << image.extent(1) << " " << image.extent(0) << " " << "255\n";

    // Now write the content of the image
    _cFile.write(reinterpret_cast<const char*>(image.dataFirst()), image.size() * sizeof(char));

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
 * @def
 * Simple macro which will return the square of the provided value @p x.
 */
#define IA3D__SQR(x) x*x


/**
 * @brief
 * This method will do all the rotation around 5 degree for us. It will also save the frames correctly.
 *
 * 
 */
void rotate(blitz::Array<unsigned char, 3> &cSrcData, blitz::Array<unsigned char, 3> &cTrg, 
            blitz::TinyVector<float, 3> & cSrcElemSize, blitz::TinyVector<float, 3> & trgElementSize, 
            const std::string & cFilename)
{
  blitz::Array<unsigned char, 2> cResImage(cTrg.extent(1), cTrg.extent(2));

	// The rotation matrix which we are using for rotating the 3d voxel ata set
  blitz::TinyMatrix<float, 4, 4> _cRM;

  for (int a = 0; a < 360; a += 5)
  {
		// Create the rotation matrix
    _cRM = createInverseRotationMatrix(cSrcData.shape(), cTrg.shape(), cSrcElemSize, trgElementSize, 0, a, 0);

		// Rotate the array containing the 3d voxel image
    transformArray(cSrcData, _cRM, cTrg);

		// Set the image to zero again, so we get no artifacts
    cResImage = 0;

		// No we loop through all levels and collect the max. value for every position, this is hopefully somehow
		// described implicitely through the line:
		// 		cResImage = blitz::max(cResImage, cTrg(l, blitz::Range::all(), blitz::Range::all()));
    for (int l = 0; l < cTrg.extent(0); ++l)
    {
      cResImage = blitz::max(cResImage, cTrg(l, blitz::Range::all(), blitz::Range::all()));
    }



    // Save the image, thereby first create the image file name
    std::ostringstream _cOutFilename;

    _cOutFilename << "result/" << cFilename << "/frame" << std::setw(3) << std::setfill('0') << a << ".pgm";

    std::string _cOutFile = _cOutFilename.str();

		// And finally save the image cResImage
    savePGMImage(cResImage, _cOutFile);
  }
}



int main(int argc, char ** argv)
{
  mkdir("result", S_IRWXU | S_IRWXG);
  // Split this in two parts, as it can be seen this is for the data set "Artemisia_pollen_71x136x136_8bit.raw"
  {
    int _nLevel = 71;
    int _nRow = 136;
    int _nCol = 136;
    mkdir("result/Artemisia_pollen_71x136x136_8bit.raw", S_IRWXU | S_IRWXG);
    std::string _sFilename = "Artemisia_pollen_71x136x136_8bit.raw"; 

    // Array containing data
    blitz::Array<unsigned char,3> _cData( _nLevel, _nRow, _nCol); 

    // Open the file 
    std::ifstream _cFile(_sFilename.c_str(), std::ifstream::binary);

    // Read the file
    _cFile.read(reinterpret_cast<char*>(_cData.dataFirst()), _cData.size() * sizeof(char));

		// We don't need the file anymore, so let's close it.
		_cFile.close();

    blitz::TinyVector<float, 3> _srcElemSize(0.4f, 0.2f, 0.2f), _trgElemSize(0.4f, 0.2f, 0.2f);

    // Create the target array here. 
    //  Keep in mind that it must be larger than the source array, because we also have to add the
    //  rotated image into it!
    int _nTemp = (int)(ceil(sqrt(IA3D__SQR(_cData.extent(0)) + IA3D__SQR(_cData.extent(2)))));
    blitz::Array<unsigned char, 3> _cTrg(_nTemp, _cData.extent(1), _nTemp);

    rotate(_cData, _cTrg, _srcElemSize, _trgElemSize, _sFilename);

  }

	// Now we do the same for the dataset "Zebrafish_71x361x604_8bit.raw"
  {
    int _nLevel = 71;
    int _nRow = 361;
    int _nCol = 604;
    mkdir("result/Zebrafish_71x361x604_8bit.raw", S_IRWXU | S_IRWXG);
    std::string _sFilename = "Zebrafish_71x361x604_8bit.raw";

    // Array containing data
    blitz::Array<unsigned char,3> _cData( _nLevel, _nRow, _nCol); 

    // Open the file 
    std::ifstream _cFile(_sFilename.c_str(), std::ifstream::binary);

    // Read the file
    _cFile.read(reinterpret_cast<char*>(_cData.dataFirst()), _cData.size() * sizeof(char));

		// We don't need the file anymore, so let's close it.
		_cFile.close();

    blitz::TinyVector<float, 3> _srcElemSize(3.9f, 1.49155f, 1.49155f), _trgElemSize(3.9f, 1.49155f, 1.49155f);

    // Create the target array here. 
    //  Keep in mind that it must be larger than the source array, because we also have to add the
    //  rotated image into it!
    int _nTemp = (int)(ceil(sqrt(IA3D__SQR(_cData.extent(0)) + IA3D__SQR(_cData.extent(2)))));
    blitz::Array<unsigned char, 3> _cTrg(_nTemp, _cData.extent(1), _nTemp);


		// All necessary parameters are calculated, so we can now call the rotate method which will do all remaining calculations and savings for us.
    rotate(_cData, _cTrg, _srcElemSize, _trgElemSize, _sFilename);
  }

	// Up to this point there shouldn't be any allocated memory, so we can simply exit

  // Everything was fine, return 0
  return 0;
}
