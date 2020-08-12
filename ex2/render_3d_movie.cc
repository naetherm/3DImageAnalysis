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

  for( int l = 0; l < trgArr.extent(0); ++l)
  {
    for( int r = 0; r < trgArr.extent(1); ++r)
    {
      for( int c = 0; c < trgArr.extent(2); ++c)
      {
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
  blitz::TinyMatrix<float, 4, 4> shiftTrgCenterToOrigin, scaleToMicrometer, rotateAroundLev, rotateAroundRow, rotateAroundCol, scaleToVoxel, shiftOriginToSrcCenter, _cResult;


  // Transformation of target center to the origin
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
                    -_fSin, 0.0f, _fCos, 0.0f, 
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
 * @def
 * Simple macro which will return the square of the provided value @p x.
 */
#define IA3D__SQR(x) x*x


int main(int argc, char ** argv)
{

    mkdir("result", S_IRWXU | S_IRWXG);
    mkdir("result/3d", S_IRWXU | S_IRWXG);
  // Split this in two parts
  {
    int _nLevel = 71;
    int _nRow = 136;
    int _nCol = 136;
    mkdir("result/3d/Artemisia_pollen_71x136x136_8bit.raw", S_IRWXU | S_IRWXG);
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



    // Create the anaglyph image here!

    // The angle as noticed in figure 1
    float _fEyeStereoAngle = 5.0f;



    blitz::TinyMatrix<float, 4, 4> _cLeftViewRotCol, _cRightViewRotCol;
    // We need a rotation matrix around the col for each view

    int _nTempRow = (int)(ceil(sqrt(IA3D__SQR(_cData.extent(0)) + IA3D__SQR(_cData.extent(1)))));
    blitz::Array<unsigned char, 3> _cResult (std::max(_nTemp, _nTempRow), _nTempRow, _nTemp);
    blitz::Array<unsigned char, 2> _cLeftViewExt(_cResult.extent(1), _cResult.extent(2));
    blitz::Array<unsigned char, 2> _cRightViewExt(_cResult.extent(1), _cResult.extent(2));


    const int _nTrgRows = 136;
    const int _nTrgCols = 136;

    blitz::Array<blitz::TinyVector<unsigned char, 3>, 2> _cOverUnderFormattedImage (2 * _nTrgRows, _nTrgCols);


    // Here we receive the center position of target and source images and the minimum number of cols/rows for both images
    //  Through this we then can easily fetch the max area that could be written to the resulting frame. In other words:
    //  We are looking for the maximum area within the source image that will fit into the target image
    const int _nSrcRows = _cResult.extent(1);
    const int _nSrcCols = _cResult.extent(2);

    blitz::TinyVector<int, 2> _lstTrgCenter(_nTrgRows / 2.0f, _nTrgCols / 2.0f);
    blitz::TinyVector<int, 2> _lstSrcCenter(_nSrcRows / 2.0f, _nSrcCols / 2.0f);
    int _nMinRows = std::min(_nSrcRows, _nTrgRows);
    int _nMinCols = std::min(_nSrcCols, _nTrgCols);

    // Copied these values into TinyVectors, before that these were single integers, ... way to confusing
    // The values for the target image, the parameters have the following meaning:
    // First and Third  : The upper left corner of the min area that fits into both (source and target)
    // Second and Fourth: The bottom right corner of the min area that fits into both (source and target)
    blitz::TinyVector<int,4> _lsttrgRange(
      std::max(_lstTrgCenter(0)-_nMinRows/2, 0), // upper y border
      std::min(_lstTrgCenter(0)-_nMinRows/2+_nMinRows-1, _nTrgRows-1), // bottom y border
      std::max(_lstTrgCenter(1)-_nMinCols/2, 0),  // left x border
      std::min(_lstTrgCenter(1)-_nMinCols/2+_nMinCols-1, _nTrgCols-1) // right x border
    );
        
    // and the same for the source image
    blitz::TinyVector<int,4> srcRange(
      std::max(_lstSrcCenter(0)-_nMinRows/2, 0), // upper y border
      std::min(_lstSrcCenter(0)-_nMinRows/2+_nMinRows-1, _nSrcRows-1), // bottom y border
      std::max(_lstSrcCenter(1)-_nMinCols/2, 0), // left x border
      std::min(_lstSrcCenter(1)-_nMinCols/2+_nMinCols-1, _nSrcCols-1) // right x border
    );

    {
      // for all angles
      for (int a = 0; a < 360; a+=5)
      {
        _cLeftViewRotCol = createInverseRotationMatrix(_cData.shape(), _cResult.shape(), _srcElemSize, _trgElemSize, 0, _fEyeStereoAngle/2, a);
        transformArray(_cData, _cLeftViewRotCol, _cResult);

        _cRightViewExt = 0;
        for (int l = 0; l < _cResult.extent(0); ++l)
        {
          _cRightViewExt = blitz::max(_cRightViewExt, _cResult(l, blitz::Range::all(), blitz::Range::all()));
        }



        _cRightViewRotCol = createInverseRotationMatrix(_cData.shape(), _cResult.shape(), _srcElemSize, _trgElemSize, 0, -_fEyeStereoAngle/2, a);
        transformArray(_cData, _cRightViewRotCol, _cResult);

        _cLeftViewExt = 0;
        for (int l = 0; l < _cResult.extent(0); ++l)
        {
          _cLeftViewExt = blitz::max(_cLeftViewExt, _cResult(l, blitz::Range::all(), blitz::Range::all()));
        }




        blitz::Array<unsigned char, 2> _lstTrgLeftView(_nTrgRows, _nTrgCols);
        blitz::Array<unsigned char, 2> _lstTrgRightView(_nTrgRows, _nTrgCols);
        _lstTrgLeftView = 0; _lstTrgRightView = 0;
        
        // create over/under image    
        _lstTrgLeftView(blitz::Range(_lsttrgRange(0),_lsttrgRange(1)), blitz::Range(_lsttrgRange(2),_lsttrgRange(3))) = 
            _cLeftViewExt(blitz::Range(srcRange(0),srcRange(1)), blitz::Range(srcRange(2),srcRange(3)));
        
        _lstTrgRightView(blitz::Range(_lsttrgRange(0),_lsttrgRange(1)), blitz::Range(_lsttrgRange(2),_lsttrgRange(3))) =
            _cRightViewExt(blitz::Range(srcRange(0),srcRange(1)), blitz::Range(srcRange(2),srcRange(3)));
            
        // over image (left view), keep in mind that the range for the row just goes to _nTrgRows-1!
        _cOverUnderFormattedImage[0](blitz::Range(0,_nTrgRows-1), blitz::Range::all()) = _lstTrgLeftView;
        _cOverUnderFormattedImage[1](blitz::Range(0,_nTrgRows-1), blitz::Range::all()) = _lstTrgLeftView;
        _cOverUnderFormattedImage[2](blitz::Range(0,_nTrgRows-1), blitz::Range::all()) = _lstTrgLeftView;
        // under image (right view), keep in mind that the range starts at _nTrgRows!
        _cOverUnderFormattedImage[0](blitz::Range(_nTrgRows,2*_nTrgRows-1), blitz::Range::all()) = _lstTrgRightView;
        _cOverUnderFormattedImage[1](blitz::Range(_nTrgRows,2*_nTrgRows-1), blitz::Range::all()) = _lstTrgRightView;
        _cOverUnderFormattedImage[2](blitz::Range(_nTrgRows,2*_nTrgRows-1), blitz::Range::all()) = _lstTrgRightView;


        std::ostringstream _os;
        _os << "result/3d/"  << _sFilename << "/movie_3d_frame_" << std::setw(3) << std::setfill('0') << a << ".ppm";
        std::string fileName = _os.str();
        std::cout << "Processed image '" << fileName << "' and saved to disk." << std::endl;
        
        savePPMImage( _cOverUnderFormattedImage, fileName); 
      }
    }
  }



  {
    int _nLevel = 71;
    int _nRow = 361;
    int _nCol = 604;
    mkdir("result/3d/Zebrafish_71x361x604_8bit.raw", S_IRWXU | S_IRWXG);
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



    // Create the anaglyph image here!

    // The angle as noticed in figure 1
    float _fEyeStereoAngle = 5.0f;

    blitz::TinyMatrix<float, 4, 4> _cLeftViewRotCol, _cRightViewRotCol;
    // We need a rotation matrix around the col for each view

    int _nTempRow = (int)(ceil(sqrt(IA3D__SQR(_cData.extent(0)) + IA3D__SQR(_cData.extent(1)))));
    blitz::Array<unsigned char, 3> _cResult (std::max(_nTemp, _nTempRow), _nTempRow, _nTemp);
    blitz::Array<unsigned char, 2> _cLeftViewExt(_cResult.extent(1), _cResult.extent(2));
    blitz::Array<unsigned char, 2> _cRightViewExt(_cResult.extent(1), _cResult.extent(2));


    const int _nTrgRows = 360;
    const int _nTrgCols = 640;

    blitz::Array<blitz::TinyVector<unsigned char, 3>, 2> _cOverUnderFormattedImage (2 * _nTrgRows, _nTrgCols);


    // Here we receive the center position of target and source images and the minimum number of cols/rows for both images
    //  Through this we then can easily fetch the max area that could be written to the resulting frame. In other words:
    //  We are looking for the maximum area within the source image that will fit into the target image
    const int _nSrcRows = _cResult.extent(1);
    const int _nSrcCols = _cResult.extent(2);

    blitz::TinyVector<int, 2> _lstTrgCenter(_nTrgRows / 2.0f, _nTrgCols / 2.0f);
    blitz::TinyVector<int, 2> _lstSrcCenter(_nSrcRows / 2.0f, _nSrcCols / 2.0f);
    int _nMinRows = std::min(_nSrcRows, _nTrgRows);
    int _nMinCols = std::min(_nSrcCols, _nTrgCols);

    // Copied these values into TinyVectors, before that these were single integers, ... way to confusing
    // The values for the target image, the parameters have the following meaning:
    // First and Third  : The upper left corner of the min area that fits into both (source and target)
    // Second and Fourth: The bottom right corner of the min area that fits into both (source and target)
    blitz::TinyVector<int,4> _lsttrgRange(
      std::max(_lstTrgCenter(0)-_nMinRows/2, 0), // upper y border
      std::min(_lstTrgCenter(0)-_nMinRows/2+_nMinRows-1, _nTrgRows-1), // bottom y border
      std::max(_lstTrgCenter(1)-_nMinCols/2, 0),  // left x border
      std::min(_lstTrgCenter(1)-_nMinCols/2+_nMinCols-1, _nTrgCols-1) // right x border
    );
        
    // and the same for the source image
    blitz::TinyVector<int,4> srcRange(
      std::max(_lstSrcCenter(0)-_nMinRows/2, 0), // upper y border
      std::min(_lstSrcCenter(0)-_nMinRows/2+_nMinRows-1, _nSrcRows-1), // bottom y border
      std::max(_lstSrcCenter(1)-_nMinCols/2, 0), // left x border
      std::min(_lstSrcCenter(1)-_nMinCols/2+_nMinCols-1, _nSrcCols-1) // right x border
    );

    {
      // for all angles
      for (int a = 0; a < 360; a+=5)
      {
        _cLeftViewRotCol = createInverseRotationMatrix(_cData.shape(), _cResult.shape(), _srcElemSize, _trgElemSize, 0, _fEyeStereoAngle/2, a);
        transformArray(_cData, _cLeftViewRotCol, _cResult);

        _cRightViewExt = 0;
        for (int l = 0; l < _cResult.extent(0); ++l)
        {
          _cRightViewExt = blitz::max(_cRightViewExt, _cResult(l, blitz::Range::all(), blitz::Range::all()));
        }



        _cRightViewRotCol = createInverseRotationMatrix(_cData.shape(), _cResult.shape(), _srcElemSize, _trgElemSize, 0, -_fEyeStereoAngle/2, a);
        transformArray(_cData, _cRightViewRotCol, _cResult);

        _cLeftViewExt = 0;
        for (int l = 0; l < _cResult.extent(0); ++l)
        {
          _cLeftViewExt = blitz::max(_cLeftViewExt, _cResult(l, blitz::Range::all(), blitz::Range::all()));
        }




        blitz::Array<unsigned char, 2> _lstTrgLeftView(_nTrgRows, _nTrgCols);
        blitz::Array<unsigned char, 2> _lstTrgRightView(_nTrgRows, _nTrgCols);
        _lstTrgLeftView = 0; _lstTrgRightView = 0;
        
        // create over/under image    
        _lstTrgLeftView(blitz::Range(_lsttrgRange(0),_lsttrgRange(1)), blitz::Range(_lsttrgRange(2),_lsttrgRange(3))) = 
            _cLeftViewExt(blitz::Range(srcRange(0),srcRange(1)), blitz::Range(srcRange(2),srcRange(3)));
        
        _lstTrgRightView(blitz::Range(_lsttrgRange(0),_lsttrgRange(1)), blitz::Range(_lsttrgRange(2),_lsttrgRange(3))) =
            _cRightViewExt(blitz::Range(srcRange(0),srcRange(1)), blitz::Range(srcRange(2),srcRange(3)));
            
        // over image (left view)
        _cOverUnderFormattedImage[0](blitz::Range(0,_nTrgRows-1), blitz::Range::all()) = _lstTrgLeftView;
        _cOverUnderFormattedImage[1](blitz::Range(0,_nTrgRows-1), blitz::Range::all()) = _lstTrgLeftView;
        _cOverUnderFormattedImage[2](blitz::Range(0,_nTrgRows-1), blitz::Range::all()) = _lstTrgLeftView;
        // under image (right view)
        _cOverUnderFormattedImage[0](blitz::Range(_nTrgRows,2*_nTrgRows-1), blitz::Range::all()) = _lstTrgRightView;
        _cOverUnderFormattedImage[1](blitz::Range(_nTrgRows,2*_nTrgRows-1), blitz::Range::all()) = _lstTrgRightView;
        _cOverUnderFormattedImage[2](blitz::Range(_nTrgRows,2*_nTrgRows-1), blitz::Range::all()) = _lstTrgRightView;




        std::ostringstream _os;
        _os << "result/3d/"  << _sFilename << "/movie_3d_frame_" << std::setw(3) << std::setfill('0') << a << ".ppm";
        std::string fileName = _os.str();
        std::cout << "Processed image '" << fileName << "' and saved to disk." << std::endl;
        
        savePPMImage( _cOverUnderFormattedImage, fileName); 
      }
    }
  }

  // Everything was fine, return 0
  return 0;
}

