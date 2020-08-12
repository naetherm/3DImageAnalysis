#include <iostream>
#include <fstream> // For streams
#include <sys/stat.h> // For mkdir
#include <cfloat> // For FLT_MAX
#include <cmath>
#include <cassert>
#include <vector>
#include <complex>

#define BZ_DEBUG	// Add debugging information to blitz library
#include <blitz/array.h>
#include <blitz/tinyvec2.h>

#include <gsl/gsl_matrix.h> // For gsl_matrix
#include <gsl/gsl_linalg.h> // For lincomp
#include <gsl/gsl_sf_legendre.h>

#include "./BlitzHDF5Helper.hh"

#define EX06_PI 3.14159265359


/* For compilation
 * g++ -Wall -O3 -L/usr/local/HDF_Group/HDF5/1.8.15/lib/ -I/usr/local/HDF_Group/HDF5/1.8.15/include/ -g compute_basis.cc -lblitz -lhdf5 -ldl -lgsl -lgslcblas -o compute_basis
 */

int computeFactorial(int n)
{
	static std::vector<int> SLookupTable;
	if (n <= 1)
		return 1;

	return computeFactorial(n-1) * n;
}


double computeNormalizationTerm(int l, int m)
{
	assert((l-m) >= 0);

	double _gResult = 0.0;

	double _gTemp1 = (2.0 * l + 1.0) / (4.0 * EX06_PI);
	double _gTemp2 = (computeFactorial(l - m)) / (computeFactorial(l + m));

	_gResult = std::sqrt(_gTemp1 * _gTemp2);

	return _gResult;
}

std::complex<double> computeSphereSurface(int l, int m, float fPhi, float fTheta)
{
	//blitz::Array<double, 1> _lstResult(2);
	//_lstResult = 0.0;

std::complex<double> _c(0, 1);
	//std::complex<double> _i(sin(fPhi), cos(fTheta));

/// NOTE: We don't have to calculate the normalized term because this is already done by using gsl_sf_legendre_sphPlm!

	double _gLegendrePolynomials = gsl_sf_legendre_sphPlm(l, m, cos(fTheta));

	std::complex<double> _gExponentialComponent = std::exp(_c*(double)m*(double)fTheta);

	return _gLegendrePolynomials * _gExponentialComponent;
}

int calculateDepth(int nLevel)
{
	int _nHelper = 1;
	for (int i = 1; i <= nLevel; ++i)
		_nHelper += i*2+1;

	return _nHelper;
}

void computeSphericalHarmonics(
	blitz::Array<double, 3> & lstOutResultReal, 
	blitz::Array<double, 3> & lstOutResultImag, 
	int nRows, 
	int nCols, 
	int l)
{
	// First resize the given array
	int _nHelper = calculateDepth(l);

	std::cout << "--- [START] DEBUGGING ---" << std::endl;
	std::cout << "levels: " << lstOutResultReal.extent(0) << std::endl;
	std::cout << "cols: " << lstOutResultReal.extent(1) << std::endl;
	std::cout << "rows: " << lstOutResultReal.extent(2) << std::endl;
	std::cout << "--- [ END ] DEBUGGING ---" << std::endl;


	// For every level calculate the sphere surface
	//int _nCounter = 0;
	for (int i = 0; i <= l; ++i)
	{
		for (int m = 0; m <= i; ++m)
		{
			std::cout << "l: " << l << " , counter: " << i*(i+1) << std::endl;

			if (m == 0)
			{
			int _nIntermediate = i*(i+1);
				blitz::Array<double, 2> _cTempR = lstOutResultReal(_nIntermediate, blitz::Range(0, nCols-1), blitz::Range(0, nRows-1));
				blitz::Array<double, 2> _cTempT = lstOutResultImag(_nIntermediate, blitz::Range(0, nCols-1), blitz::Range(0, nRows-1));
				//_cTemp = 0.0f;
				
				for (int r = 0; r < nRows; ++r)
				{
					for (int c = 0; c < nCols; ++c)
					{
						//_cTemp(c, r) = 0.0f;
						
					//	{
							
							
							std::complex<double> _lstTemp = computeSphereSurface(i, m, c, r);

						//	std::cout << " " << _lstTemp(0) << " " << _lstTemp(1) << " ";
							
							_cTempR(c, r) = _lstTemp.real();
							_cTempT(c, r) = _lstTemp.imag();
							//lstOutResult[i]() = _lstTemp(1);

							if (m != 0)
							{
								//lstOutResult(i)(i*(i+1)-m) = _lstTemp(1);
							}
						
					}
				}

				lstOutResultReal(_nIntermediate, blitz::Range(0, nCols-1), blitz::Range(0, nRows-1)) = _cTempR;
				lstOutResultImag(_nIntermediate, blitz::Range(0, nCols-1), blitz::Range(0, nRows-1)) = _cTempT;
			}
			else 
			{
			int _nIntermediate = i*(i+1);
				blitz::Array<double, 2> _cTempR = lstOutResultReal(_nIntermediate+m, blitz::Range(0, nCols-1), blitz::Range(0, nRows-1));
				blitz::Array<double, 2> _cTempT = lstOutResultImag(_nIntermediate+m, blitz::Range(0, nCols-1), blitz::Range(0, nRows-1));
				blitz::Array<double, 2> _cTempRM = lstOutResultReal(_nIntermediate-m, blitz::Range(0, nCols-1), blitz::Range(0, nRows-1));
				blitz::Array<double, 2> _cTempTM = lstOutResultImag(_nIntermediate-m, blitz::Range(0, nCols-1), blitz::Range(0, nRows-1));
				//_cTemp = 0.0f;
				
				for (int r = 0; r < nRows; ++r)
				{
					for (int c = 0; c < nCols; ++c)
					{
						//_cTemp(c, r) = 0.0f;
						
					//	{
							
							
							std::complex<double> _lstTemp = computeSphereSurface(i, m, c, r);

						//	std::cout << " " << _lstTemp(0) << " " << _lstTemp(1) << " ";
							
							_cTempR(c, r) = _lstTemp.real();
							_cTempT(c, r) = _lstTemp.imag();
							_cTempRM(c, r) = std::pow(-1, m)*_lstTemp.real();
							_cTempTM(c, r) = std::pow(-1, m)*_lstTemp.imag();
							//lstOutResult[i]() = _lstTemp(1);

							if (m != 0)
							{
								//lstOutResult(i)(i*(i+1)-m) = _lstTemp(1);
							}
						
					}
				}

				lstOutResultReal(_nIntermediate+m, blitz::Range(0, nCols-1), blitz::Range(0, nRows-1)) = _cTempR;
				lstOutResultImag(_nIntermediate+m, blitz::Range(0, nCols-1), blitz::Range(0, nRows-1)) = _cTempT;
				lstOutResultReal(_nIntermediate-m, blitz::Range(0, nCols-1), blitz::Range(0, nRows-1)) = _cTempRM;
				lstOutResultImag(_nIntermediate-m, blitz::Range(0, nCols-1), blitz::Range(0, nRows-1)) = _cTempTM;
			}
		
		}
	}
}


int main(int argc, char ** argv)
{
	const int _nCols = 360;
	const int _nRows = 180;
	const int _nLevel = calculateDepth(30);
	std::cout << "Depth is: " << _nLevel << std::endl;
	std::string _sOutFilename = "out.h5";
	std::string _sLocation = "image";

	blitz::Array<double, 3> _lstSphericalHarmonicsReal(_nLevel, _nRows, _nCols);
	blitz::Array<double, 3> _lstSphericalHarmonicsImag(_nLevel, _nRows, _nCols);
	_lstSphericalHarmonicsReal = 0.0;
	_lstSphericalHarmonicsImag = 0.0;

	computeSphericalHarmonics(_lstSphericalHarmonicsReal, _lstSphericalHarmonicsImag, _nCols, _nRows, 30);


	// Now save the results
	writeBlitzToHDF5(_lstSphericalHarmonicsReal, "basis_real", _sOutFilename);
	writeBlitzToHDF5(_lstSphericalHarmonicsImag, "basis_imag", _sOutFilename);

	return 0;
}
