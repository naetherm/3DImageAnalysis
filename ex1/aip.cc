/*bu
 * Comments from Benjamin Ummenhofer start with 'bu'
 */
// Markus Näther

#include <iostream>
#include <fstream>

	// static const data
static const int nLevels = 124;
static const int nRows = 216;
static const int nCols = 181;

	// The middle position
static const int nSliceLevel = 62;
static const int nSliceRow = 108;
static const int nSliceCol = 90;

// Main method
int main(int argc, char ** argv)
{

	// now, create the necessary memory which is used
  const int nNumVoxels = nLevels * nRows * nCols;
	unsigned char * pData = new unsigned char[nNumVoxels];

	// read in the file 
	std::ifstream cFile( "whatisit_124x216x181_8bit.raw", std::ifstream::binary);
  // we could simply use .read, because we have the allocated data and we know the num of elements to read
  cFile.read(reinterpret_cast<char*>(pData), nNumVoxels);


	// Calculate the average intensity projection (AIP)
	unsigned char * pAIP = new unsigned char[nRows*nCols];

	// For every row and col
	for (int row = 0; row < nRows; ++row)
	{
		for (int col = 0; col < nCols; ++col)
		{
			// Sum up the values at this position for every level
			float _value = 0.0f;
/*bu
 *
 * Hinweis:
 * Die Daten liegen in level Richtung sehr weit auseinander 
 * -> viele cache misses
 *
 * Wenn möglich sollte die level for Schleife immer ganz außen sein.
 * 
 * Vorteil deiner Lösung: man braucht nur einen float als Zwischenspeicher
 *
 *
 */
			for (int level = 0; level < nLevels; ++level)
			{
				_value += pData[level*nRows*nCols + row*nCols + col];

			}

			// Calculate the average value
			pAIP[row*nCols + col] = _value / nLevels;
		}
	}

	// Extraced the AIP, now write the result to some pgm file
	std::ofstream cFileAIP("outAIP.pgm", std::ofstream::binary);
	cFileAIP << "P5\n" << nCols << " " << nRows << " 255\n";
	cFileAIP.write(reinterpret_cast<char*>(pAIP), nRows*nCols);

	// Clean up data
	if (pAIP)
		delete[] pAIP;


	// Don't forget to free pData
	if (pData)
		delete[] pData;

	return 0;
}
