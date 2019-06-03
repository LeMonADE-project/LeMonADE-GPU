#pragma once 

/**
 * For each species this stores the
 *  - max. number of neighbors
 *  - location in memory for the species edge matrix
 *  - number of elements / monomers
 *  - pitched number of monomers / bytes to get alignment
 */
template< typename T >
struct AlignedMatrices
{
    using value_type = T;
    size_t mnBytesAlignment;
    /* bytes one row takes up ( >= nCols * sizeof(T) ) */
    struct MatrixMemoryInfo {
        size_t nRows, nCols, iOffsetBytes, nBytesPitch;
    };
    std::vector< MatrixMemoryInfo > mMatrices;

    inline AlignedMatrices( size_t rnBytesAlignment = 512u )
     : mnBytesAlignment( rnBytesAlignment )
    {
        /* this simplifies some "recursive" calculations */
        MatrixMemoryInfo m;
        m.nRows        = 0;
        m.nCols        = 0;
        m.iOffsetBytes = 0;
        m.nBytesPitch  = 0;
        mMatrices.push_back( m );
    }

    inline void newMatrix
    (
        size_t const nRows,
        size_t const nCols
    )
    {
        auto const & l = mMatrices[ mMatrices.size()-1 ];
        MatrixMemoryInfo m;
        m.nRows        = nRows;
        m.nCols        = nCols;
        m.iOffsetBytes = l.iOffsetBytes + l.nRows * l.nBytesPitch;
        m.nBytesPitch  = ceilDiv( nCols * sizeof(T), mnBytesAlignment ) * mnBytesAlignment;
        mMatrices.push_back( m );
    }

    inline size_t getMatrixOffsetBytes( size_t const iMatrix ) const
    {
        /* 1+ because of dummy 0-th element */
        return mMatrices.at( 1+iMatrix ).iOffsetBytes;
    }
    inline size_t getMatrixPitchBytes( size_t const iMatrix ) const
    {
        return mMatrices.at( 1+iMatrix ).nBytesPitch;
    }
    inline size_t getRequiredBytes( void ) const
    {
        auto const & l = mMatrices[ mMatrices.size()-1 ];
        return l.iOffsetBytes + l.nRows * l.nBytesPitch;
    }

    inline size_t bytesToElements( size_t const nBytes ) const
    {
        assert( nBytes / sizeof(T) * sizeof(T) == nBytes );
        return nBytes / sizeof(T);
    }

    inline size_t getMatrixOffsetElements( size_t const iMatrix ) const
    {
        return bytesToElements( getMatrixOffsetBytes( iMatrix ) );
    }
    inline size_t getMatrixPitchElements( size_t const iMatrix ) const
    {
        return bytesToElements( getMatrixPitchBytes( iMatrix ) );
    }
    inline size_t getRequiredElements( void ) const
    {
        return bytesToElements( getRequiredBytes() );
    }
};