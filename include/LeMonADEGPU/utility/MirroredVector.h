#ifndef LEMONADEGPU_UTILITY_MIRROREDVECTOR_H
#define LEMONADEGPU_UTILITY_MIRROREDVECTOR_H

#include <LeMonADEGPU/utility/cudacommon.hpp>

template< class T >
class MirroredVector;

template< class T >
class MirroredTexture;


#ifdef __CUDACC__

/**
 * https://stackoverflow.com/questions/10535667/does-it-make-any-sense-to-use-inline-keyword-with-templates
 */
template< class T >
class MirroredVector
{
    #define DEBUG_MIRRORED_VECTOR 0
public:
    typedef T value_type;

    T *                host     ;
    T *                device   ;
    T *                cpu      ;
    T *                gpu      ;
    size_t       const nElements;
    size_t       const nBytes   ;
    cudaStream_t const mStream  ;
    bool         const mAsync   ;

    inline MirroredVector()
     : host( NULL ), device( NULL ), cpu( NULL ), gpu( NULL ),
       nElements( 0 ), nBytes( 0 ), mStream( 0 ), mAsync( false )
    {}

    inline void malloc()
    {
        if ( host == NULL )
        {
            #if DEBUG_MIRRORED_VECTOR > 10
                std::cerr << "[" << __FILENAME__ << "::MirroredVector::malloc]"
                    << "Allocate " << prettyPrintBytes( nBytes ) << " on host.\n";
            #endif
            host = (T*) ::malloc( nBytes );
        }
        if ( gpu == NULL )
        {
            #if DEBUG_MIRRORED_VECTOR > 10
                std::cerr << "[" << __FILENAME__ << "::MirroredVector::malloc]"
                    << "Allocate " << prettyPrintBytes( nBytes ) << " on GPU.\n";
            #endif
            CUDA_ERROR( cudaMalloc( (void**) &gpu, nBytes ) );
        }
        if ( ! ( host != NULL && gpu != NULL ) )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::MirroredVector::malloc] "
                << "Something went wrong when trying to allocate memory "
                << "(host=" << (void*) host << ", gpu=" << (void*) gpu
                << ", nBytes=" << nBytes << std::endl;
            throw std::runtime_error( msg.str() );
        }
        cpu    = host;
        device = gpu ;
    }

    inline MirroredVector
    (
        size_t const rnElements,
        cudaStream_t rStream = 0,
        bool const   rAsync  = false
    )
     : host( NULL ), device( NULL ), cpu( NULL ), gpu( NULL ),
       nElements( rnElements ), nBytes( rnElements * sizeof(T) ),
       mStream( rStream ), mAsync( rAsync )
    {
        this->malloc();
    }

    /**
     * Uses async, but not that by default the memcpy gets queued into the
     * same stream as subsequent kernel calls will, so that a synchronization
     * will be implied
     * @param[in] rAsync -1 uses the default as configured using the constructor
     *                    0 (false) synchronizes stream after memcpyAsync
     *                    1 (true ) will transfer asynchronously
     */
    inline void push( int const rAsync = -1 ) const
    {
        if ( ! ( host != NULL || gpu != NULL || nBytes == 0 ) )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::MirroredVector::push] "
                << "Can't push, need non NULL pointers and more than 0 elements. "
                << "(host=" << (void*) host << ", gpu=" << (void*) gpu
                << ", nBytes=" << nBytes << std::endl;
            throw std::runtime_error( msg.str() );
        }
        CUDA_ERROR( cudaMemcpyAsync( (void*) gpu, (void*) host, nBytes,
                                     cudaMemcpyHostToDevice, mStream ) );
        CUDA_ERROR( cudaPeekAtLastError() );
        if ( ( rAsync == -1 && ! mAsync ) || ! rAsync )
            CUDA_ERROR( cudaStreamSynchronize( mStream ) );
    }
    inline void pushAsync( void ) const { push( true ); }

    inline void pop( int const rAsync = -1 ) const
    {
        if ( ! ( host != NULL || gpu != NULL || nBytes == 0 ) )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::MirroredVector::pop] "
                << "Can't pop, need non NULL pointers and more than 0 elements. "
                << "(host=" << (void*) host << ", gpu=" << (void*) gpu
                << ", nBytes=" << nBytes << std::endl;
            throw std::runtime_error( msg.str() );
        }
        CUDA_ERROR( cudaMemcpyAsync( (void*) host, (void*) gpu, nBytes,
                                     cudaMemcpyDeviceToHost, mStream ) );
        CUDA_ERROR( cudaPeekAtLastError() );
        if ( ( rAsync == -1 && ! mAsync ) || ! rAsync )
            CUDA_ERROR( cudaStreamSynchronize( mStream ) );
    }
    inline void popAsync( void ) const { pop( true ); }

    inline void memset( uint8_t const value = 0, int const rAsync = -1 )
    {
        assert( host != NULL );
        assert( gpu  != NULL );
        CUDA_ERROR( cudaMemsetAsync( (void*) gpu, value, nBytes, mStream ) );
        std::memset( (void*) host, value, nBytes );
        if ( ( rAsync == -1 && ! mAsync ) || ! rAsync )
            CUDA_ERROR( cudaStreamSynchronize( mStream ) );
    }
    inline void memsetAsync( uint8_t const value = 0 ){ memset( value, true ); }

    inline void memcpyFrom( MirroredVector<T> const & from, int const rAsync = -1 )
    {
        auto const nMinBytes = std::min( nBytes, from.nBytes );
        //assert( nBytes == from->nBytes );
        assert( from.host != NULL );
        assert( from.gpu  != NULL );
        assert( host != NULL );
        assert( gpu  != NULL );
        CUDA_ERROR( cudaMemcpyAsync( (void*) gpu, (void*) from.gpu, nMinBytes,
                                     cudaMemcpyDeviceToDevice, mStream ) );
        std::memcpy( (void*) host, (void*) from.host, nMinBytes );
        if ( ( rAsync == -1 && ! mAsync ) || ! rAsync )
            CUDA_ERROR( cudaStreamSynchronize( mStream ) );
    }
    inline void memcpyFromAsync( MirroredVector<T> const & from ){ memcpyFrom( from, true ); }

    inline void free()
    {
        if ( host != NULL )
        {
            ::free( host );
            host = NULL;
        }
        if ( gpu != NULL )
        {
            CUDA_ERROR( cudaFree( gpu ) );
            gpu = NULL;
        }
    }

    inline ~MirroredVector()
    {
        this->free();
    }

    #undef DEBUG_MIRRORED_VECTOR
};

template< typename T >
std::ostream & operator<<( std::ostream & out, MirroredVector<T> const & x )
{
    out << "( nElements = " << x.nElements << ", "
        << "nBytes = " << x.nBytes << ","
        << "sizeof(T) = " << sizeof(T) << ","
        << "host = " << x.host << ","
        << "gpu = " << x.gpu << " )";
    return out;
}

template< class T >
class MirroredTexture : public MirroredVector<T>
{
public:
    cudaResourceDesc    mResDesc;
    cudaTextureDesc     mTexDesc;
    cudaTextureObject_t texture ;

    /**
     * @see http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html
     * @see https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
     * @see http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory
     */
    inline void bind()
    {
        memset( &mResDesc, 0, sizeof( mResDesc ) );
        /**
         * enum cudaResourceType
         *   cudaResourceTypeArray          = 0x00
         *   cudaResourceTypeMipmappedArray = 0x01
         *   cudaResourceTypeLinear         = 0x02
         *   cudaResourceTypePitch2D        = 0x03
         */
        mResDesc.resType = cudaResourceTypeLinear;
        /**
         * this might be used for interpolation ?!
         * enum cudaChannelFormatKind
         *   cudaChannelFormatKindSigned   = 0
         *   cudaChannelFormatKindUnsigned = 1
         *   cudaChannelFormatKindFloat    = 2
         *   cudaChannelFormatKindNone     = 3
         */
        mResDesc.res.linear.desc.f      = cudaChannelFormatKindUnsigned;
        mResDesc.res.linear.desc.x      = sizeof(T) * CHAR_BIT; // bits per channel
        mResDesc.res.linear.devPtr      = this->gpu;
        mResDesc.res.linear.sizeInBytes = this->nBytes;

        memset( &mTexDesc, 0, sizeof( mTexDesc ) );
        /**
         * enum cudaTextureReadMode
         *   cudaReadModeElementType     = 0
         *     Read texture as specified element type
         *   cudaReadModeNormalizedFloat = 1
         *     Read texture as normalized float
         */
        mTexDesc.readMode = cudaReadModeElementType;

        /* the last three arguments are pointers to constants! */
        cudaCreateTextureObject( &texture, &mResDesc, &mTexDesc, NULL );
    }

    inline MirroredTexture
    (
        size_t const rnElements,
        cudaStream_t rStream = 0,
        bool const   rAsync  = false
    )
     : MirroredVector<T>( rnElements, rStream, rAsync ), texture( 0 )
    {
        this->bind();
    }

    inline ~MirroredTexture()
    {
        cudaDestroyTextureObject( texture );
        texture = 0;
        this->free();
    }
};

template< typename T >
std::ostream & operator<<( std::ostream & out, MirroredTexture<T> const & x )
{
    out << "( nElements = " << x.nElements << ", "
        << "nBytes = " << x.nBytes << ","
        << "sizeof(T) = " << sizeof(T) << ","
        << "host = " << x.host << ","
        << "gpu = " << x.gpu << " )";
    return out;
}

#endif // __CUDACC__

#endif