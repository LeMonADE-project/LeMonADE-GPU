/**
 * @brief convinience class for a fast deletation of objects....
 */
struct DeleteMirroredObject
{
    size_t nBytesFreed = 0;

    template< typename S >
    void operator()( MirroredVector< S > * & p, std::string const & name = "" )
    {
        if ( p != NULL )
        {
            std::cerr
                << "Free MirroredVector " << name << " at " << (void*) p
                << " which holds " << prettyPrintBytes( p->nBytes ) << "\n";
            nBytesFreed += p->nBytes;
            delete p;
            p = NULL;
        }
    }

    template< typename S >
    void operator()( MirroredTexture< S > * & p, std::string const & name = "" )
    {
        if ( p != NULL )
        {
            nBytesFreed += p->nBytes;
            delete p;
            p = NULL;
        }
    }
};