/*
 - n = 108 is the number of allowed bonds
 - naive implementation: try out all 2^108 combinations
 - optimization 1: instead of iterating over ALL combinations, iterate only
   over the next bond if it is indeed allowed by volume exclusion.
    => Will result in <=2^m combinations with m~20 = maximum neighbors
 - implementation: Count up a vector of length <= m with it's elements bounded
   by 108,107, ... or smaller if already end. Increment first from last, i.e.
   for (0,0,0,0) always the next allowed bond with the smallest index will
   be used! => seems easy to implement and easy enough it could work
 - optimization 2: create lookup table for volume exclusion which stores
   which bonds j become forbidden when adding bond i:
   i\j 0 1 2 3  => bond 0 forbids 0,2,3, bond 1 only forbids itself
    0  1 0 1 1  this table is symmetrical, suffices to store one half i<j and
    1  0 1 0 0  the diagonal obviousky also is always 1
    2  1 0 1 0
    3  1 0 0 1
 - estimation upper bound: Max bond length in one dimension is 3
   => upper bound for volume is 7^3 = 343.
   The densest filling with valid volume exclusion in 1D is 50%, in 2D it's
   25%, in 3D its 12.5% (1 in 8)
   => maximum bound by volume exclusion is 343/8 = 42.87 < 108. Knowing that
   one is at the center we could try to be more exact, but that's too complex
 - idea: high-dimensional probing => Monte-Carlo based findMaximum method
     => works wonderfully well :3
 - optimization 3: Order does not matter, so require at each step, that bondIDs
   are sorted / increasing! With 20 neighbors maximum this would give an upper
   bound of operations to do as 108!/[(108-20)!20!] = choose(108,20) = 2.9e21
   The observer for the version with 1/20! missing already seemed much faster.
   A more realistic estimate would be to change the 108!/88! = 108!*...89! such
   that the last is not 89!, but 1! and the rest is evenly spaced, i.e.
   108/20=5.4 ~> 108!103!*...*1! <- 20 factors =>
       import math, numpy as np
       np.product( np.arange(1,108,5.4) ) / math.factorial(20)
     => 7 742 713 855 154.2754 = 7.7e12, should be very well doable!
 - optimization 4: Because of how the bond vectors were created from permutation
   sets, the problem is highly symmetrical which could be used to reduce problem
   size, then again a asymmetrical solution might be the best, or can't it?
 - optimization 5: sort bonds by amount of other bonds they forbid, so that the
   "best" bonds are chosen first! This already could be thought of naively as
   THE solution, i.e. following the local maximum to get the global one, but
   that shouldn't be right.
 - optimization 6: increment current index element, instead of all lower ones,
   if nAllowedBondsRemaining + nBondsUsed <= nMaxDegreeAlreadyFound. This is
   better the sooner a very good nMaxDegreeAlreadyFound is found and with
   optimization 5 the currently highest known neighbors (20) are found in the
   very first step.
*/

/*
g++ -Wall -Wextra -O3 -fopenmp -I ../extern/ -std=c++11 ../src/findMaxNeighbors.cpp && ./a.out
*/

#include <algorithm>                        // find, sort, next_permutation
#include <array>
#include <vector>

/**
 * E.g. input (2,1,0) gets first permutated to all possible configurations:
 *   (0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)
 * and then also all combinations fo signs will be created:
 *   (0,1,2), (0,-1,2), (0,-1,-2), (0,1,-2), (0,2,1), ...
 * So for one n-dimensional array n! * 2^n permutations will be returned in
 * the limiting case, but as only unique vectors will be returned the effective
 * length of the return vector will be less than that.
 *
 * @param[in] perms list of the raw unpermutated vectors.
 * @return all permutations for each vector in the given list
 */
template < int T_Dim = 3, typename T >
inline
std::vector< std::array< T, T_Dim > >
getSignedPermutations
(
    std::vector< std::array< T, T_Dim > > const & perms
)
{
    std::vector< std::array< T, T_Dim > > result;
    for ( auto perm : perms )
    {
        /**
         * need to get all n! permutations. Wow C++-STL actually has this
         * and the implementation is really cool to see:
         * http://wordaligned.org/articles/next-permutation
         * In order to get ALL permutations we could sort the first input,
         * because next_permutation returns false when the input is reverse
         * sorted, or we could compare each current permutation with the input
         * and break if they are equal, but that should be slower.
         * Additionally next_permutation only returns only unique permutations,
         * so we don't even need to check if we alreday have it,
         * it's just too cool.
         */
        std::sort( perm.begin(), perm.end() );
        if ( perm.begin() != perm.end() ) do
        {
            /**
             * @todo implement something similar to next_permutation, which
             * basically would do the same as what is being done to the
             * bit representation when adding 1. E.g. 1011 -> 1100 is
             * equivalent to: (-3,+2,-1,-0) -> (-3,-2,+1,+0), but that would
             * in average have to touch n/2 elements vs. the full apply which
             * needs to touch n elements. So there is no complexity decrease
             */
            /* count zeros in order to not loop over them */
            size_t nZeros = 0;
            for ( auto const & x : perm )
                nZeros += x == 0;
            /* this is correct for the case of only having zeros! */
            size_t const nSignPermutations = size_t(1) << ( perm.size() - nZeros );
            for ( size_t iSignPerm = 0u; iSignPerm < nSignPermutations; ++iSignPerm )
            {
                /* apply sign array 1 is - and 0 is +, plus we want the right
                 * most element to be changed first, i.e. 0b001 should be
                 * (1,2,-3) */
                auto x = perm;
                auto signs = iSignPerm;
                auto i = x.end();
                while ( i != x.begin() )
                {
                    --i;
                    if ( *i == 0 )
                        continue;
                    *i = std::abs( *i );
                    if ( signs & 1  )
                        *i = -*i;
                    signs >>= 1;
                }
                /* ADD TO RESULT. THIS WILL BE UNIQUE AS LONG AS INPUT IS UNIQUE */
                result.push_back( x );
            }
        }
        while( std::next_permutation( perm.begin(), perm.end() ) );
    }
    return result;
}


#include <cmath>                                    // pow
#include <iostream>
#include <map>
#include <extern/Fundamental/Fundamental.hpp> // factorial
#include <extern/Fundamental/toString.hpp>           // operator<< overloads
template < int T_Dim = 3, typename T = int >
inline
bool testGetSignedPermutations( std::array< T, T_Dim > const & perm = { 0,1,2 } )
{
    auto const perms = getSignedPermutations( { perm } );
    std::cout << perms << std::endl;

    /* find doubles and zeros in order to calculate how large result should be */
    std::map< int, size_t > freqs;
    for ( auto const & x : perm )
        freqs[x]++;
    size_t nPermutationsTheoretical = factorial( perm.size() );
    for ( auto const & x : freqs )
        nPermutationsTheoretical /= (size_t) factorial( x.second );
    nPermutationsTheoretical *= std::pow( 2, perm.size() - freqs[0] );

    std::cout
        << "We expected " << nPermutationsTheoretical << " permutations "
        << "and got " << perms.size() << std::endl;

    return nPermutationsTheoretical == perms.size();
}

template < int T_Dim = 3, typename T_BondInt = short int >
inline
std::vector< std::vector< bool > >
createExclusionTable
(
    std::vector< std::array< T_BondInt, T_Dim > > const & bonds,
    unsigned int const cubeHalfSize = 1
)
{
    std::vector< std::vector< bool > > result(
        bonds.size(), std::vector< bool >( bonds.size(), true )
    );
    for ( size_t i = 0u; i < result.size(); ++i )
    for ( size_t j = 0u; j < i; ++j )
    for ( size_t k = 0u; k < T_Dim; ++k )
    if ( std::abs( bonds[i][k] - bonds[j][k] ) > cubeHalfSize )
    {
        result[i][j] = false;
        break;
    }

    /* fill in symmetry and known diagonal */
    for ( size_t i = 0u; i < result.size(); ++i )
    {
        result[i][i] = 1;
        for ( size_t j = i+1; j < result[i].size(); ++j )
            result[i][j] = result[j][i];
    }

    return result;
}

#include <cassert>
#include <limits>                       // numeric_limits

/**
 * @param[in] excludes a 2D matrix storing in excludes[i][j] whether i
 *                     forbids bond j because of volume exclusion / collision
 * @return a compressed vector of lists storing the excluded bonds' IDs. The
 *         size of the list is fixed and can be calculated from result.size() /
 *         excludes.size(). Not full lists are padded with the maximum value
 *         the assumedly unsigned data type can hold e.g. USHRT_MAX.
 *         Selfexclusion is not stored because it is trivial that
 *         excludes[i][i] = true.
 *         The list size is chosen to be a power of two or a multiple of 64 Byte
 *         as this is the cache line size for Intel Broadwell among others
 *         @see http://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf
 *         Query cache line with: grep . /sys/devices/system/cpu/cpu[0-9]/cache/index[0-9]/coherency_line_size
 */
template < int T_Dim = 3, typename T_BondId = short int >
inline std::vector< T_BondId > compressExclusionTable
(
    std::vector< std::vector< bool > > const & excludes
)
{
    /* count max. amount of other bonds a chosen bond i forbids */
    unsigned int nMaxForbidden = 0;
    for ( auto i = 0u; i < excludes.size(); ++i )
    {
        unsigned int nForbidden = 0u;
        for ( auto j = 0u; j < excludes.size(); ++j )
            nForbidden += excludes[i][j] /* && i != j*/;
        nMaxForbidden = std::max( nMaxForbidden, nForbidden );
    }

    /* find a "nice" padding length */
    nMaxForbidden += 1; // by enforcing one more element, it suffices to just check for SHRT_MAX, we don't have to check for nPadding in adition to that when using this compressed able
    unsigned int constexpr nCacheLine = 64u;
    unsigned int nPadding = 1;
    while ( nPadding < nMaxForbidden )
        nPadding <<= 1;
    if ( nPadding > nCacheLine )
        nPadding = nCacheLine * ( ( nMaxForbidden + nCacheLine - 1 ) / nCacheLine ); // (x+m-1)/m = ceil(x/m)
    assert( nPadding >= nMaxForbidden );

    /* compresses 'excludes'-matrix into:
     *  34  12   2 256
     *   8   7 256 256
     * 256 256 256 256
     *  1    5   3 256
     *  ...
     */
    std::vector< T_BondId > result( excludes.size() * nPadding, std::numeric_limits< T_BondId >::max() );
    for ( size_t i = 0u; i < excludes.size(); ++i )
    {
        unsigned int nForbidden = 0u;
        for ( size_t j = 0u; j < excludes.size(); ++j )
        {
            if ( excludes[i][j] /* && i != j */ )
                result[ i * nPadding + nForbidden++ ] = j;
        }
    }
    return result;
}

template<
    int      T_Dim     = 3,
    typename T_BondInt = short int,
    typename T_BondId  = unsigned short int
>
inline
std::vector< T_BondId > getNeighborConfiguration
(
    std::vector< T_BondId >            const & iThAllowed,
    std::vector< std::vector< bool > > const & excludes
)
{
    /**
     * Stores bond IDs, i.e. index in 'bonds', of all neighbors
     * for the current configuration.
     */
    std::vector< T_BondId > chosenBonds;
    /**
     * For the current configuration given in chosenBonds it stores how
     * often each other bond is prohibited by an already existing bond.
     * 0 means we can still add it. 3 means we have to remove 3 bonds
     * which forbid this bond before we can add it.
     */
    std::vector< T_BondId > prohibitedBonds( excludes.size(), 0 );

    /* initialize table for 1st configuration, so that we can only update it
     * in a differential fashion */
    for ( size_t i = 0u; i < iThAllowed.size(); ++i )
    {
        /* find first index[i]-th allowed bond (special case: find first allowed) */
        int iChosen = -1;
        for ( size_t j = 0u, nAllowed = 0u; j < prohibitedBonds.size(); ++j )
        {
            nAllowed += prohibitedBonds[j] == 0;
            if ( nAllowed > iThAllowed[i] )
            {
                iChosen = j;
                break;
            }
        }
        if ( iChosen < 0 )
        {
            /* finished, there is no further allowed bond anymore! */
            break;
        }
        /* apply the chosen bond */
        chosenBonds.push_back( iChosen );
        for ( size_t j = 0u; j < excludes[ iChosen ].size(); ++j )
            prohibitedBonds[j] += excludes[ iChosen ][j];
    }
    /*
    std::cout << "chosenBonds: "     << chosenBonds     << "\n";
    std::cout << "prohibitedBonds: " << prohibitedBonds << "\n";
    std::cout << "number of bonds / neighbors: " << chosenBonds.size() << "\n";
    */
    return chosenBonds;
}

#include <cstring>                          // std:memset

template<
    int      T_Dim     = 3,
    typename T_BondInt = short int,
    typename T_BondId  = unsigned short int
>
inline
std::vector< T_BondId > findMaxNeighborsNaive
(
    std::vector< std::array< T_BondInt, T_Dim > > const & bonds,
    unsigned int const cubeHalfSize = 1
)
{
    auto const excludes = createExclusionTable< T_Dim, T_BondInt >( bonds, cubeHalfSize );
    /**
     * stores which n-th allowed bond we should choose. E.g. (0,1,2)
     * chooses the first bond allowed, then the second bond of still
     * allowed bonds and then the third bond which is still allowed after
     * adding the prior two bonds to chosenBonds.
     */
    std::vector< T_BondId > index( bonds.size(), 0 );
    std::vector< T_BondId > maxConfig, chosenBonds;
    chosenBonds = getNeighborConfiguration< T_Dim, T_BondInt, T_BondId >( index, excludes );
    maxConfig = chosenBonds;

    std::cout << "chosenBonds: "     << chosenBonds     << "\n";
    std::cout << "number of bonds / neighbors: " << chosenBonds.size() << "\n";

    T_BondId nNeighborsLastStep = chosenBonds.size();
    size_t nTested = 0;
    auto const t0 = now();
    while ( chosenBonds.size() > 0 )
    {
        /* increment index noting that the length is variable ... */
        ++index[ chosenBonds.size() - 1 ];
        for ( auto i = chosenBonds.size(); i < nNeighborsLastStep; ++i )
            index[i] = 0;
        nNeighborsLastStep = chosenBonds.size();
        chosenBonds = getNeighborConfiguration< T_Dim, T_BondInt, T_BondId >( index, excludes );

        if ( chosenBonds.size() > maxConfig.size() )
        {
            maxConfig = chosenBonds;
            std::cout
            << "!!! After testing " << nTested << " configurations, "
            << "we found a new maximum bondset with " << maxConfig.size()
            << " neighbors:\n" << maxConfig << std::endl;
        }

        ++nTested;
        if ( nTested % 1000000 == 0 )
        {
            std::cout << "Tested " << nTested << " current index: (" << (int) index[0];
            for ( auto i = 1u; i < chosenBonds.size(); ++i )
                std::cout << ", " << (int) index[i];
            std::cout << ") \nwhich translates to: " << chosenBonds << "\n";
            std::cout << "Current length: " << chosenBonds.size() << " vs. max length: " << maxConfig.size() << " at " << maxConfig << std::endl;
            std::cout << "Testing at a rate of " << nTested / diffNow( t0, now() ) << " configs/s" << std::endl;
        }
    }
    std::cout << "There are in total " << nTested << " possible configurations for the given bondset!\n";
    return maxConfig;
}

/**
 * @todo Currently we don't use the fact, that the order does not matter to
 *       reduce the problem size
 * @todo We first increase the last indexes all the while they have the least
 *       influence... Would be better to increment the first, but that would
 *       breadth first search would be hard to implement, because we have to
 *       store every unfinished branch.
 *        -> The other idea of following the least amount of forbidden ones,
 *           might just exactly tackle this problem by reordering the search
 */
template<
    int      T_Dim     = 3,
    typename T_BondInt = short int,
    typename T_BondId  = unsigned short int
>
inline
std::vector< T_BondId > findMaxNeighborsNaiveInlined
(
    std::vector< std::array< T_BondInt, T_Dim > > const & bonds,
    unsigned int const cubeHalfSize = 1
)
{
    auto const excludes = createExclusionTable< T_Dim, T_BondInt >( bonds, cubeHalfSize );
    /**
     * stores which n-th allowed bond we should choose. E.g. (0,1,2)
     * chooses the first bond allowed, then the second bond of still
     * allowed bonds and then the third bond which is still allowed after
     * adding the prior two bonds to chosenBonds.
     */
    std::vector< T_BondId > index( bonds.size(), 0 );
    auto chosenBonds = getNeighborConfiguration< T_Dim, T_BondInt, T_BondId >( index, excludes );
    auto maxConfig   = chosenBonds;

    /* things needed for inlined getNeighborConfiguration */
    T_BondId nChosenBonds = chosenBonds.size();
    chosenBonds.resize( bonds.size(), 0 );  // work with one size in order to avoid allocations
    std::vector< T_BondId > prohibitedBonds( excludes.size(), 0 );
    auto const compressedExcludes = compressExclusionTable< T_Dim, T_BondId >( excludes );
    auto const nPadded = compressedExcludes.size() / excludes.size();

    T_BondId nNeighborsLastStep = chosenBonds.size();
    size_t nTested = 0;
    auto const t0 = now();
    while ( nChosenBonds > 0 )
    {
        /* increment index noting that the length is variable ... */
        ++index[ nChosenBonds - 1 ];
        for ( auto i = nChosenBonds; i < nNeighborsLastStep; ++i )
            index[i] = 0;
        nNeighborsLastStep = nChosenBonds;

        /* own getNeighborConfiguration implementation */
        std::memset( &prohibitedBonds[0], 0, prohibitedBonds.size() * sizeof( prohibitedBonds[0] ) );
        nChosenBonds = 0;
        for ( T_BondId i = 0u; i < index.size(); ++i )
        {
            /* find first index[i]-th allowed bond (special case: find first allowed) */
            int iChosen = -1;
            for ( T_BondId j = 0u, nAllowed = 0u; j < prohibitedBonds.size(); ++j )
            {
                nAllowed += prohibitedBonds[j] == 0;
                if ( nAllowed > index[i] )
                {
                    iChosen = j;
                    break;
                }
            }
            if ( iChosen < 0 )
                break;
            chosenBonds[ nChosenBonds++ ] = iChosen;
            for ( T_BondId j = 0u; compressedExcludes[ iChosen*nPadded+j ] < bonds.size(); ++j )
                ++prohibitedBonds[ compressedExcludes[ iChosen*nPadded+j ] ];
        }

        if ( nChosenBonds > maxConfig.size() )
        {
            maxConfig = chosenBonds;
            maxConfig.resize( nChosenBonds );
            std::cout
            << "!!! After testing " << nTested << " configurations, "
            << "we found a new maximum bondset with " << maxConfig.size()
            << " neighbors:\n" << maxConfig << std::endl;
        }

        ++nTested;
        if ( nTested % 10000000 == 0 )
        {
            std::cout << "Tested " << nTested << " current index: (" << (int) index[0];
            for ( auto i = 1u; i < nChosenBonds; ++i )
                std::cout << ", " << (int) index[i];
            std::cout << ") \nwhich translates to: (" << chosenBonds[0];
            for ( auto i = 1u; i < nChosenBonds; ++i )
                std::cout << ", " << (int) chosenBonds[i];
            std::cout << ")\n";
            std::cout << "Current length: " << (int) nChosenBonds << " vs. max length: " << maxConfig.size() << " at " << maxConfig << std::endl;
            std::cout << "Testing at a rate of " << nTested / diffNow( t0, now() ) << " configs/s" << std::endl;
        }
    }
    std::cout << "There are in total " << nTested << " possible configurations for the given bondset!\n";
    return maxConfig;
}

template<
    int      T_Dim     = 3,
    typename T_BondInt = short int,
    typename T_BondId  = unsigned short int
>
inline
std::vector< T_BondId > findMaxNeighborsNaiveInlinedOnlyIncreasing
(
    std::vector< std::array< T_BondInt, T_Dim > > const & bonds,
    unsigned int const cubeHalfSize = 1
)
{
    auto const excludes = createExclusionTable< T_Dim, T_BondInt >( bonds, cubeHalfSize );
    /**
     * stores which n-th allowed bond we should choose. E.g. (0,1,2)
     * chooses the first bond allowed, then the second bond of still
     * allowed bonds and then the third bond which is still allowed after
     * adding the prior two bonds to chosenBonds.
     */
    std::vector< T_BondId > index( bonds.size(), 0 );
    auto chosenBonds = getNeighborConfiguration< T_Dim, T_BondInt, T_BondId >( index, excludes );
    auto maxConfig   = chosenBonds;

    /* things needed for inlined getNeighborConfiguration */
    T_BondId nChosenBonds = chosenBonds.size();
    chosenBonds.resize( bonds.size(), 0 );  // work with one size in order to avoid allocations
    std::vector< T_BondId > prohibitedBonds( excludes.size(), 0 );
    auto const compressedExcludes = compressExclusionTable< T_Dim, T_BondId >( excludes );
    auto const nPadded = compressedExcludes.size() / excludes.size();

    T_BondId constexpr nMaxNeighborsAssumed = 43; // 42.87, see comments at start of file
    std::vector< size_t > frequencies( nMaxNeighborsAssumed, 0 );
    T_BondId nNeighborsLastStep = chosenBonds.size();
    size_t nTested = 0;
    auto const t0 = now();
    while ( nChosenBonds > 0 )
    {
        /* increment index noting that the length is variable ... */
        ++index[ nChosenBonds - 1 ];
        for ( auto i = nChosenBonds; i < nNeighborsLastStep; ++i )
            index[i] = 0;
        nNeighborsLastStep = nChosenBonds;

        /* own getNeighborConfiguration implementation */
        std::memset( &prohibitedBonds[0], 0, prohibitedBonds.size() * sizeof( prohibitedBonds[0] ) );
        nChosenBonds = 0;
        int iLastChosen = 0;
        for ( T_BondId i = 0u; i < index.size(); ++i )
        {
            /* find first index[i]-th allowed bond (special case: find first allowed) */
            /* plus new condition: the bond must be of higher ID than the last!
             * Using prohibiteBonds down below is almost half as fast as just
             * rechecking it here */
            int iChosen = -1;
            for ( T_BondId j = 0u, nAllowed = 0u; j < prohibitedBonds.size(); ++j )
            {
                nAllowed += prohibitedBonds[j] == 0 && j >= iLastChosen /* !!! NEW !!! */;
                if ( nAllowed > index[i] )
                {
                    iChosen = j;
                    break;
                }
            }
            if ( iChosen < 0 )
                break;
            chosenBonds[ nChosenBonds++ ] = iChosen;
            iLastChosen = iChosen;
            for ( T_BondId j = 0u; compressedExcludes[ iChosen*nPadded+j ] < bonds.size(); ++j )
                ++prohibitedBonds[ compressedExcludes[ iChosen*nPadded+j ] ];
            /* disallow all of smaller ID, so that the chosenBonds vector will
             * be in increasing sorted order! */
            /* for ( T_BondId j = 0u; j < iChosen; ++j )
                ++prohibitedBonds[j]; */
        }

        if ( nChosenBonds > maxConfig.size() )
        {
            maxConfig = chosenBonds;
            maxConfig.resize( nChosenBonds );
            std::cout
            << "!!! After testing " << nTested << " configurations, "
            << "we found a new maximum bondset with " << maxConfig.size()
            << " neighbors:\n" << maxConfig << std::endl;
        }

        ++frequencies[ nChosenBonds ];
        ++nTested;

        if ( nTested % 10000000 == 0 )
        {
            std::cout << "\nTested " << nTested << " at a rate of " << nTested / diffNow( t0, now() ) << " configs/s\n";
            std::cout << "current index:\n(" << (int) index[0];
            for ( auto i = 1u; i < nChosenBonds; ++i )
                std::cout << ", " << (int) index[i];
            std::cout << ") \n";

            auto maxIndex = index;
            std::memset( &prohibitedBonds[0], 0, prohibitedBonds.size() * sizeof( prohibitedBonds[0] ) );
            int iLastChosen = 0; nChosenBonds = 0;
            for ( T_BondId i = 0u; i < index.size(); ++i )
            {
                int iChosen = -1, nAllowed = 0u;
                for ( T_BondId j = 0u; j < prohibitedBonds.size(); ++j )
                {
                    nAllowed += prohibitedBonds[j] == 0 && j >= iLastChosen;
                    if ( nAllowed > index[i] && iChosen == -1 )
                        iChosen = j;
                }
                maxIndex[i] = nAllowed;
                if ( iChosen < 0 )
                    break;
                ++nChosenBonds;
                iLastChosen = iChosen;
                for ( T_BondId j = 0u; compressedExcludes[ iChosen*nPadded+j ] < bonds.size(); ++j )
                    ++prohibitedBonds[ compressedExcludes[ iChosen*nPadded+j ] ];
            }

            std::cout << "These are the maximum number of possible bonds per each index entry for the current configuration:\n(" << maxIndex[0];
            for ( auto i = 1u; i < nChosenBonds; ++i )
                std::cout << ", " << (int) maxIndex[i];
            std::cout << ") \n";
            double nToTest = 1;
            for ( auto i = 1u; i < nChosenBonds; ++i )
                nToTest *= maxIndex[i];
            std::cout
            << "Based on these maximums we have done an estimated "
            << nTested * 100. / nToTest << "% of the work in "
            << diffNow( t0, now() ) << "s => will finish in "
            << diffNow( t0, now() ) / nTested * nToTest << "s\n";

            double ratioDone = 1;
            for ( auto i = 1u; i < nChosenBonds; ++i )
                ratioDone *= (double) (index[i]+1) / maxIndex[i];
            std::cout << "Possibly better estimate: " << ratioDone * 100
            << "% of the work => will finish in " << diffNow( t0, now() ) / ratioDone << "s\n";

            std::cout << "The index translates to these bond IDs: (" << chosenBonds[0];
            for ( auto i = 1u; i < nChosenBonds; ++i )
                std::cout << ", " << (int) chosenBonds[i];
            std::cout << ")\n";
            std::cout << "Current length: " << (int) nChosenBonds << " vs. max length: " << maxConfig.size() << " at " << maxConfig << std::endl;

            for ( auto i = 0u; i < nMaxNeighborsAssumed; ++i )
            {
                if ( frequencies[i] == 0 )
                    continue;
                std::cout << i << ":" << std::setprecision(4) << frequencies[i] * 100. / nTested << "%, ";
            }
            std::cout << std::setprecision(7) << std::endl;
        }
    }
    std::cout << "There are in total " << nTested << " possible configurations for the given bondset!\n";
    return maxConfig;
}

/**
 * same as findMaxNeighborsNaiveInlined, but index doesn't get increased
 * sequentially, but instead gets assigned random numbers. And if if an
 * index is out of bounds for the allowed monomers, even though there are
 * some allowed, then wrap the index around using modulo
 *
 * Example Output:
 * @verbatim
 * Testing at a rate of 278162 configs/s
 * After testing 866 configurations, we found a new maximum bondset
 * with 20 neighbors:
 * {61, 59, 20, 54, 19, 17, 56, 60, 103, 15, 16, 105, 26, 58, 96, 14,
 *  99, 102, 55, 57}
 * Sorted:
 * {14, 15, 16, 17, 19, 20, 26, 55, 54, 56, 57, 58, 59, 60, 61, 96, 99, 102, 103, 105 }
 * @endverbatim
 */
template<
    int      T_Dim     = 3,
    typename T_BondInt = short int,
    typename T_BondId  = unsigned short int
>
inline
std::vector< T_BondId > findMaxNeighborsNaiveInlinedMonteCarlo
(
    std::vector< std::array< T_BondInt, T_Dim > > const & bonds,
    unsigned int const cubeHalfSize = 1
)
{
    auto const excludes = createExclusionTable< T_Dim, T_BondInt >( bonds, cubeHalfSize );
    /**
     * stores which n-th allowed bond we should choose. E.g. (0,1,2)
     * chooses the first bond allowed, then the second bond of still
     * allowed bonds and then the third bond which is still allowed after
     * adding the prior two bonds to chosenBonds.
     */
    std::vector< T_BondId > index( bonds.size(), 0 );
    auto chosenBonds = getNeighborConfiguration< T_Dim, T_BondInt, T_BondId >( index, excludes );
    auto maxConfig   = chosenBonds;

    /* things needed for inlined getNeighborConfiguration */
    T_BondId nChosenBonds = chosenBonds.size();
    chosenBonds.resize( bonds.size(), 0 );  // work with one size in order to avoid allocations
    std::vector< T_BondId > prohibitedBonds( excludes.size(), 0 );
    auto const compressedExcludes = compressExclusionTable< T_Dim, T_BondId >( excludes );
    auto const nPadded = compressedExcludes.size() / excludes.size();

    std::srand( 26545721 );
    //T_BondId constexpr nMaxNeighborsAssumed = 43; // 42.87, see comments at start of file
    T_BondId constexpr nMaxNeighborsAssumed = 30; // 42.87, see comments at start of file
    std::vector< size_t > frequencies( nMaxNeighborsAssumed, 0 );
    size_t nTested = 0;
    auto const t0 = now();
    while ( nChosenBonds > 0 )
    {
        /* choose random index */
        for ( auto i = 0u; i < nMaxNeighborsAssumed; ++i )
            index[i] = std::rand(); // don't even need modulo, gets typecasted anyway

        /* own getNeighborConfiguration implementation */
        std::memset( &prohibitedBonds[0], 0, prohibitedBonds.size() * sizeof( prohibitedBonds[0] ) );
        nChosenBonds = 0;
        for ( T_BondId i = 0u; i < index.size(); ++i )
        {
            /* find first index[i]-th allowed bond (special case: find first allowed) */
            int iChosen = -1;
            T_BondId nAllowedTotal = 0u;
            for ( T_BondId j = 0u; j < prohibitedBonds.size(); ++j )
                nAllowedTotal += prohibitedBonds[j] == 0;
            if ( nAllowedTotal == 0 )
                break;
            index[i] %= nAllowedTotal;

            T_BondId nAllowed = 0u;
            for ( T_BondId j = 0u; j < prohibitedBonds.size(); ++j )
            {
                nAllowed += prohibitedBonds[j] == 0;
                if ( nAllowed > index[i] )
                {
                    iChosen = j;
                    break;
                }
            }
            assert( iChosen != -1 );
            chosenBonds[ nChosenBonds++ ] = iChosen;
            for ( T_BondId j = 0u; compressedExcludes[ iChosen*nPadded+j ] < bonds.size(); ++j )
                ++prohibitedBonds[ compressedExcludes[ iChosen*nPadded+j ] ];
        }

        if ( nChosenBonds > maxConfig.size() )
        {
            maxConfig = chosenBonds;
            maxConfig.resize( nChosenBonds );
            std::cout
            << "!!! After testing " << nTested << " configurations, "
            << "we found a new maximum bondset with " << maxConfig.size()
            << " neighbors:\n" << maxConfig << std::endl;
        }

        ++frequencies[ nChosenBonds ];
        ++nTested;
        if ( nTested % 1000000 == 0 )
        {
            std::cout << "Tested " << nTested << " current index: (" << (int) index[0];
            for ( auto i = 1u; i < nChosenBonds; ++i )
                std::cout << ", " << (int) index[i];
            std::cout << ") \nwhich translates to: (" << chosenBonds[0];
            for ( auto i = 1u; i < nChosenBonds; ++i )
                std::cout << ", " << (int) chosenBonds[i];
            std::cout << ")\n";
            std::cout << "Current length: " << (int) nChosenBonds << " vs. max length: " << maxConfig.size() << " at " << maxConfig << std::endl;
            std::cout << "Testing at a rate of " << nTested / diffNow( t0, now() ) << " configs/s" << std::endl;
            for ( auto i = 0u; i < nMaxNeighborsAssumed; ++i )
            {
                if ( frequencies[i] == 0 )
                    continue;
                std::cout << i << ":" << std::setprecision(4) << frequencies[i] * 100. / nTested << "%, ";
            }
            std::cout << std::setprecision(7) << std::endl;
        }
    }
    std::cout << "Checked " << nTested << " configurations in total!\n";
    return maxConfig;
}



#include <cmath>                                // pow

#include <extern/Fundamental/vectorIndex.hpp>   // convertVectorToLinearIndex

template<
    int      T_Dim     = 3,
    typename T_BondInt = short int,
    typename T_BondId  = unsigned short int
>
inline bool checkConfiguration
(
    std::vector< std::array< T_BondInt, T_Dim > > const & bonds,
    std::vector< T_BondId >                       const & config,
    unsigned int const cubeHalfSize = 1
)
{
    /* find min and max bond vectors per each dimension */
    std::vector< T_BondInt > mins( T_Dim, 0 ), maxs( T_Dim, 0 );
    std::vector< size_t > center( T_Dim, 0 ), ns( T_Dim, 0 );
    for ( auto const & bond : bonds )
    for ( int i = 0; i < T_Dim; ++i )
    {
        mins[i] = std::min( mins[i], bond[i] );
        maxs[i] = std::max( maxs[i], bond[i] );
    }
    /* create lattice */
    for ( int i = 0u; i < T_Dim; ++i )
        ns[i] = maxs[i] - mins[i] + 1 /* center */ + 2 * cubeHalfSize /* or else we couldn't check neighbors of outmost monomers */;
    size_t nCells = 1;
    for ( int i = 0u; i < T_Dim; ++i )
        nCells *= ns[i];
    std::vector< bool > lattice( nCells, false );
    /* center must be such, that center-mins and center+maxs is inside lattice */
    for ( int i = 0; i < T_Dim; ++i )
        center[i] = -mins[i] + cubeHalfSize; /* => center[i] + mins[i] == 0 and center[i] + maxs[i] == maxs[i]-mins[i] */
    /*
    std::cout << "mins = " << mins << "\n";
    std::cout << "maxs = " << maxs << "\n";
    std::cout << "center = " << center << "\n";
    std::cout << "ns = " << ns << "\n";
    */
    /* try to insert all bonds given in 'config' into lattice thereby
     * checking for collisions */
    auto const nNeighbors = std::pow( 2*cubeHalfSize+1, T_Dim );
    std::vector< unsigned int > cubeDimensions( T_Dim, 2*cubeHalfSize+1 );

    //std::cout << "center " << center << " at linid " << convertVectorToLinearIndex( center, ns ) << std::endl;
    lattice[ convertVectorToLinearIndex( center, ns ) ] = true;
    for ( auto const & iBond : config )
    {
        /* check all points in cubeHalfSize */
        for ( size_t linId = 0u; linId < nNeighbors; ++linId )
        {
            /* we create a relative vector with this and need to shift it
             * to the current bond's position */
            auto vecNeighbor = convertLinearToVectorIndex( linId, cubeDimensions );
            for ( int j = 0u; j < T_Dim; ++j )
                vecNeighbor[j] += center[j] + bonds[iBond][j] - cubeHalfSize;
            /* we don't have to ignore ourselves, because we aren't inserted yet! */
            /*
            std::cout
                << "neighbor " << linId << " = " << vecNeighbor << " = "
                << convertLinearToVectorIndex( linId, cubeDimensions ) << " - "
                << cubeHalfSize << " + " << center << " + "
                << toString< T_BondInt, T_Dim >( bonds[iBond] ) << " in " << ns
                << " at linid " << convertVectorToLinearIndex( vecNeighbor, ns )
                << std::endl;
            */
            if ( lattice.at( convertVectorToLinearIndex( vecNeighbor, ns ) ) )
                return false;
        }
        /* if we are here, then all is ok and we can insert */
        auto newPos = center;
        for ( int i = 0u; i < T_Dim; ++i )
            newPos[i] += bonds[iBond][i];
        //std::cout << "add new monomer " << iBond << " at " << toString< T_BondInt, T_Dim >( bonds[iBond] ) << " => " << newPos << " at linid " << convertVectorToLinearIndex( newPos, ns ) << std::endl;
        lattice.at( convertVectorToLinearIndex( newPos, ns ) ) = true;
    }

    return true;
}

/**
 * This version is the same as findMaxNeighborsNaiveInlinedOnlyIncreasing,
 * but additionally it keeps track of the upper bound for the maximum
 * neighbors. Combined with a sorting of the bondIDs such that bond 0 forbids
 * the least amount of bonds, meaning the very first iterations could already
 * give the currently known maximum neighbor estimate of 20, this could reduce
 * the number of tests 10-fold, e.g. for this case:
 *   Bonds used:   1   2   3   4   5   6   7   8   9  10  11 12 13
 *   Index     :(  0,  0,  0,  4, 11,  0,  5,  0,  0,  1,  0, 0, 2)
 *   Max. free :(108, 94, 80, 66, 51, 33, 27, 19, 16, 13, 10, 7, 4)
 *     => For that index we could already see at bond 11 that this results in
 *        only 7 free choosable bonds after that, meaning upper bound 18 < 20
 *        Therefore we could roughly save 7*4~30x tests!!
 */
template<
    int      T_Dim     = 3,
    typename T_BondInt = short int,
    typename T_BondId  = unsigned short int
>
inline
std::vector< T_BondId >
findMaxNeighborsNaiveInlinedOnlyIncreasingTrackUpperBounds
(
    std::vector< std::array< T_BondInt, T_Dim > > * const bonds,
    unsigned int                                    const cubeHalfSize =  1,
    double                                          const timeout      = -1
)
{
    auto const excludes = createExclusionTable< T_Dim, T_BondInt >( *bonds, cubeHalfSize );
    /* sort bonds such that the lowest ID forbids the least amount of other
     * bonds. First count forbidden bonds, the create a list of pairs with
     * (bondId,nForbiden) so that we get a mapping of index -> bondId */
    std::vector< std::pair< T_BondId, T_BondId > > bondIdMapping( excludes.size() );
    for ( auto i = 0u; i < excludes.size(); ++i )
    {
        T_BondId nForbidden = 0u;
        for ( auto j = 0u; j < excludes[i].size(); ++j )
            nForbidden += excludes[i][j];
        bondIdMapping[i].first  = i;
        bondIdMapping[i].second = nForbidden;
    }
    std::sort( bondIdMapping.begin(), bondIdMapping.end(),
        []( std::pair< T_BondId, T_BondId > const & a,
            std::pair< T_BondId, T_BondId > const & b )
        {
            /* if excluded bonds are equal, then sort by their original ID */
            return ( a.second != b.second ? a.second < b.second : a.first < b.first );
        }
    );
    auto bondsSorted = *bonds;
    for ( auto i = 0u; i < excludes.size(); ++i )
        bondsSorted[i] = (*bonds)[ bondIdMapping[i].first ];
    *bonds = bondsSorted;
    /* we could try to manually sort excludes, but then we not only would have
     * to reorder the rows, but also the columns! In the end creating it anew
     * from the sorted bonds should be easier and doesn't matter as it is
     * just initialization */
    auto const excludesSorted = createExclusionTable< T_Dim, T_BondInt >( bondsSorted, cubeHalfSize );

    /* debug print bonds in sorted order, so we can understand the solution */
    std::cout << "All " << bondsSorted.size() << " bonds sorted by the amount of bonds they forbid:\n";
    std::cout << "Scheme: <new bond ID>[<old bond ID>]:<bond vector>=<number of bonds it forbids>";
    for ( size_t i = 0u; i < bondsSorted.size(); ++i )
    {
        if ( i % 4 == 0 )
            std::cout << std::endl;
        std::cout
        << std::setw(4) << i << "[" << std::setw(3) << bondIdMapping[i].first << "]:("
        << std::setw(2) << (int) bondsSorted[i][0] << ","
        << std::setw(2) << (int) bondsSorted[i][1] << ","
        << std::setw(2) << (int) bondsSorted[i][2] << ")="
        << std::setw(2) << (int) bondIdMapping[i].second
        << ( i < bondsSorted.size() - 1 ? "," : "\n" );
    }
    std::cout << std::endl;

    std::cout << "Table for bond i excludes bond j:\n";
    unsigned int const nRows = 40u, nCols = 40u;
    //unsigned int const nRows = bonds->size(), nCols = bonds->size();
    std::cout << "i\\j|";
    for ( auto j = 0u; j < nCols; ++j )
        std::cout << std::setw(2) << j % 10;
    std::cout << "\n" << std::setw(4+2*nCols) << std::setfill('-') << "" << std::setfill(' ') << "\n";
    for ( auto i = 0u; i < nRows; ++i )
    {
        std::cout << std::setw(3) << i << "|";
        for ( auto j = 0u; j < nCols; ++j )
            std::cout << std::setw(2) << ( excludesSorted[i][j] ? "x" : " " );
        std::cout << std::endl;
    }

    /**
     * stores which n-th allowed bond we should choose. E.g. (0,1,2)
     * chooses the first bond allowed, then the second bond of still
     * allowed bonds and then the third bond which is still allowed after
     * adding the prior two bonds to chosenBonds.
     */
    std::vector< T_BondId > index( bonds->size(), 0 );
    auto chosenBonds = getNeighborConfiguration< T_Dim, T_BondInt, T_BondId >( index, excludesSorted );
    auto maxConfig   = chosenBonds;

    std::cout << "chosenBonds: " << chosenBonds << "\n";
    std::cout << "number of bonds / neighbors: " << chosenBonds.size() << "\n";
    std::cout << "Result of checkConfiguration: " << checkConfiguration<3>( bondsSorted, chosenBonds, cubeHalfSize ) << std::endl;

    /* things needed for inlined getNeighborConfiguration */
    T_BondId nChosenBonds = chosenBonds.size();
    chosenBonds.resize( bonds->size(), 0 );  // work with one size in order to avoid allocations
    std::vector< T_BondId > prohibitedBonds( excludesSorted.size(), 0 );
    auto const compressedExcludes = compressExclusionTable< T_Dim, T_BondId >( excludes );
    auto const nPadded = compressedExcludes.size() / excludesSorted.size();

    std::cout << "compressedExcludes was padded to: " << nPadded << " so it has a total of " << compressedExcludes.size() << " elements in " << compressedExcludes.size() * sizeof( compressedExcludes[0] ) << " Bytes:\n";
    for ( auto i = 0u; i < excludesSorted.size(); ++i )
    {
        std::cout << std::setw(3) << i << ": ";
        for ( auto j = 0u; j < nPadded; ++j )
            std::cout << std::setw(3) << (int) compressedExcludes[ i * nPadded + j ] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    T_BondId constexpr nMaxNeighborsAssumed = 43; // 42.87, see comments at start of file
    std::vector< size_t > frequencies( nMaxNeighborsAssumed, 0 );
    T_BondId nNeighborsLastStep = chosenBonds.size();
    size_t nTested = 0;
    auto const t0 = now();
    while ( nChosenBonds > 0 )
    {
        /* increment index noting that the length is variable ... */
        ++index[ nChosenBonds - 1 ];
        if ( nChosenBonds < nNeighborsLastStep )
            std::memset( &index[nChosenBonds], 0, ( nNeighborsLastStep - nChosenBonds ) * sizeof( index[0] ) );
        nNeighborsLastStep = nChosenBonds;

        /* own getNeighborConfiguration implementation */
        std::memset( &prohibitedBonds[0], 0, prohibitedBonds.size() * sizeof( prohibitedBonds[0] ) );
        nChosenBonds = 0;
        int iLastChosen = 0;
        bool upperBoundTooLow = false;
        T_BondId constexpr iStartCheckingUpperBound = 7;
        for ( T_BondId i = 0u; i < index.size(); ++i )
        {
            /* find first index[i]-th allowed bond (special case: find first allowed) */
            /* plus new condition: the bond must be of higher ID than the last!
             * Using prohibiteBonds down below is almost half as fast as just
             * rechecking it here */
            int iChosen = -1;
            T_BondId nAllowed = 0u;
            for ( T_BondId j = 0u; j < prohibitedBonds.size(); ++j )
            {
                nAllowed += prohibitedBonds[j] == 0 && j >= iLastChosen;
                if ( nAllowed > index[i] && iChosen == -1 )
                {
                    iChosen = j;
                    /* the missing break almost (or the iChosen == -1)
                     * comparison(?) incurred almost a 3x slowdown. Plus the
                     * statistics signal, that the most common lengths begin at
                     * i >= 7. We could therefore maybe save almost 40% by
                     * limiting the full nAllowed calculation to i >= 7 ! */
                    if ( i < iStartCheckingUpperBound )
                        break;
                }
            }
            if ( i >= iStartCheckingUpperBound && i + nAllowed <= maxConfig.size() )
            {
                //std::cout << "Currently at i=" << (int) i << "-th element of 'index' at which nAllowed=" << nAllowed << " allowed bonds are left. Therefore as i+nAllowed <= " << maxConfig.size() << " the current best contender, we can skip over this branch. Therefore in the next step element " << nChosenBonds-1 << " will be incremented and all right to it set to 0" << std::endl;
                //return {};
                upperBoundTooLow = true;
                break;
            }
            //std::cout << (int) i << ":" << index[i] << "/" << nAllowed << " ";
            if ( iChosen < 0 )
                break;
            chosenBonds[ nChosenBonds++ ] = iChosen;
            iLastChosen = iChosen;
            for ( T_BondId j = 0u; compressedExcludes[ iChosen*nPadded+j ] < bonds->size(); ++j )
                ++prohibitedBonds[ compressedExcludes[ iChosen*nPadded+j ] ];
            /* disallow all of smaller ID, so that the chosenBonds vector will
             * be in increasing sorted order! */
            /* for ( T_BondId j = 0u; j < iChosen; ++j )
                ++prohibitedBonds[j]; */
        }
        ++frequencies[ nChosenBonds ];
        ++nTested;

        //std::cout << "\nupperBoundTooLow = " << upperBoundTooLow << std::endl;
        if ( upperBoundTooLow )
            continue;
        //return {};
        /* we still can land here, if the loop above breaks, because an element
         * in index would be out of range (less allowed than index is large) */

        if ( nChosenBonds > maxConfig.size() )
        {
            maxConfig = chosenBonds;
            maxConfig.resize( nChosenBonds );
            std::cout
            << "!!! After testing " << nTested << " configurations, "
            << "we found a new maximum bondset with " << maxConfig.size()
            << " neighbors:\n" << maxConfig << std::endl;
        }

        if ( nTested % 1000000 == 0 )
        {
            if ( timeout > 0 && diffNow( t0, now() ) > timeout )
            {
                std::cout << "Timout of " << timeout << "s reached, quitting preemptively\n";
                break;
            }

            std::cout << "\nTested " << nTested << " at a rate of " << nTested / diffNow( t0, now() ) << " configs/s\n";
            std::cout << "current index:\n(" << (int) index[0];
            for ( auto i = 1u; i < nChosenBonds; ++i )
                std::cout << ", " << (int) index[i];
            std::cout << ") \n";

            auto maxIndex = index;
            std::memset( &prohibitedBonds[0], 0, prohibitedBonds.size() * sizeof( prohibitedBonds[0] ) );
            int iLastChosen = 0; nChosenBonds = 0;
            for ( T_BondId i = 0u; i < index.size(); ++i )
            {
                int iChosen = -1, nAllowed = 0u;
                for ( T_BondId j = 0u; j < prohibitedBonds.size(); ++j )
                {
                    nAllowed += prohibitedBonds[j] == 0 && j >= iLastChosen;
                    if ( nAllowed > index[i] && iChosen == -1 )
                        iChosen = j;
                }
                maxIndex[i] = nAllowed;
                if ( iChosen < 0 )
                    break;
                ++nChosenBonds;
                iLastChosen = iChosen;
                for ( T_BondId j = 0u; compressedExcludes[ iChosen*nPadded+j ] < bonds->size(); ++j )
                    ++prohibitedBonds[ compressedExcludes[ iChosen*nPadded+j ] ];
            }

            std::cout << "These are the maximum number of possible bonds per each index entry for the current configuration:\n(" << maxIndex[0];
            for ( auto i = 1u; i < nChosenBonds; ++i )
                std::cout << ", " << (int) maxIndex[i];
            std::cout << ") \n";
            double nToTest = 1;
            for ( auto i = 1u; i < nChosenBonds; ++i )
                nToTest *= maxIndex[i];
            std::cout
            << "Based on these maximums we have done an estimated "
            << nTested * 100. / nToTest << "% of the work in "
            << diffNow( t0, now() ) << "s => will finish in "
            << diffNow( t0, now() ) / nTested * nToTest << "s\n";

            double ratioDone = 1;
            for ( auto i = 1u; i < nChosenBonds; ++i )
                ratioDone *= (double) (index[i]+1) / maxIndex[i];
            std::cout << "Possibly better estimate: " << ratioDone * 100
            << "% of the work => will finish in " << diffNow( t0, now() ) / ratioDone << "s\n";

            std::cout << "The index translates to these (sorted) bond IDs: (" << (int) chosenBonds[0];
            for ( auto i = 1u; i < nChosenBonds; ++i )
                std::cout << ", " << (int) chosenBonds[i];
            std::cout << ")\n";

            std::cout << "In the original bond ordering the IDs would be: (" << (int) bondIdMapping[ chosenBonds[0] ].first;
            for ( auto i = 1u; i < nChosenBonds; ++i )
                std::cout << ", " << (int) bondIdMapping[ chosenBonds[i] ].first;
            std::cout << ")\n";

            std::cout << "How many each bond ID excludes: (" << (int) bondIdMapping[ chosenBonds[0] ].second;
            for ( auto i = 1u; i < nChosenBonds; ++i )
                std::cout << ", " << (int) bondIdMapping[ chosenBonds[i] ].second;
            std::cout << ")\n";

            std::cout << "Current length: " << (int) nChosenBonds << " vs. max length: " << maxConfig.size() << " at " << maxConfig << std::endl;

            /* 2:2.558e-06%, 3:0.0001937%, 4:0.005896%, 5:0.09599%, 6:0.9111%, 7:5.141%, 8:16.75%, 9:29.13%, 10:26.19%, 11:14.46%, 12:5.49%, 13:1.497%, 14:0.2941%, 15:0.03623%, 16:0.003378%, 17:0.0003279%, 18:2.814e-05%, */
            for ( auto i = 0u; i < nMaxNeighborsAssumed; ++i )
            {
                if ( frequencies[i] == 0 )
                    continue;
                std::cout << i << ":" << std::setprecision(4) << frequencies[i] * 100. / nTested << "%, ";
            }
            std::cout << std::setprecision(7) << std::endl;
        }
    }
    std::cout << "Checked " << nTested << " configurations in total!\n";
    return maxConfig;
}


#include <fstream>


template<
    int      T_Dim     = 3,
    typename T_BondInt = short int,
    typename T_BondId  = unsigned short int
>
inline void saveConfiguration
(
    std::vector< std::array< T_BondInt, T_Dim > > const & bonds,
    std::vector< T_BondId >                       const & config,
    std::string                                   const & fname
)
{
    /* find min and max bond vectors per each dimension */
    std::vector< T_BondInt > mins( T_Dim, 0 ), maxs( T_Dim, 0 );
    std::vector< size_t > center( T_Dim, 0 ), ns( T_Dim, 0 );
    for ( auto const & bond : bonds )
    for ( int i = 0; i < T_Dim; ++i )
    {
        mins[i] = std::min( mins[i], bond[i] );
        maxs[i] = std::max( maxs[i], bond[i] );
    }
    /* create lattice */
    auto const cubeHalfSize = 1; // just margin for this case, named like this, because copied from checkConfiguration
    for ( int i = 0u; i < T_Dim; ++i )
        ns[i] = maxs[i] - mins[i] + 1 /* center */ + 2 * cubeHalfSize /* or else we couldn't check neighbors of outmost monomers */;
    /* center must be such, that center-mins and center+maxs is inside lattice */
    for ( int i = 0; i < T_Dim; ++i )
        center[i] = -mins[i] + cubeHalfSize; /* => center[i] + mins[i] == 0 and center[i] + maxs[i] == maxs[i]-mins[i] */

    std::ofstream bfm( fname, std::ios::out | std::ios::binary );
    bfm << "#!version=2.0\n"
        << "!number_of_monomers=" << config.size()+1 << "\n";
    if ( T_Dim >= 1 )
        bfm << "!box_x=" << ns[0] << "\n";
    if ( T_Dim >= 2 )
        bfm << "!box_y=" << ns[1] << "\n";
    if ( T_Dim >= 3 )
        bfm << "!box_z=" << ns[2] << "\n";

    if ( T_Dim >= 1 )
        bfm << "!periodic_x=0\n";
    if ( T_Dim >= 2 )
        bfm << "!periodic_y=0\n";
    if ( T_Dim >= 3 )
        bfm << "!periodic_z=0\n";

    bfm << "!set_of_bondvectors\n";
    auto const bondIdOffset = 21; // better to begin with 21, so we exclude 20 and 13 (newline) and null .. -.-
    for ( size_t i = 0u; i < bonds.size(); ++i )
        bfm << (int) bonds[i][0] << " " << (int) bonds[i][1] << " " << (int) bonds[i][2] << ":" << (int) i + bondIdOffset << "\n";

    /* connectome for not implicit bonds. We won't use implicit bonds, so add all */
    bfm << "!bonds\n";
    /* center monomer is to be ID 1 (not counting from zero -.-) */
    auto const idCenter = 1;
    for ( auto i = 0u; i < config.size(); ++i )
        bfm << idCenter << " " << int( idCenter + i + 1 ) << "\n";

    /* positions */
    bfm << "!mcs=0\n";
    bfm << (int) center[0] << " " << (int) center[1] << " " << (int) center[2] << "\n";
    for ( auto i = 0u; i < config.size(); ++i )
    {
        bfm << (int)( center[0] + bonds[ config[i] ][0] ) << " "
            << (int)( center[1] + bonds[ config[i] ][1] ) << " "
            << (int)( center[2] + bonds[ config[i] ][2] ) << "\n";
    }
}


int main( void )
{
    //testGetSignedPermutations();
    //testGetSignedPermutations( { 0,0,2 } );
    /* using short int actually seems to be the fastest, faster than even
     * unsigned char. Seems like automatic SIMD optimizations aren't working
     * very well */
    std::vector< std::array< short int, 3 > > const bondSetSc = {
        {2,0,0}, {2,1,0}, {2,1,1}, {2,2,1}, {3,0,0}, {3,1,0}
    };
    auto const bonds = getSignedPermutations<3>( bondSetSc );
    std::cout << "All " << bonds.size() << " bonds:";
    for ( size_t i = 0u; i < bonds.size(); ++i )
    {
        if ( i % 6 == 0 )
            std::cout << std::endl;
        std::cout
        << std::setw(4) << i << ":("
        << std::setw(2) << (int) bonds[i][0] << ","
        << std::setw(2) << (int) bonds[i][1] << ","
        << std::setw(2) << (int) bonds[i][2] << ")"
        << ( i < bonds.size() - 1 ? "," : "\n" );
    }
    std::cout << std::endl;

    /* print table */
    auto const excludes = createExclusionTable<3>( bonds ); // , decltype( bonds )::value_type::value_type
    std::cout << "Table for bond i excludes bond j:\n";
    unsigned int const nRows = 40u, nCols = 40u;
    //unsigned int const nRows = bonds.size(), nCols = bonds.size();
    std::cout << "i\\j|";
    for ( auto j = 0u; j < nCols; ++j )
        std::cout << std::setw(2) << j % 10;
    std::cout << "\n" << std::setw(4+2*nCols) << std::setfill('-') << "" << std::setfill(' ') << "\n";
    for ( auto i = 0u; i < nRows; ++i )
    {
        std::cout << std::setw(3) << i << "|";
        for ( auto j = 0u; j < nCols; ++j )
            std::cout << std::setw(2) << ( excludes[i][j] ? "x" : " " );
        std::cout << std::endl;
    }
    /* Count how many bonds (including itself) each bond forbids */
    /**
     *   0:14,   1:14,   2:14,   3:14,   4:14,   5:14,
     *   6:15,   7:15,   8:15,   9:15,  10:15,  11:15,
     *  12:15,  13:15,  14:15,  15:15,  16:15,  17:15,
     *  18:15,  19:15,  20:15,  21:15,  22:15,  23:15,
     *  24:15,  25:15,  26:15,  27:15,  28:15,  29:15,
     *  30:14,  31:14,  32:14,  33:14,  34:14,  35:14,
     *  36:14,  37:14,  38:14,  39:14,  40:14,  41:14,
     *  42:14,  43:14,  44:14,  45:14,  46:14,  47:14,
     *  48:14,  49:14,  50:14,  51:14,  52:14,  53:14,
     *  54:10,  55:10,  56:10,  57:10,  58:10,  59:10,
     *  60:10,  61:10,  62:10,  63:10,  64:10,  65:10,
     *  66:10,  67:10,  68:10,  69:10,  70:10,  71:10,
     *  72:10,  73:10,  74:10,  75:10,  76:10,  77:10,
     *  78:14,  79:14,  80:14,  81:14,  82:14,  83:14,
     *  84:12,  85:12,  86:12,  87:12,  88:12,  89:12,
     *  90:12,  91:12,  92:12,  93:12,  94:12,  95:12,
     *  96:12,  97:12,  98:12,  99:12, 100:12, 101:12,
     * 102:12, 103:12, 104:12, 105:12, 106:12, 107:12
     * In the worst case one bond forbids 15 other bonds
     */
    unsigned int nMaxForbidden = 0;
    for ( auto i = 0u; i < excludes.size(); ++i )
    {
        if ( i % 6 == 0 )
            std::cout << std::endl;
        std::cout << std::setw(3) << i << ":";
        unsigned int nForbidden = 0u;
        for ( auto j = 0u; j < excludes[i].size(); ++j )
            nForbidden += excludes[i][j];
        std::cout << std::setw(1) << nForbidden << ", ";
        nMaxForbidden = std::max( nMaxForbidden, nForbidden );
    }
    std::cout << "\nIn the worst case one bond forbids " << nMaxForbidden << " other bonds" << std::endl;

    /**
     * compressedExcludes was padded to: 16 so it has a total of 1728 elements in 1728 Bytes:
     *    0:   0   6   8  14  16  30  32  34  36  78  84  86  92  94 255 255
     *    1:   1   7   9  15  17  31  33  35  37  79  85  87  93  95 255 255
     *    2:   2  10  11  18  20  38  39  42  43  80  88  89  96  98 255 255
     *    3:   3  12  13  19  21  40  41  44  45  81  90  91  97  99 255 255
     *    4:   4  22  23  26  27  46  47  48  49  82 100 101 104 105 255 255
     *    5:   5  24  25  28  29  50  51  52  53  83 102 103 106 107 255 255
     *    6:   0   6  10  14  16  30  34  38  42  54  58  78  84  92  94 255
     *    7:   1   7  11  15  17  31  35  39  43  55  59  79  85  93  95 255
     *    8:   0   8  12  14  16  32  36  40  44  56  60  78  86  92  94 255
     *    9:   1   9  13  15  17  33  37  41  45  57  61  79  87  93  95 255
     *   10:   2   6  10  18  20  30  34  38  42  54  58  80  88  96  98 255
     *   11:   2   7  11  18  20  31  35  39  43  55  59  80  89  96  98 255
     *   12:   3   8  12  19  21  32  36  40  44  56  60  81  90  97  99 255
     *   13:   3   9  13  19  21  33  37  41  45  57  61  81  91  97  99 255
     *   14:   0   6   8  14  22  30  32  46  48  62  64  78  84  86  92 255
     *   15:   1   7   9  15  23  31  33  47  49  63  65  79  85  87  93 255
     *   16:   0   6   8  16  24  34  36  50  52  66  68  78  84  86  94 255
     *   17:   1   7   9  17  25  35  37  51  53  67  69  79  85  87  95 255
     *   18:   2  10  11  18  26  38  39  46  47  70  71  80  88  89  96 255
     *   19:   3  12  13  19  27  40  41  48  49  72  73  81  90  91  97 255
     *   20:   2  10  11  20  28  42  43  50  51  74  75  80  88  89  98 255
     *   21:   3  12  13  21  29  44  45  52  53  76  77  81  90  91  99 255
     *   22:   4  14  22  26  27  30  32  46  48  62  64  82 100 104 105 255
     *   23:   4  15  23  26  27  31  33  47  49  63  65  82 101 104 105 255
     *   24:   5  16  24  28  29  34  36  50  52  66  68  83 102 106 107 255
     *   25:   5  17  25  28  29  35  37  51  53  67  69  83 103 106 107 255
     *   26:   4  18  22  23  26  38  39  46  47  70  71  82 100 101 104 255
     *   27:   4  19  22  23  27  40  41  48  49  72  73  82 100 101 105 255
     *   28:   5  20  24  25  28  42  43  50  51  74  75  83 102 103 106 255
     *   29:   5  21  24  25  29  44  45  52  53  76  77  83 102 103 107 255
     *   30:   0   6  10  14  22  30  38  46  54  62  70  78  84  92 255 255
     *   31:   1   7  11  15  23  31  39  47  55  63  71  79  85  93 255 255
     *   32:   0   8  12  14  22  32  40  48  56  64  72  78  86  92 255 255
     *   33:   1   9  13  15  23  33  41  49  57  65  73  79  87  93 255 255
     *   34:   0   6  10  16  24  34  42  50  58  66  74  78  84  94 255 255
     *   35:   1   7  11  17  25  35  43  51  59  67  75  79  85  95 255 255
     *   36:   0   8  12  16  24  36  44  52  60  68  76  78  86  94 255 255
     *   37:   1   9  13  17  25  37  45  53  61  69  77  79  87  95 255 255
     *   38:   2   6  10  18  26  30  38  46  54  62  70  80  88  96 255 255
     *   39:   2   7  11  18  26  31  39  47  55  63  71  80  89  96 255 255
     *   40:   3   8  12  19  27  32  40  48  56  64  72  81  90  97 255 255
     *   41:   3   9  13  19  27  33  41  49  57  65  73  81  91  97 255 255
     *   42:   2   6  10  20  28  34  42  50  58  66  74  80  88  98 255 255
     *   43:   2   7  11  20  28  35  43  51  59  67  75  80  89  98 255 255
     *   44:   3   8  12  21  29  36  44  52  60  68  76  81  90  99 255 255
     *   45:   3   9  13  21  29  37  45  53  61  69  77  81  91  99 255 255
     *   46:   4  14  18  22  26  30  38  46  54  62  70  82 100 104 255 255
     *   47:   4  15  18  23  26  31  39  47  55  63  71  82 101 104 255 255
     *   48:   4  14  19  22  27  32  40  48  56  64  72  82 100 105 255 255
     *   49:   4  15  19  23  27  33  41  49  57  65  73  82 101 105 255 255
     *   50:   5  16  20  24  28  34  42  50  58  66  74  83 102 106 255 255
     *   51:   5  17  20  25  28  35  43  51  59  67  75  83 103 106 255 255
     *   52:   5  16  21  24  29  36  44  52  60  68  76  83 102 107 255 255
     *   53:   5  17  21  25  29  37  45  53  61  69  77  83 103 107 255 255
     *   54:   6  10  30  38  46  54  62  70  84  88 255 255 255 255 255 255
     *   55:   7  11  31  39  47  55  63  71  85  89 255 255 255 255 255 255
     *   56:   8  12  32  40  48  56  64  72  86  90 255 255 255 255 255 255
     *   57:   9  13  33  41  49  57  65  73  87  91 255 255 255 255 255 255
     *   58:   6  10  34  42  50  58  66  74  84  88 255 255 255 255 255 255
     *   59:   7  11  35  43  51  59  67  75  85  89 255 255 255 255 255 255
     *   60:   8  12  36  44  52  60  68  76  86  90 255 255 255 255 255 255
     *   61:   9  13  37  45  53  61  69  77  87  91 255 255 255 255 255 255
     *   62:  14  22  30  38  46  54  62  70  92 100 255 255 255 255 255 255
     *   63:  15  23  31  39  47  55  63  71  93 101 255 255 255 255 255 255
     *   64:  14  22  32  40  48  56  64  72  92 100 255 255 255 255 255 255
     *   65:  15  23  33  41  49  57  65  73  93 101 255 255 255 255 255 255
     *   66:  16  24  34  42  50  58  66  74  94 102 255 255 255 255 255 255
     *   67:  17  25  35  43  51  59  67  75  95 103 255 255 255 255 255 255
     *   68:  16  24  36  44  52  60  68  76  94 102 255 255 255 255 255 255
     *   69:  17  25  37  45  53  61  69  77  95 103 255 255 255 255 255 255
     *   70:  18  26  30  38  46  54  62  70  96 104 255 255 255 255 255 255
     *   71:  18  26  31  39  47  55  63  71  96 104 255 255 255 255 255 255
     *   72:  19  27  32  40  48  56  64  72  97 105 255 255 255 255 255 255
     *   73:  19  27  33  41  49  57  65  73  97 105 255 255 255 255 255 255
     *   74:  20  28  34  42  50  58  66  74  98 106 255 255 255 255 255 255
     *   75:  20  28  35  43  51  59  67  75  98 106 255 255 255 255 255 255
     *   76:  21  29  36  44  52  60  68  76  99 107 255 255 255 255 255 255
     *   77:  21  29  37  45  53  61  69  77  99 107 255 255 255 255 255 255
     *   78:   0   6   8  14  16  30  32  34  36  78  84  86  92  94 255 255
     *   79:   1   7   9  15  17  31  33  35  37  79  85  87  93  95 255 255
     *   80:   2  10  11  18  20  38  39  42  43  80  88  89  96  98 255 255
     *   81:   3  12  13  19  21  40  41  44  45  81  90  91  97  99 255 255
     *   82:   4  22  23  26  27  46  47  48  49  82 100 101 104 105 255 255
     *   83:   5  24  25  28  29  50  51  52  53  83 102 103 106 107 255 255
     *   84:   0   6  14  16  30  34  54  58  78  84  92  94 255 255 255 255
     *   85:   1   7  15  17  31  35  55  59  79  85  93  95 255 255 255 255
     *   86:   0   8  14  16  32  36  56  60  78  86  92  94 255 255 255 255
     *   87:   1   9  15  17  33  37  57  61  79  87  93  95 255 255 255 255
     *   88:   2  10  18  20  38  42  54  58  80  88  96  98 255 255 255 255
     *   89:   2  11  18  20  39  43  55  59  80  89  96  98 255 255 255 255
     *   90:   3  12  19  21  40  44  56  60  81  90  97  99 255 255 255 255
     *   91:   3  13  19  21  41  45  57  61  81  91  97  99 255 255 255 255
     *   92:   0   6   8  14  30  32  62  64  78  84  86  92 255 255 255 255
     *   93:   1   7   9  15  31  33  63  65  79  85  87  93 255 255 255 255
     *   94:   0   6   8  16  34  36  66  68  78  84  86  94 255 255 255 255
     *   95:   1   7   9  17  35  37  67  69  79  85  87  95 255 255 255 255
     *   96:   2  10  11  18  38  39  70  71  80  88  89  96 255 255 255 255
     *   97:   3  12  13  19  40  41  72  73  81  90  91  97 255 255 255 255
     *   98:   2  10  11  20  42  43  74  75  80  88  89  98 255 255 255 255
     *   99:   3  12  13  21  44  45  76  77  81  90  91  99 255 255 255 255
     *  100:   4  22  26  27  46  48  62  64  82 100 104 105 255 255 255 255
     *  101:   4  23  26  27  47  49  63  65  82 101 104 105 255 255 255 255
     *  102:   5  24  28  29  50  52  66  68  83 102 106 107 255 255 255 255
     *  103:   5  25  28  29  51  53  67  69  83 103 106 107 255 255 255 255
     *  104:   4  22  23  26  46  47  70  71  82 100 101 104 255 255 255 255
     *  105:   4  22  23  27  48  49  72  73  82 100 101 105 255 255 255 255
     *  106:   5  24  25  28  50  51  74  75  83 102 103 106 255 255 255 255
     *  107:   5  24  25  29  52  53  76  77  83 102 103 107 255 255 255 255
     */
    auto const compressedExcludes = compressExclusionTable< 3, unsigned char >( excludes );
    auto const nPadded = compressedExcludes.size() / excludes.size();
    std::cout << "compressedExcludes was padded to: " << nPadded << " so it has a total of " << compressedExcludes.size() << " elements in " << compressedExcludes.size() * sizeof( compressedExcludes[0] ) << " Bytes:\n";
    for ( auto i = 0u; i < excludes.size(); ++i )
    {
        std::cout << std::setw(3) << i << ": ";
        for ( auto j = 0u; j < nPadded; ++j )
            std::cout << std::setw(3) << (int) compressedExcludes[ i * nPadded + j ] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;

    /* actually do algorithm */
    /* unsigned short is fastest, don't use unsighed char! */
    //findMaxNeighborsNaive<3>( bonds, 1 );
    //findMaxNeighborsNaiveInlined<3>( bonds, 1 );
    //findMaxNeighborsNaiveInlinedMonteCarlo<3>( bonds, 1 );
    //findMaxNeighborsNaiveInlinedOnlyIncreasing<3>( bonds, 1 );
    auto bondsToBeSorted = bonds;
    //findMaxNeighborsNaiveInlinedOnlyIncreasingTrackUpperBounds<3>( &bondsToBeSorted, 1 );
    /**
     * Tested 8 592 577 954 at a rate of 840080.2 configs/s in 10210.93s => 2h 50m 28.2829 1156s
     * max length: 20 at {0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43}
     * 1:6.878e-07%, 2:4.735e-05%, 3:0.001494%, 4:0.02828%, 5:0.3548%, 6:3.088%, 7:19.06%, 8:15.71%, 9:23.66%, 10:20.48%, 11:11.44%, 12:4.483%, 13:1.298%, 14:0.3107%, 15:0.06288%, 16:0.01243%, 17:0.004329%, 18:0.001666%, 19:9.423e-05%,
     */
    auto const maxConfig = findMaxNeighborsNaiveInlinedOnlyIncreasingTrackUpperBounds<3>( &bondsToBeSorted, 1, -1 /* timeout */ );
    saveConfiguration<3>( bondsToBeSorted, maxConfig, "maxConfig.bfm" );
}
