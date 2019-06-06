/**
 * When not reordering the neighbor information as struct of array,
 * then increasing this leads to performance degradataion!
 * But currently, as the reordering is implemented, it just leads to
 * higher memory usage.
 * In the 3D case more than 20 makes no sense for the standard bond vector
 * set, as the volume exclusion plus the bond vector set make 20 neighbors
 * the maximum. In real use cases 8 are already very much / more than sufficient.
 */
#pragma once
#define MAX_CONNECTIVITY 8
/* stores amount and IDs of neighbors for each monomer */
struct MonomerEdges
{
    uint32_t size; // could also be uint8_t as it is limited by MAX_CONNECTIVITY
    uint32_t neighborIds[ MAX_CONNECTIVITY ];
};
