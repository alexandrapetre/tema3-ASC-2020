#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#define blockSize	1024
#include "gpu_hashtable.hpp"
#define load_factor 0.80

__device__ int hash4(int data, int limit) {
	return ((long)abs(data) * primeList2[70] ) % primeList2[96] % limit;
}

/*
* Initializez vectorul de (key, value) cu cheia  = 0
*/
__global__ void kernel_init(hashMap *hashmap, int hashSize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < hashSize)
		hashmap[id].key = KEY_INVALID;
}

/*
*	Pentru fiecare pereche din noul hashmap, verific daca cheia
*	este 0 sau nu, daca este 0 inserez pe pozitia data de functia de hash
*	daca nu este 0, trec la urmatoarea pozitie (hash + 1) % hashSize (parcurc 
*	circular vectorul)
*/

__global__ void kernel_reshape(hashMap *hashmap, hashMap *newHashmap, int hashSize, int newSize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t hashKey;
	uint32_t oldKey;
	uint32_t hash;
	uint32_t inserted = 0;

	if (id >= hashSize)
		return;

	hashKey = hashmap[id].key;
	hash = hash4(hashKey, newSize);

	if (hashKey == KEY_INVALID)
		return;
	
	while (inserted != 1) {
		hash = hash % newSize;
		oldKey = atomicCAS(&newHashmap[hash].key, KEY_INVALID, hashKey);

		if (oldKey == KEY_INVALID || oldKey == hashKey) {
			inserted = 1;
			newHashmap[hash].value = hashmap[id].value;
			return;
		}

		hash = (hash + 1);
	}
	return;
}

/*
*	Inserez in hashmap (key, value) pe care le primesc
*	ca parametrii
*/

__device__ void kernel_insert(hashMap *hashmap, uint32_t key, uint32_t value, uint32_t hashSize)
{
	uint32_t hash;

	hash = hash4(key, hashSize);


	while (true) {
		hash = hash % hashSize;
		uint32_t prev = atomicCAS(&hashmap[hash].key, KEY_INVALID, key);

		if (prev == KEY_INVALID || prev == key) {
			hashmap[hash].value = value;
			return;
		}

		hash = (hash + 1);
	}
}

/*
* 	Cu aceasta functie fiecare thread ia cheia si valoarea indicata
*	de id din vectorul de chei si valori primit ca parametru
*/

__global__ void gpu_insert_kernel(hashMap *hashmap, uint32_t *keys, uint32_t *values, uint32_t numKeys, uint32_t hashSize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t key;
	uint32_t value;

	if (id < numKeys) {
		key = keys[id];
		value = values[id];
		kernel_insert(hashmap, key, value, hashSize);
	}
	return;
}

/*
*	Caut cheia in hashmap si daca o gasesc pe pozitia
*	indicata de functia de hash o adaug in vectorul de
*	valori
*/

__global__ void kernel_get(hashMap *hashmap, uint32_t *keys, uint32_t *values, uint32_t numKeys, uint32_t hashSize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t hash;
	uint32_t hashKey;
	uint32_t oldKey;

	if (id >= numKeys)
		return;
	
	hashKey = keys[id];
	hash = hash4(hashKey, hashSize);
	oldKey = hashmap[hash].key;

	if (oldKey == KEY_INVALID)
		return;

	while(true) {
		hash = hash % hashSize;
		oldKey = hashmap[hash].key;

		if (oldKey == hashKey) {
			values[id] = hashmap[hash].value;
			return;
		}

		if (oldKey == KEY_INVALID) {
			values[id] = KEY_INVALID;
			return;
		}
		
		hash = (hash + 1);
	}
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	uint32_t numBlocks =  size / blockSize;
	hashSize = size;
	pairs = 0;

	if (size % blockSize != 0)
		numBlocks++;

	cudaMalloc(&hashmap, size * sizeof(hashMap));
	kernel_init <<< numBlocks, blockSize >>> (hashmap, hashSize);
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashmap);
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	hashMap *newHashmap;
	uint32_t hashSizeNew = numBucketsReshape;
	uint32_t numBlocks = hashSize /  blockSize;
	uint32_t numBlocksNew = numBucketsReshape / blockSize;

	if (hashSizeNew % blockSize != 0)
		numBlocksNew++;
	
	if (hashSize % blockSize != 0)
		numBlocks++;

	cudaMalloc(&newHashmap, hashSizeNew * sizeof(hashMap));
	kernel_init <<< numBlocksNew, blockSize >>> (newHashmap, hashSizeNew);

	kernel_reshape <<< numBlocks, blockSize >>> (hashmap, newHashmap, hashSize, hashSizeNew);

	cudaDeviceSynchronize();
	cudaFree(hashmap);
	hashmap = newHashmap;
	hashSize = numBucketsReshape;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	uint32_t *deviceKeys;
	uint32_t *deviceValues;
	uint32_t numBlocks;

	numBlocks = numKeys / blockSize;

	if (numKeys % blockSize != 0)
		numBlocks++;

	if (numKeys + pairs > hashSize * load_factor) 
		reshape((uint32_t)((numKeys + pairs) / 0.8));

	cudaMalloc((void **)&deviceKeys, numKeys * sizeof(uint32_t));
	cudaMalloc((void **)&deviceValues, numKeys * sizeof(uint32_t));

	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(uint32_t), cudaMemcpyHostToDevice);

	gpu_insert_kernel <<< numBlocks, blockSize >>> (hashmap, deviceKeys, deviceValues,
		numKeys, hashSize);

	pairs += numKeys;
	cudaDeviceSynchronize();
	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	return true;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	uint32_t *deviceKeys;
	uint32_t *deviceValues;
	int *values;
	uint32_t numBlocks;

	numBlocks = numKeys / blockSize;
	if (numKeys % blockSize != 0)
		numBlocks++;

	cudaMalloc((void **)&deviceKeys,  numKeys * sizeof(uint32_t));
	cudaMalloc((void **)&deviceValues,  numKeys * sizeof(uint32_t));


	values = (int *)malloc(numKeys * sizeof(int));

	cudaMemcpy(deviceKeys, keys,  numKeys * sizeof(uint32_t), cudaMemcpyHostToDevice);
	
	kernel_get <<<numBlocks, blockSize >>> (hashmap, deviceKeys, deviceValues, numKeys, hashSize);
	cudaDeviceSynchronize();

	cudaMemcpy(values, deviceValues,  numKeys * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	return values;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	return (float) pairs / hashSize; // no larger than 1.0f = 100%
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
