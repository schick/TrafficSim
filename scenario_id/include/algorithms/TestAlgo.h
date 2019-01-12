//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_TESTALGO_H
#define PROJECT_TESTALGO_H

#include "algorithm"
#include "AdvanceAlgorithm.h"
#include "Car_id.h"
#include "Scenario_id.h"
#include "Visualization_id.h"
#include <list>
#include <map>
#include "AlgorithmWrapper.h"


#define MAX(a, b) (a < b ? b : a)
#define MIN(a, b) (a < b ? a : b)

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

template <typename ObjectType, ObjectType Z>
class BucketContainer;

template <typename ObjectType, ObjectType Z>
class Bucket {
    friend class BucketContainer<ObjectType, Z>;
private:
    size_t buffer_size;
    ObjectType *objects;
    size_t size;

    CUDA_HOSTDEV void _resize(size_t new_size) {
        ObjectType *new_objects = (ObjectType *) malloc(new_size * sizeof(ObjectType));
        for(int i=0; i < 20; i++) if (new_objects == nullptr) new_objects = (ObjectType *) malloc(new_size * sizeof(ObjectType));
        assert(new_objects != nullptr);
        size_t i = 0;
        for(; i < MIN(buffer_size, new_size); i++)
            new_objects[i] = objects[i];
        for(; i < new_size; i++) {
            assert(i < new_size && new_objects != nullptr);
            new_objects[i] = Z;
        }
        buffer_size = new_size;
        if(objects != nullptr)
            free(objects);
        objects = new_objects;
    }

public:
    CUDA_HOSTDEV Bucket() : objects(nullptr), buffer_size(0), size(0) {}

    CUDA_HOSTDEV Bucket(size_t size) {
        this->size = 0;
        this->buffer_size = size;
        objects = (ObjectType*) malloc(size * sizeof(ObjectType));
        assert(objects != nullptr);
        for(int i=0; i < size; i++) objects[i] = Z;
    }

    // move constructor
    CUDA_HOSTDEV Bucket(Bucket&& other) {
        objects = other.objects;
        buffer_size = other.size;
        other.size = 0;
        other.objects = nullptr;
    }

    // copy constructor
    CUDA_HOSTDEV Bucket(Bucket& other) {
        printf("copying bucket...");
        if (buffer_size != other.size && objects != nullptr) {
            free(objects);
            buffer_size = other.size;
            objects = (ObjectType*) malloc(buffer_size * sizeof(ObjectType));
            assert(objects != nullptr);
        }
        for(int i=0; i < buffer_size; i++) objects[i] = other.objects[i];
    }

    // destructor
    CUDA_HOSTDEV ~Bucket() {
        free(objects);
    }

    CUDA_HOSTDEV size_t getSize() {
        return size;
    }

    CUDA_HOSTDEV void resize(size_t new_size) {
        if(new_size <= buffer_size) {
            if (new_size < buffer_size / 4) {
                _resize(new_size * 2);
            }
        } else {
            _resize(MAX(MAX(buffer_size * 2, 15), new_size));
        }
        size = new_size;
    }

    CUDA_HOSTDEV ObjectType &operator[](size_t index) {
        if(index >= size) printf("%lu\n", index);
        assert(index < size);
        return objects[index];
    }



};


template <typename ObjectType, ObjectType Z>
class BucketContainer {

public:
    typedef Bucket<ObjectType, Z> BucketT;

    CUDA_HOSTDEV BucketContainer(size_t num_buckets, size_t default_size) {
        buckets = (BucketT *) malloc(num_buckets * sizeof(BucketT));
        assert(buckets != nullptr);
        this->num_buckets = num_buckets;
        if (default_size == 0)
            for (int i = 0; i < num_buckets; i++) new(&buckets[i])BucketT();
        else
            for(int i=0; i < num_buckets; i++) new(&buckets[i])BucketT(default_size);
    }

    CUDA_HOSTDEV BucketT &operator[](size_t index) {
        return buckets[index];
    }

    CUDA_HOSTDEV size_t getNumBuckets() {
        return num_buckets;
    }

    template<typename Cmp>
    void sort(Cmp cmp) {
        for (int i = 0; i < num_buckets; i++) {
            auto &bucket = buckets[i];
            if (bucket.getSize() > 1) {
                std::sort(bucket.objects, bucket.objects + bucket.getSize(), cmp);
            }
        }
    }

    template<typename Cmp>
    CUDA_HOSTDEV void testSort(Cmp cmp) {
        for(int i = 0; i < getNumBuckets(); i++) {
            if (buckets[i].getSize() == 0) continue;
            //printf("Lane %d\n", i);
            for(int j = 0; j < buckets[i].getSize() - 1; j++) {
                //printf("%lu,%lu: %f <= %f (%d, %d)\n", buckets[i][j]->id, buckets[i][j + 1]->id,  buckets[i][j]->x, buckets[i][j + 1]->x,
                //        cmp(buckets[i][j], buckets[i][j + 1]), buckets[i][j]->x <= buckets[i][j + 1]->x);
                assert(cmp(buckets[i][j], buckets[i][j + 1]) && buckets[i][j]->x <= buckets[i][j + 1]->x);
            }
        }
        printf("TESTING: sortTestPassed\n");
    }

    CUDA_HOSTDEV size_t numElements() {
        size_t total = 0;
        for(int i=0; i < num_buckets; i++)
            total += buckets[i].getSize();
        return total;
    }

    template<typename Cmp>
    static void sort_device_bucket(BucketContainer<ObjectType, Z> *container, Cmp cmp, size_t num_buckets);

    template<typename T>
    CUDA_HOSTDEV static BucketContainer<ObjectType, Z> construct(T &scenario);

    template<typename T>
    CUDA_HOST static void construct_device(T *scenario, BucketContainer<ObjectType, Z> *bucketContainer, size_t num_lanes);

private:
    size_t num_buckets;
    BucketT *buckets;


};


class TestAlgo : public AdvanceAlgorithm {

public:
    ADVANCE_ALGO_INIT(TestAlgo, Scenario_id, Visualization_id);

    explicit TestAlgo(std::shared_ptr<BaseScenario> scenario) : AdvanceAlgorithm(scenario) {};

    Scenario_id* getIDScenario() {
        return dynamic_cast<Scenario_id*>(getScenario().get());
    }

    void advance(size_t steps) override;
};

#endif //PROJECT_SEQUENTIALALGORITHM_H
