
#include "model/Lane.h"
#include "model/Car.h"

void Lane::sortCars() {
    if (!isSorted) {
        std::sort(mCars.begin(), mCars.end(), Car::Cmp());
        isSorted = true;
    }
}