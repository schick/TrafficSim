
#include "model/Lane.h"
#include "model/Car.h"

void Lane::sortCars() {
    if (!isSorted) {
        std::sort(mCars.begin(), mCars.end(), Car::Cmp);
        isSorted = true;
    }
}

Lane::NeighboringObjects Lane::getNeighboringObjects(Car *trafficObject) {
    Lane::NeighboringObjects result;

    //if null reference or is empty return empty struct
    if (mCars.empty()) {
        return result;
    }

    auto it = std::lower_bound(getCars().begin(), getCars().end(), trafficObject, Car::Cmp);

    if (it != getCars().begin()) {
        result.back = *(it - 1);
    }

    if (getCars().end() == it || *it != trafficObject) {
        if (it != getCars().end()) {
            result.front = *it;
        }
    } else {
        if (it + 1 != getCars().end()) {
            result.front = *(it + 1);
        }
    }

    /**
    if (trafficLight.isRed) {
        if (result.front == nullptr || result.front->x >= trafficLight.x) {
            if (trafficObject->x < trafficLight.x) {
                result.front = &trafficLight;
            }
        }
    }
     */

    return result;
}
