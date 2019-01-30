
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
    auto end = getCars().end();
    auto begin = getCars().begin();

    auto it = std::lower_bound(begin, end, trafficObject, Car::Cmp);

    if (it != begin)
        result.back = *(it - 1);

    // skip ego vehicle
    while(it != end && *it == trafficObject)
        it++;

    if(it != end) result.front = *it;


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
