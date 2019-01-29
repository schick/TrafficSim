//
// Created by oke on 29.01.19.
//

#include "model/TrafficLight.h"
#include "model/Lane.h"

// traffic lights have -1 id, because traffic lights are always at the end of road.
TrafficLight::TrafficLight(Lane *lane) : TrafficObject((uint64_t ) -1, 0, lane->length - 35. / 2.) {}