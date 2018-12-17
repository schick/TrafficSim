//
// Created by oke on 07.12.18.
//

#include "Visualization_id.h"


void Visualization_id::setVideoPath(std::string &fn, double fps) {
    assert(video == nullptr && imageBasePath.length() == 0);
    video = std::make_shared<VideoWriter>(fn, VideoWriter::fourcc('M','J','P','G'), fps, Size(base_image.size()));
}

void Visualization_id::setImageBasePath(std::string &fn) {
    assert(video == nullptr && imageBasePath.length() == 0);
    imageBasePath = fn;
}

void Visualization_id::close() {
    if (video != nullptr) {
        video->release();
        video = nullptr;
    }
}

void Visualization_id::initialize() {
    // calculate borders
    Point2d min(10000, 10000), max(-10000, -10000);
    for (auto &junction : scenario->junctions) {
        if (junction.x < min.x) {
            min.x = junction.x;
        }
        if (junction.y < min.y) {
            min.y = junction.y;
        }
        if (junction.x > max.x) {
            max.x = junction.x;
        }
        if (junction.y > max.y) {
            max.y = junction.y;
        }
    }
    // fix pixel_per_m to match max_size
    Point2d base_image_size = (max - min) * pixel_per_m +
                              Point2d(junction_radius * pixel_per_m * 2, junction_radius * pixel_per_m * 2);
    double f1 = max_size.x / (float) base_image_size.x;
    double f2 = max_size.y / (float) base_image_size.y;
    pixel_per_m = (f1 < f2 ? f1 : f2) * pixel_per_m;

    // calculate offset
    base_image_size = (max - min) * pixel_per_m +
                      Point2d(junction_radius * pixel_per_m * 2, junction_radius * pixel_per_m * 2);
    offset = min * pixel_per_m + Point2d(junction_radius * pixel_per_m, junction_radius * pixel_per_m);

    // create base_image
    base_image = Mat::zeros(Point(base_image_size), CV_8UC3);
    base_image.setTo(Scalar(175, 175, 175));

    for (auto &r: scenario->roads) {
        int count = 0;
        for(int lid : r.lanes) if (lid != -1) count += 2;
        line(base_image, junctionPoint(scenario->junctions.at(r.from)), junctionPoint(scenario->junctions.at(r.to)),
             Scalar(50, 50, 50), (int) (lane_width * count * pixel_per_m));
    }

    // print junctions
    for (auto &junction : scenario->junctions) {
        circle(base_image, junctionPoint(junction),
               (int) (35. * pixel_per_m / 2.),
               Scalar(100, 100, 100), -1);
    }
}

Point2d Visualization_id::junctionPoint(Junction_id &j) {
    return Point2d((int) (j.x * pixel_per_m), (int) (j.y * pixel_per_m)) + offset;
}

Point2d Visualization_id::directionVector(Junction_id::Direction direction) {
    switch(direction) {
        case Junction_id::Direction::NORTH:
            return Point2d(0, -1);
        case Junction_id::Direction::EAST:
            return Point2d(1, 0);
        case Junction_id::Direction::SOUTH:
            return Point2d(0, 1);
        case Junction_id::Direction::WEST:
            return Point2d(-1, 0);
    };

    //Error assertion
}

Point2d getOuterLaneDirection_id(Point2d point) {
    if (point == Point2d(0,-1)) {
        return Point2d(1,0);

    } else if (point == Point2d(1,0)) {
        return Point2d(0,1);

    } else if(point == Point2d(0,1)) {
        return Point2d(-1,0);

    } else if(point == Point2d(-1,0)) {
        return Point2d(0, -1);
    }

    //Error assertion
}

void Visualization_id::render_image() {
    numFrame++;
    Mat image = base_image.clone();

    for(auto &j: scenario->junctions) {
        for (int i = 0; i < 4; i++) {
            if (j.incoming[i] != -1)
                circle(image, junctionPoint(j) + (directionVector(static_cast<Junction_id::Direction>(i)) * junction_radius * pixel_per_m), pixel_per_m * 1,
                       (scenario->signals[j.signal_begin + j.current_signal_id].direction == i) ? Scalar(0, 255, 0) : Scalar(0, 0, 255), -1);
        }
    }

    for(auto &car : scenario->cars) {
        Point2d from = junctionPoint(scenario->junctions[scenario->roads[scenario->lanes.at(car.getLane()).road].from]);
        Point2d to = junctionPoint(scenario->junctions[scenario->roads[scenario->lanes.at(car.getLane()).road].to]);
        Point2d dir = (to - from);

        //get directions
        dir = dir / sqrt(pow(dir.x, 2) + pow(dir.y, 2));
        Point2d outerLaneDir = getOuterLaneDirection_id(dir);

        //scale directions
        auto scaledDir = dir * pixel_per_m;
        auto scaledOuterLaneDir = outerLaneDir * pixel_per_m;

        //calculate offsets
        Point2d carOffset = scaledDir * (car.getPosition() - car.length / 2);
        Point2d laneOffset = scaledOuterLaneDir * ((double) scenario->lanes.at(car.getLane()).lane_num) * lane_width;
        Point2d laneBorderOffset = scaledOuterLaneDir * lane_border;
        Point2d carSizeOffset = scaledDir * car_length + scaledOuterLaneDir * car_width;

        Point2d start = from + carOffset + laneOffset + laneBorderOffset;
        Point2d end = start + carSizeOffset;

        Point2d front_middle = end - 0.5 * car_width * scaledOuterLaneDir;

        rectangle(image, start, end, Scalar(0, 0, 255), -1);

        circle(image, front_middle, (int) (car_width / 4 * pixel_per_m), Scalar(0, 255, 255), -1);
        putText(image, std::to_string(car.id), start, 0, pixel_per_m / 14, Scalar(255, 255, 0), 2);
    }

    // no need to flip image -> linkshändisches koordinatensystem
    if (video != nullptr) {
        video->write(image);
    }
    if (imageBasePath.length() > 0){
        std::string fn = imageBasePath + std::to_string(numFrame) + ".jpg";
        imwrite(fn, image);
    }
}

