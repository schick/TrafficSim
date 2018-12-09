//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_VISUALIZATION_H
#define PROJECT_VISUALIZATION_H

#include <opencv2/opencv.hpp>
#include <Scenario.h>

using namespace cv;


class Visualization {
public:
    Scenario *scenario;
    
    Visualization(Scenario *map) : scenario(map) {
        video = nullptr;
        initialize();
    };

    /*
     * display settings
     */
    double car_width = 2;
    double junction_radius = 35. / 2.;
    Point2d max_size = Point(2400, 1200);
    double lane_width = 3.75;

private:
    /**
     * helper variables
     */
    Mat base_image = Mat::zeros(1000, 1000, CV_8UC3 );
    Point2d offset = Point(500, 500);
    double pixel_per_m = 20;
    std::shared_ptr<VideoWriter> video;

    Point2d junctionPoint(Junction *j) {
        return Point2d((int) (j->x * pixel_per_m), (int) (j->y * pixel_per_m)) + offset;
    }

public:

    /**
     * open video file - after this call all calls to 'render_image' will be saved in video.
     * @param fn file name
     * @param fps fps
     */
    void open(std::string &fn, double fps = 1) {
        assert(video == nullptr);
        video = std::make_shared<VideoWriter>(fn, VideoWriter::fourcc('M','J','P','G'), fps, Size(base_image.size()));
    }

    /**
     * close video file
     */
    void close() {
        if (video != nullptr) {
            video->release();
            video = nullptr;
        }
    }

    /**
     * init image and variables
     */
    void initialize() {
        // calculate borders
        Point2d min(10000, 10000), max(-10000, -10000);
        for (std::unique_ptr<Junction> &junction : scenario->junctions) {
            if (junction->x < min.x) {
                min.x = junction->x;
            }
            if (junction->y < min.y) {
                min.y = junction->y;
            }
            if (junction->x > max.x) {
                max.x = junction->x;
            }
            if (junction->y > max.y) {
                max.y = junction->y;
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
        base_image.setTo(Scalar(255, 255, 255));

        for (std::unique_ptr<Road> &r: scenario->roads) {
            size_t count = r->lanes.size() * 2;
            line(base_image, junctionPoint(r->from), junctionPoint(r->to),
                 Scalar(50, 50, 50), (int) (lane_width * count * pixel_per_m));
        }

        // print junctions
        for (std::unique_ptr<Junction> &junction : scenario->junctions) {
            circle(base_image, junctionPoint(junction.get()),
                   (int) (35. * pixel_per_m / 2.),
                   Scalar(100, 100, 100), -1);
        }
    }

    Point2d directionVector(Junction::Direction direction) {
        switch(direction) {
            case Junction::Direction::NORTH:
                return Point2d(0, 1);
            case Junction::Direction::EAST:
                return Point2d(1, 0);
            case Junction::Direction::SOUTH:
                return Point2d(0, -1);
            case Junction::Direction::WEST:
                return Point2d(-1, 0);
        };
    }

    /**
     * render a state
     * @return
     */
    
    Mat render_image() {
        Mat image = base_image.clone();

        for(std::unique_ptr<Junction> &j: scenario->junctions) {
            for (int i = 0; i < 4; i++) {
                if (j->incoming[i] != nullptr)
                    circle(image, junctionPoint(j.get()) + (directionVector(static_cast<Junction::Direction>(i)) * junction_radius * pixel_per_m), pixel_per_m * 1,
                       (j->signals[j->current_signal_id].direction == i) ? Scalar(0, 255, 0) : Scalar(0, 0, 255), -1);
            }
        }

        for(std::unique_ptr<Car> &car : scenario->cars) {
            Point2d from = junctionPoint(car->getLane()->road->from);
            Point2d to = junctionPoint(car->getLane()->road->to);
            Point2d dir = (to - from);
            dir = dir / sqrt(dir.x * dir.x + dir.y * dir.y) * pixel_per_m;

            Point2d orth;
            if (dir.x > 0) {
                orth = Point2d(0, 1);
            } else if (dir.x < 0) {
                orth = Point2d(0, -1);
            } else if (dir.y > 0) {
                orth = Point2d(-1, 0);
            } else if (dir.y < 0) {
                orth = Point2d(1, 0);
            }

            orth = pixel_per_m * orth;
            Point2d o = ((float) car->getLane()->lane_id + 0.5) * lane_width * orth;
            circle(image,
                   junctionPoint(car->getLane()->road->from) - o + dir * car->x,
                   pixel_per_m * car_width, Scalar(0, 0, 255), -1);
        }

        Mat image_flip;

        cv::flip(image, image_flip, 0);

        if (video != nullptr) {
            video->write(image_flip);
        }

        return image_flip;
    }


};


#endif //PROJECT_VISUALIZATION_H
