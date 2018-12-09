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

    Point2d junctionPoint(Junction *j);
    Point2d directionVector(Junction::Direction direction);

public:

    /**
     * open video file - after this call all calls to 'render_image' will be saved in video.
     * @param fn file name
     * @param fps fps
     */
    void open(std::string &fn, double fps = 1);

    /**
     * close video file
     */
    void close();

    /**
     * init image and variables
     */
    void initialize();

    /**
     * render a state
     * @return
     */

    Mat render_image();


};


#endif //PROJECT_VISUALIZATION_H
