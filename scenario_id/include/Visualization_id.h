//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_VISUALIZATION_ID_H
#define PROJECT_VISUALIZATION_ID_H

#ifdef VISUALIZATION_ENABLED
#include <opencv2/opencv.hpp>
#include "model/Scenario_id.h"
#include "BaseVisualizationEngine.h"
using namespace cv;

class Visualization_id : public BaseVisualizationEngine {
public:
    std::shared_ptr<Scenario_id> scenario;
    
    explicit Visualization_id(std::shared_ptr<BaseScenario> &map) : scenario(dynamic_cast<Scenario_id*>(map.get())), video(nullptr),
        imageBasePath("") {
        initialize();
    };

    /*
     * display settings
     */
    const double car_width = 2.0;
    const double car_length= 5.0;

    const double junction_radius = 35. / 2.;
    Point2d max_size = Point(2000, 2000);
    const double lane_width = 4.0;
    const double lane_border = 1.0;

private:
    /**
     * helper variables
     */
     uint64_t numFrame = 0;
    Mat base_image = Mat::zeros(1000, 1000, CV_8UC3 );
    Point2d offset = Point(500, 500);
    double pixel_per_m = 20;
    std::shared_ptr<VideoWriter> video;
    std::string imageBasePath;

    Point2d junctionPoint(Junction_id &j);
    Point2d directionVector(Junction_id::Direction direction);

public:

    /**
     * open video file - after this call all calls to 'render_image' will be saved in video.
     * @param fn file name
     * @param fps fps
     */
    void setVideoPath(std::string &fn, double fps);


    /**
     * set image path. render_image will save image for each frame.
     * @param fn image base_file name. '%d.jpg' will be appended before saving file.
     */
    void setImageBasePath(std::string &fn);

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

    void render_image();


};
#endif
#endif //PROJECT_VISUALIZATION_H
