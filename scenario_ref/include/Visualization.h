//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_VISUALIZATION_H
#define PROJECT_VISUALIZATION_H

#ifdef VISUALIZATION_ENABLED
#include <opencv2/opencv.hpp>
#include "Scenario.h"
#include "BaseVisualizationEngine.h"

using namespace cv;


class Visualization : public BaseVisualizationEngine {
public:
    Scenario *scenario;
    
    explicit Visualization(std::shared_ptr<BaseScenario> &map) :
        scenario(dynamic_cast<Scenario *>(map.get())), video(nullptr), imageBasePath(""), BaseVisualizationEngine()
         {
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

    Point2d junctionPoint(Junction *j);
    Point2d directionVector(Junction::Direction direction);

public:
    /**
     * open video file - after this call all calls to 'render_image' will be saved in video.
     * @param fn file name
     * @param fps fps
     */
    void setVideoPath(std::string &fn, double fps) override;


    /**
     * set image path. render_image will save image for each frame.
     * @param fn image base_file name. '%d.jpg' will be appended before saving file.
     */
    void setImageBasePath(std::string &fn) override;

    /**
     * close video file
     */
    void close() override;

    /**
     * init image and variables
     */
    void initialize() override;

    /**
     * render a state
     * @return
     */

    void render_image() override;


};
#endif

#endif //PROJECT_VISUALIZATION_H
