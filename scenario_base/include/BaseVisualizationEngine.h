//
// Created by oke on 17.12.18.
//

#ifndef TRAFFIC_SIM_VISUALIZATIONBASE_H
#define TRAFFIC_SIM_VISUALIZATIONBASE_H

#include <memory>
#include "BaseScenario.h"

class BaseVisualizationEngine {
public:

    /**
     * open video file - after this call all calls to 'render_image' will be saved in video.
     * @param fn file name
     * @param fps fps
     */
    virtual void setVideoPath(std::string &fn, double fps) = 0;


    /**
     * set image path. render_image will save image for each frame.
     * @param fn image base_file name. '%d.jpg' will be appended before saving file.
     */
    virtual void setImageBasePath(std::string &fn) = 0;

    /**
     * close video file
     */
    virtual void close() = 0;

    /**
     * init image and variables
     */
    virtual void initialize() = 0;

    /**
     * render a state
     * @return
     */

    virtual void render_image() = 0;

};

#endif //TRAFFIC_SIM_VISUALIZATIONBASE_H
