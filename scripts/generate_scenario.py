#!/usr/bin/python
from __future__ import print_function
import argparse
import json
import os

import numpy as np


def get_direction(f, t):
    if f["y"] < t["y"]:
        return 2
    elif f["y"] > t["y"]:
        return 0
    elif f["x"] < t["x"]:
        return 1
    elif f["x"] > t["x"]:
        return 3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("-g", "--grid_size", type=int, default=20)
    parser.add_argument("-j", "--junction_probability", type=float, default=0.8)
    parser.add_argument("-r", "--road_probability", type=float, default=0.8)
    parser.add_argument("-c", "--mean_cars_per_road", type=int, default=6)
    parser.add_argument("-t", "--time_steps", type=int, default=100)

    options = parser.parse_args()
    junctions_exits = np.random.uniform(size=(options.grid_size, options.grid_size)) > (1 - options.junction_probability)
    junctions = np.array(np.where(junctions_exits)).T

    possible_roads = []
    for i1 in range(len(junctions)):
        j1 = junctions[i1]
        for j2 in junctions[i1 + 1:]:
            if np.sum((j1 -j2) == 0) == 1:
                l = np.sqrt(np.sum(np.power(j1 - j2, 2)))
                if l > 1:
                    a = [j2 + (j1 - j2) / l * i for i in range(1, int(l))]
                    if np.sum([junctions_exits[int(x), int(y)] for x, y in a]) != 0:
                        continue
                possible_roads.append((j1, j2))

    roads = np.array(possible_roads)[np.where(np.random.uniform(size=len(possible_roads)) > (1 - options.road_probability))[0]]

    junctions = {i: {"id": int(i), "x": int(x), "y": int(y), "dirs": []} for i, (x, y) in enumerate(junctions)}

    junction_by_pos = {str((j["x"], j["y"])): j for j in junctions.values()}
    roads = [{
        "junction1": junction_by_pos[str(tuple(j1))]["id"], "junction2": junction_by_pos[str(tuple(j2))]["id"],
        "lanes": int(np.random.randint(1, 4)), "limit": int(np.random.choice([30, 50, 70, 90, 100, 120], 1)[0])
    } for j1, j2 in roads]

    roads = list(roads)

    for road in roads:
        junction1 = junctions[road["junction1"]]
        junction2 = junctions[road["junction2"]]
        dir = get_direction(junction1, junction2)
        junction1["dirs"].append(dir)
        junction2["dirs"].append((dir + 2) % 4)

    remove = []
    for junction in junctions.values():
        junction["signals"] = [{"dir": dir, "time": np.random.randint(4, 16)} for dir in junction["dirs"]]
        del junction["dirs"]
        if len(junction["signals"]) == 0:
            remove.append(junction)

    for junction in remove:
        del junctions[junction["id"]]

    def create_random_car(i):
        road = np.random.choice(roads, 1)[0]
        f_t = np.array([road["junction1"], road["junction2"]])
        np.random.shuffle(f_t)
        return {
            "id": i,
            "target_velocity": float(np.random.randint(70, 150)),
            "max_acceleration": float(np.random.uniform(1, 3)),
            "target_deceleration": float(np.random.uniform(2, 4)),
            "min_distance": float(np.random.uniform(2, 4)),
            "target_headway": float(np.random.uniform(1, 3)),
            "politeness": float(np.random.uniform(0.1, 0.5)),
            "start": {
                "from": int(f_t[0]),
                "to": int(f_t[1]),
                "lane": int(np.random.randint(0, road["lanes"])),
                "distance": float(np.random.uniform(0, abs(junctions[f_t[0]]["x"] - junctions[f_t[1]]["x"]) +
                             abs(junctions[f_t[0]]["y"] - junctions[f_t[1]]["y"])) * 100)

            },
            "route": list([int(i) for i in np.random.randint(0, 4, size=np.random.randint(2, 5))])

        }
    cars = [create_random_car(i) for i in range(options.mean_cars_per_road * len(roads))]

    print("Grid: %dx%d" % (options.grid_size, options.grid_size))
    print("Junctions: ", len(junctions))
    print("Roads: ", len(roads))
    print("Cars: ", len(cars))
    with open(os.path.expanduser(options.output), "w") as f:
        json.dump({
            "time_steps": options.time_steps,
            "junctions": list(junctions.values()),
            "roads": roads,
            "cars": cars,
        }, f, indent=4)



