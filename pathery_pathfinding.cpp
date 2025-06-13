#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct Node {
    int x, y, cost;
    std::vector<std::pair<int, int>> path;

    bool operator>(const Node& other) const {
        return cost > other.cost;
    }
};

std::vector<std::pair<int, int>> find_path_cpp(
    int width, int height,
    const std::vector<std::pair<int, int>>& walls,
    const std::vector<std::pair<int, int>>& rocks,
    const std::pair<int, int>& start,
    const std::pair<int, int>& finish
) {
    std::set<std::pair<int, int>> wall_set(walls.begin(), walls.end());
    std::set<std::pair<int, int>> rock_set(rocks.begin(), rocks.end());
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    std::set<std::pair<int, int>> visited;

    pq.push({start.first, start.second, 0, {{start.first, start.second}}});

    while (!pq.empty()) {
        Node current = pq.top();
        pq.pop();

        if (current.x == finish.first && current.y == finish.second) {
            return current.path;
        }

        if (visited.count({current.x, current.y})) {
            continue;
        }
        visited.insert({current.x, current.y});

        int neighbors[4][2] = {{current.x, current.y - 1}, {current.x + 1, current.y}, {current.x, current.y + 1}, {current.x - 1, current.y}};

        for (auto& neighbor : neighbors) {
            int nx = neighbor[0];
            int ny = neighbor[1];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                !wall_set.count({nx, ny}) && !rock_set.count({nx, ny})) {
                
                std::vector<std::pair<int, int>> new_path = current.path;
                new_path.push_back({nx, ny});
                pq.push({nx, ny, (int)new_path.size(), new_path});
            }
        }
    }

    return {}; // Return an empty path if no path is found
}

PYBIND11_MODULE(pathery_pathfinding, m) {
    m.doc() = "pybind11 plugin for Pathery pathfinding";
    m.def("find_path_cpp", &find_path_cpp, "A function that finds a path using C++");
}
