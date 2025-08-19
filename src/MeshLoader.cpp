#include "MeshLoader.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <vector>
#include <stdexcept>
#include <cassert>

bool MeshLoader::load(const std::string& filename, int numThreads) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Failed to open node file: " << filename << "\n";
        return false;
    }

    std::vector<std::string> lines;
    std::string line;

    while (std::getline(infile, line)) {
        if (!line.empty()) lines.push_back(line);
    }

    infile.close();

    if (lines.empty()) return false;

    dimension = countColumns(lines[0]);
    if (dimension != 2 && dimension != 3) {
        std::cerr << "Unsupported dimension: " << dimension << "\n";
        return false;
    }

    omp_set_num_threads(numThreads);

    if (dimension == 2) {
        std::vector<std::vector<Node2D>> threadNodes(numThreads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int i = 0; i < lines.size(); ++i) {
                size_t pos1 = lines[i].find(',');
                try {
                    Node2D node;
                    node.x = std::stod(lines[i].substr(0, pos1));
                    node.y = std::stod(lines[i].substr(pos1 + 1));
                    threadNodes[tid].push_back(node);
                } catch (...) {
                    #pragma omp critical
                    std::cerr << "Skipping malformed node line: " << lines[i] << "\n";
                }
            }
        }

        for (const auto& chunk : threadNodes) {
            nodes2D.insert(nodes2D.end(), chunk.begin(), chunk.end());
        }

    } else {
        std::vector<std::vector<Node3D>> threadNodes(numThreads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int i = 0; i < lines.size(); ++i) {
                size_t pos1 = lines[i].find(',');
                size_t pos2 = lines[i].find(',', pos1 + 1);
                try {
                    Node3D node;
                    node.x = std::stod(lines[i].substr(0, pos1));
                    node.y = std::stod(lines[i].substr(pos1 + 1, pos2 - pos1 - 1));
                    node.z = std::stod(lines[i].substr(pos2 + 1));
                    threadNodes[tid].push_back(node);
                } catch (...) {
                    #pragma omp critical
                    std::cerr << "Skipping malformed node line: " << lines[i] << "\n";
                }
            }
        }

        for (const auto& chunk : threadNodes) {
            nodes3D.insert(nodes3D.end(), chunk.begin(), chunk.end());
        }
    }

    return true;
}

bool MeshLoader::loadElements(const std::string& elementFilename, int numThreads) {
    std::ifstream infile(elementFilename);
    if (!infile) {
        std::cerr << "Failed to open element file: " << elementFilename << "\n";
        return false;
    }

    std::vector<std::string> lines;
    std::string line;

    while (std::getline(infile, line)) {
        if (!line.empty()) lines.push_back(line);
    }

    infile.close();

    if (lines.empty()) return false;

    int elementColumns = countColumns(lines[0]);
    omp_set_num_threads(numThreads);

    if (dimension == 2) {
        if (elementColumns != 3 && elementColumns != 4) {
            std::cerr << "Unsupported 2D element with " << elementColumns << " nodes.\n";
            return false;
        }

        std::vector<std::vector<Element2D>> threadElems(numThreads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int i = 0; i < lines.size(); ++i) {
                Element2D elem;
                size_t start = 0, end;
                for (int j = 0; j < elementColumns; ++j) {
                    end = lines[i].find(',', start);
                    std::string token = lines[i].substr(start, end - start);
                    try {
                        elem.node_ids.push_back(std::stoi(token));
                    } catch (...) {
                        #pragma omp critical
                        std::cerr << "Malformed 2D element line: " << lines[i] << "\n";
                        break;
                    }
                    start = (end == std::string::npos) ? std::string::npos : end + 1;
                }
                if (elem.node_ids.size() == elementColumns)
                    threadElems[tid].push_back(elem);
            }
        }

        for (const auto& chunk : threadElems)
            elements2D.insert(elements2D.end(), chunk.begin(), chunk.end());

    } else if (dimension == 3) {
        if (elementColumns != 4 && elementColumns != 8) {
            std::cerr << "Unsupported 3D element with " << elementColumns << " nodes.\n";
            return false;
        }

        std::vector<std::vector<Element3D>> threadElems(numThreads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int i = 0; i < lines.size(); ++i) {
                Element3D elem;
                size_t start = 0, end;
                for (int j = 0; j < elementColumns; ++j) {
                    end = lines[i].find(',', start);
                    std::string token = lines[i].substr(start, end - start);
                    try {
                        elem.node_ids.push_back(std::stoi(token));
                    } catch (...) {
                        #pragma omp critical
                        std::cerr << "Malformed 3D element line: " << lines[i] << "\n";
                        break;
                    }
                    start = (end == std::string::npos) ? std::string::npos : end + 1;
                }
                if (elem.node_ids.size() == elementColumns)
                    threadElems[tid].push_back(elem);
            }
        }

        for (const auto& chunk : threadElems)
            elements3D.insert(elements3D.end(), chunk.begin(), chunk.end());
    }

    return true;
}

bool MeshLoader::loadISPV(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Failed to open ISPV file: " << filename << "\n";
        return false;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    infile.close();

    int numThreads = omp_get_max_threads();
    std::vector<std::vector<IndexSetEntry>> threadEntries(numThreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
            size_t pos = lines[i].find(',');
            if (pos == std::string::npos) {
                #pragma omp critical
                std::cerr << "Malformed ISPV line: " << lines[i] << "\n";
                continue;
            }

            try {
                IndexSetEntry entry;
                entry.nodeId = std::stoi(lines[i].substr(0, pos));
                entry.setId = std::stoi(lines[i].substr(pos + 1));
                threadEntries[tid].push_back(entry);
            } catch (...) {
                #pragma omp critical
                std::cerr << "Failed to parse ISPV line: " << lines[i] << "\n";
            }
        }
    }

    for (const auto& chunk : threadEntries)
        ispv.insert(ispv.end(), chunk.begin(), chunk.end());

    return true;
}

bool MeshLoader::loadISSV(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Failed to open ISSV file: " << filename << "\n";
        return false;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    infile.close();

    int numThreads = omp_get_max_threads();
    std::vector<std::vector<IndexSetEntry>> threadEntries(numThreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
            size_t pos = lines[i].find(',');
            if (pos == std::string::npos) {
                #pragma omp critical
                std::cerr << "Malformed ISSV line: " << lines[i] << "\n";
                continue;
            }

            try {
                IndexSetEntry entry;
                entry.nodeId = std::stoi(lines[i].substr(0, pos));
                entry.setId = std::stoi(lines[i].substr(pos + 1));
                threadEntries[tid].push_back(entry);
            } catch (...) {
                #pragma omp critical
                std::cerr << "Failed to parse ISSV line: " << lines[i] << "\n";
            }
        }
    }

    for (const auto& chunk : threadEntries)
        issv.insert(issv.end(), chunk.begin(), chunk.end());

    return true;
}


void MeshLoader::saveDisplacements(const std::string& filename, const Eigen::VectorXd& displacements, int degreeOfFreedom) const 
{
    // Determine number of nodes from dimension
    int numNodes = (dimension == 2 ? 
                     static_cast<int>(nodes2D.size()) : 
                     static_cast<int>(nodes3D.size()));
    // Sanity check length
    if (displacements.size() != numNodes * degreeOfFreedom) {
        throw std::runtime_error(
          "saveDisplacements: displacements.size()="
          + std::to_string(displacements.size())
          + " but expected " 
          + std::to_string(numNodes * degreeOfFreedom)
        );
    }

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error(
          "saveDisplacements: could not open " + filename);
    }

    // Write each node’s DOFs on its own line
    for (int i = 0; i < numNodes; ++i) {
        for (int d = 0; d < degreeOfFreedom; ++d) {
            int idx = i * degreeOfFreedom + d;
            out << displacements[idx];
            if (d + 1 < degreeOfFreedom) out << ',';
        }
        out << '\n';
    }
    out.close();
}


int MeshLoader::countColumns(const std::string& line) const {
    return static_cast<int>(std::count(line.begin(), line.end(), ',')) + 1;
}

int MeshLoader::getDimension() const {
    return dimension;
}

const std::vector<Node2D>& MeshLoader::getNodes2D() const {
    return nodes2D;
}

const std::vector<Node3D>& MeshLoader::getNodes3D() const {
    return nodes3D;
}

const std::vector<Element2D>& MeshLoader::getElements2D() const {
    return elements2D;
}

const std::vector<Element3D>& MeshLoader::getElements3D() const {
    return elements3D;
}

const std::vector<IndexSetEntry>& MeshLoader::getISPV() const {
    return ispv;
}

const std::vector<IndexSetEntry>& MeshLoader::getISSV() const {
    return issv;
}
