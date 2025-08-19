// MeshLoader.h
#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <unordered_map>

struct Node2D 
{
    double x, y;
};

struct Node3D 
{
    double x, y, z;
};

struct Element2D 
{
    std::vector<int> node_ids; // can be 3 (tri) or 4 (quad)
};

struct Element3D 
{
    std::vector<int> node_ids; // can be 4 (tetra) or 8 (hexa)
};

struct IndexSetEntry 
{
    int nodeId;
    int setId;
};

class MeshLoader 
{
public:
    bool load(const std::string& filename, int numThreads = 1);
    bool loadElements(const std::string& elementFilename, int numThreads = 1);
    bool loadISPV(const std::string& filename);
    bool loadISSV(const std::string& filename);


    int getDimension() const;
    const std::vector<Node2D>& getNodes2D() const;
    const std::vector<Node3D>& getNodes3D() const;
    const std::vector<Element2D>& getElements2D() const;
    const std::vector<Element3D>& getElements3D() const;
    const std::vector<IndexSetEntry>& getISPV() const;
    const std::vector<IndexSetEntry>& getISSV() const;

    // save computed displacements
    void saveDisplacements(const std::string& filename, const Eigen::VectorXd& displacements, int degreeOfFreedom) const;

private:
    int countColumns(const std::string& line) const;
    int dimension = 0;
    std::vector<Node2D> nodes2D;
    std::vector<Node3D> nodes3D;
    std::vector<Element2D> elements2D;
    std::vector<Element3D> elements3D;
    std::vector<IndexSetEntry> ispv;
    std::vector<IndexSetEntry> issv;
};
