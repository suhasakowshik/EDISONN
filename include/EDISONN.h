// EDISONN.h
#pragma once

#include <utility>
#include <vector>
#include <iostream>
#include <string>
#include <set>
#include <algorithm>
#include <Eigen/Dense>



//custom headers
#include "MeshLoader.h"


struct Coordinate 
{
    double x, y, z = 0.0; // Compatible with 2D and 3D
};

class EDISONN 
{
public:

    //function find connected nodes using element connecteivity per node
    std::vector<std::vector<int>> findConnectedNodes(const std::vector<std::vector<int>>& elementConnectivity,int numNodes);

    //function to find shread area per node and connected elements per node
    std::pair<std::vector<std::vector<int>>, std::vector<double>> getSharedAreasAndConnectedElements(
        const std::vector<Coordinate>& coordinates,
        const std::vector<std::vector<int>>& nodalConnectivity
    );

    //Gradient operator
    std::vector<Eigen::MatrixXd> GradientOperator(const std::vector<Coordinate>& nodes, 
        const std::vector<std::vector<int>>& connectedNodes, int degreeOfFreedom,int getDomension);

    // Material Property Matrix
    Eigen::MatrixXd materialProperty(double E,double nu,double thkns,const std::string& problemType,const std::string& planeType);

    // Nodal Stiffness Matrix
    std::pair<std::vector<Eigen::MatrixXd>,std::vector<Eigen::MatrixXd>> NodalStiffness(Eigen::MatrixXd& matProp,
        const std::vector<std::vector<int>>& adjacency,std::vector<Eigen::MatrixXd>& gradient,
        std::vector<double>& sharedVolume,int degreeOfFreedom);
};
