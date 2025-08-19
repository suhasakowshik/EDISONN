#include <iostream>
#include <iomanip>
#include <omp.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <unordered_map>
#include <string>

// custom headers
#include "EDISONN.h"
#include "MeshLoader.h"
#include "SOLVER.h"
#include "LSQSmoother.h"

// reading input file
std::unordered_map<std::string, std::string> parseInputFile(const std::string& filename)
{
    std::unordered_map<std::string, std::string> params;
    std::ifstream infile(filename);
    std::string line;

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::string key, eq, value;
        if (!(iss >> key >> eq >> value)) continue; // skip invalid lines
        if (eq == "=")
            params[key] = value;
    }
    return params;
}

int main(int argc,char* argv[]) 
{   
    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try
    {   
        if(argc != 2 || std::string(argv[1]) == "--help")
        {
            std::cout << "Usage: ./edisonn <inputFile>\n";
            return 0;
        }

        auto params = parseInputFile(argv[1]);

        int v = params.count("version") ? std::stoi(params["version"]) : 1;
        std::string meshType = params.count("meshType") ? params["meshType"] : "quad";
        std::string simName = params.count("simName") ? params["simName"] : "cooks_membrane";
        std::string problemType = params.count("problemType") ? params["problemType"] : "plane_elasticity";
        std::string planeType = params.count("planeType") ? params["planeType"] : "plane_strain";
        std::string loadType = params.count("loadType") ? params["loadType"] : "distributed_load";
        std::string smooth = params.count("smooth") ? params["smooth"] : "true";
        std::string smoothFitType = params.count("smoothFitType") ? params["smoothFitType"] : "linear";
        double load = params.count("load") ? std::stod(params["load"]) : 1;
        double E = params.count("E") ? std::stod(params["E"]) : 100;
        double nu = params.count("nu") ? std::stod(params["nu"]) : 0.3;
        double thickness = params.count("thickness") ? std::stod(params["thickness"]) : 1.0;
        int smoothIter = params.count("smoothIter") ? std::stoi(params["smoothIter"]) : 2;
        int numThreads = params.count("numThreads") ? std::stoi(params["numThreads"]) : 1;

        bool doSmooth = (smooth == "true" || smooth == "1");
                
        // set threads for openMP
        omp_set_num_threads(numThreads);

        // Construct simType using simName and meshType
        std::string simType = simName + "_" + meshType + "-";

        // File names for various input txt files
        std::string nodeFile = simType + std::to_string(v) + "_nodes.txt";
        std::string elementFile = simType + std::to_string(v) + "_elem.txt";
        std::string ispvFile = simType + std::to_string(v) + "_ispv.txt";
        std::string issvFile = simType + std::to_string(v) + "_issv.txt";

        // displacement data save file name
        std::string saveDispFile = simType + std::to_string(v) + "_displacement_output.txt";

        //log file
        std::string loggingFile = simType + std::to_string(v) + "_simulation_log.txt";

        // Echo inputs for clarity
        std::cout << "[INFO] Parsed input parameters:\n";
        std::cout << "  Version: " << v << "\n";
        std::cout << "  Mesh Type: " << meshType << "\n";
        std::cout << "  Simulation Name: " << simName << "\n";
        std::cout << "  Problem Type: " << problemType << "\n";
        std::cout << "  Plane Type: " << planeType << "\n";
        std::cout << "  Load Type: " << loadType << "\n";
        std::cout << "  Smooth : " << smooth << "\n";
        std::cout << "  Load: " << load << "\n";
        std::cout << "  E: " << E << "\n";
        std::cout << "  nu: " << nu << "\n";
        std::cout << "  Thickness: " << thickness << "\n";
        std::cout << "  Smooth Itertaions: "<< smoothIter << "\n";
        std::cout << "  Num Threads: " << numThreads << "\n";

        // load the MESH object
        MeshLoader loader;

        std::cout << "Loading nodes...\n";
        if (!loader.load(nodeFile, numThreads)) 
        {
            std::cerr << "Failed to load nodes.\n";
            return 1;
        }

        std::cout << "Loading elements...\n";
        if (!loader.loadElements(elementFile, numThreads)) 
        {
            std::cerr << "Failed to load elements.\n";
            return 1;
        }

        std::cout << "Loading Displacement boundary conditions...\n";
        if (!loader.loadISPV(ispvFile)) {
            std::cerr << "Failed to load ISPV data.\n";
        }

        std::cout << "Loading Force boundary conditions...\n";
        if (!loader.loadISSV(issvFile)) {
            std::cerr << "Failed to load ISSV data.\n";
        } 

        // get 2D or 3D
        std::cout << "Detected dimension: " << loader.getDimension() << "D\n";

        // declare number of nodes in the mesh
        int numNodes;

        // declare EDISONN object
        EDISONN ed;
        
        // declare neighboring nodes connectivity and coordinates
        std::vector<std::vector<int>> connectivity;
        std::vector<Coordinate> coordinates;

        //dimension of the problem
        int dim=loader.getDimension();


        if(loader.getDimension()==2)
        {   
            numNodes=loader.getNodes2D().size();
            for (const auto& elem:loader.getElements2D()) {
                connectivity.push_back(elem.node_ids);  
            }

            for(const auto& coords:loader.getNodes2D()){
                coordinates.push_back(Coordinate{coords.x,coords.y,0.0});
            }
        }
        else {

            numNodes=loader.getNodes3D().size();
            for(const auto& elem:loader.getElements3D())
            {
                connectivity.push_back(elem.node_ids);   
            }
            for(const auto& coords:loader.getNodes3D()){
                coordinates.push_back(Coordinate{coords.x,coords.y,coords.z});
            }
        }

        //declare DOF and totalDOFS
        int degreeOfFreedom=2;
        int numDofs=degreeOfFreedom*numNodes;

        //declare SOLVER object
        SOLVER sl(numDofs);

        // primary variable or Dritchlet BC or ISPV
        auto ISPV=loader.getISPV();
        int NSPV=ISPV.size();
        Eigen::VectorXd VSPV=Eigen::VectorXd::Zero(NSPV);

        // secondaoty variable or Neumann or ISSV
        auto ISSV=loader.getISSV();
        int NSSV=ISSV.size();
        auto VSSV=sl.computeNeumannValues(load,ISSV,thickness,coordinates,loadType);

        //find neigbouring nodes
        auto nbn=ed.findConnectedNodes(connectivity,numNodes);

        //find neighbouring elements and shared volume and/or area
        auto [nbe, sharedAreas] = ed.getSharedAreasAndConnectedElements(coordinates, connectivity);

        // material propety matrix
        auto matProp=ed.materialProperty(E,nu,thickness,problemType,planeType);

        // calculate gradient operator
        auto Gradient=ed.GradientOperator(coordinates,nbn,degreeOfFreedom,dim);

        // calculate nodal stiffness matrix array - local stiffness and local force vector
        auto [elf,elk]=ed.NodalStiffness(matProp,nbn,Gradient,sharedAreas,degreeOfFreedom);

        //assemble local stiffness to global stiffness
        sl.assembleFromLocal(nbn, elk, degreeOfFreedom);

        // assemble local force to global
        sl.assembleForceFromLocal(nbn,elf,degreeOfFreedom);

        //apply primary or drictlet or ISPV boundary condition
        if(NSPV>0)
        {
            sl.applyDirichletBCs(ISPV,VSPV,degreeOfFreedom);
        }
        //apply secondary or neumann or ISSV boundary condition
        if(NSSV>0)
        {   
            sl.applyNeumannBCs(ISSV,VSSV,degreeOfFreedom);
        }

        // solving KU=F (cholesky) to global displacement values glu
        //auto glu=sl.matrixSolver();
        auto glu=sl.matrixSolver_cg();
        auto U_sm=glu;

        //check if user wants smoothing from input file
        if(doSmooth)
        {
            //declare least squares moothing object
            int coordDim = loader.getDimension();     // 2 or 3
            int N = coordinates.size();
            Eigen::MatrixXd nodes_lsq(N, coordDim);
            for(int i=0;i<N;++i)
            {
                nodes_lsq(i,0)=coordinates[i].x;
                nodes_lsq(i,1)=coordinates[i].y;
                if (coordDim==3) 
                {
                    nodes_lsq(i, 2) = coordinates[i].z;
                }
            }

            LSQSmoother lsqSmooth(nodes_lsq,nbn,smoothFitType);
            auto U_sm=lsqSmooth.smooth(glu,smoothIter,degreeOfFreedom);
        }

        //saving displacements to txt file
        loader.saveDisplacements(saveDispFile, U_sm, degreeOfFreedom);

        // End timer
        auto endTime = std::chrono::high_resolution_clock::now();

        std::cout << "[INFO] Simulation completed.\n";

        // Calculate duration in seconds
        std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "[INFO] Total computation time: " << elapsed.count() << " seconds.\n";

        // After simulation ends
        std::ofstream logFile(loggingFile, std::ios::app); // append mode

        if(logFile.is_open())
        {
            logFile << "==== Simulation Log Entry ====\n";
            logFile << "Version: " << v << "\n";
            logFile << "Simulation Name: " << simName << "\n";
            logFile << "Mesh Type: " << meshType << "\n";
            logFile << "Problem Type: " << problemType << "\n";
            logFile << "Plane Type: " << planeType << "\n";
            logFile << "Load Type: " << loadType << "\n";
            logFile << "Smooth : " << smooth << "\n";
            logFile << "Load: " << load << "\n";
            logFile << "E (Young's Modulus): " << E << "\n";
            logFile << "nu (Poisson's Ratio): " << nu << "\n";
            logFile << "Thickness: " << thickness << "\n";
            logFile << "Smooth Itertaions: "<< smoothIter << "\n";
            logFile << "Num Threads: " << numThreads << "\n";
            logFile << "Total computation time: " << elapsed.count() << " seconds.\n";
            logFile << "==============================\n\n";

            logFile.close();
        }
        else
        {
            std::cerr << "[WARNING] Could not open log file for writing.\n";
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {   
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;

        std::cerr << "[ERROR] " << e.what() << "\n";
        std::cerr << "[INFO] Total computation time before error: " << elapsed.count() << " seconds.\n";

        return EXIT_FAILURE;
    }
}
