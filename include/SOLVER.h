// SOLVER.h
#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <unordered_set>

// custom headers
#include "MeshLoader.h"
#include "EDISONN.h"



class SOLVER
{
public:
    using Triplet=Eigen::Triplet<double>;

    SOLVER(int numDofs);

    void reserve(int nz);

    // sparse triplets assembly
    void assembleFromLocal(const std::vector<std::vector<int>>& adjacency, const std::vector<Eigen::MatrixXd>&  localStiffness,
        int degreeOfFreedom);

    // Direct CSR assembly (no triplets): (Compressed Sparse Row)
    void assembleFromLocalCSR( const std::vector<std::vector<int>>& adjacency, const std::vector<Eigen::MatrixXd>&  localStiffness,
        int degreeOfFreedom);

    // global force vector assembly
    void assembleForceFromLocal(const std::vector<std::vector<int>>& adjacency, const std::vector<Eigen::MatrixXd>&  localForce,
        int degreeOfFreedom);

    // primary variable (ISPV) BC
    void applyDirichletBCs(const std::vector<IndexSetEntry>& ISPV, const Eigen::VectorXd& VSPV, int degreeOfFreedom);

    // secondary variable (ISSV) BC
    void applyNeumannBCs(const std::vector<IndexSetEntry>& ISSV,const Eigen::VectorXd& VSSV,int degreeOfFreedom);

    //old 
    Eigen::VectorXd computeNeumannValues(double q, std::vector<IndexSetEntry>& issv, double thkns,
        const std::vector<Coordinate>& pts,const std::string& loadType);
    
    // Solve Ax=B , returs displacement (cholesky)
    Eigen::VectorXd matrixSolver();
    
    //congugate grad iterative solver
    Eigen::VectorXd matrixSolver_cg();
    
    /// Accessors
    const Eigen::SparseMatrix<double>& matrixK() const { return K; }
    const Eigen::VectorXd& vectorF() const { return F; }


private:
    int   numDofs;                 // number of DOFs
    std::vector<Triplet> triplets; // stiffness contributions
    Eigen::SparseMatrix<double> K;  // global stiffness
    Eigen::VectorXd F;          // global force
};