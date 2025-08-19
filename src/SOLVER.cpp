#include "SOLVER.h"
#include <stdexcept>
#include <iomanip>
#include <cassert>
#include <omp.h>

SOLVER::SOLVER(int totalDofs): numDofs(totalDofs),K(totalDofs,totalDofs),F(Eigen::VectorXd::Zero(totalDofs)) {}

void SOLVER::reserve(int nz)
{
    triplets.reserve(nz);
}

void SOLVER::assembleFromLocal(const std::vector<std::vector<int>>& adjacency,
    const std::vector<Eigen::MatrixXd>& localStiffness, int degreeOfFreedom)
{
    int numNodes = static_cast<int>(adjacency.size());

    triplets.clear();

    // Parallel assembly
    std::vector<std::vector<Eigen::Triplet<double>>> tripletsPerThread;

    #pragma omp parallel
    {
        int threadID = omp_get_thread_num();

        // Resize tripletsPerThread only once
        #pragma omp single
        {
            tripletsPerThread.resize(omp_get_num_threads());
        }

        auto& localTriplets = tripletsPerThread[threadID];

        #pragma omp for
        for (int n = 0; n < numNodes; ++n)
        {
            const auto& neigh = adjacency[n];
            int h = static_cast<int>(neigh.size());
            int ndofLocal = h * degreeOfFreedom;

            const auto& Kloc = localStiffness[n];

            if (Kloc.rows() != ndofLocal || Kloc.cols() != ndofLocal)
            {   
                std::cerr << "assembleFromLocal ERROR: node " << n
                            << ", neighbors h = " << h
                            << ", degreeOfFreedom = " << degreeOfFreedom
                            << ", expected ndofLocal = " << ndofLocal
                            << ", Kloc.rows = " << Kloc.rows()
                            << ", Kloc.cols = " << Kloc.cols()
                            << std::endl;
                throw std::invalid_argument("local Stiffness[" + std::to_string(n) + "] has wrong dimensions");
            }

            std::vector<int> gdofs(ndofLocal);

            for (int j = 0; j < h; ++j)
            {
                int nodeID = neigh[j];
                if (nodeID < 0 || nodeID >= numNodes)
                {
                    throw std::out_of_range("adjacency[" + std::to_string(n) + "][" + std::to_string(j) + "] out of range");
                }

                for (int k = 0; k < degreeOfFreedom; ++k)
                {
                    gdofs[j * degreeOfFreedom + k] = nodeID * degreeOfFreedom + k;
                }
            }

            for (int a = 0; a < ndofLocal; ++a)
            {
                for (int b = 0; b < ndofLocal; ++b)
                {
                    localTriplets.emplace_back(gdofs[a], gdofs[b], Kloc(a, b));
                }
            }
        }
    }

    // Merge
    for (const auto& local : tripletsPerThread)
    {
        triplets.insert(triplets.end(), local.begin(), local.end());
    }

    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
}


void SOLVER::assembleFromLocalCSR(const std::vector<std::vector<int>>& adjacency,
    const std::vector<Eigen::MatrixXd>&  localStiffness,int degreeOfFreedom)
{
    int numNodes = static_cast<int>(adjacency.size());

    // count nonzero values per global row
    std::vector<int> rowNNZ(numDofs,0);
    for(int n=0;n<numNodes;++n)
    {
        int h=static_cast<int>(adjacency[n].size());
        int ndofLocal=h*degreeOfFreedom;

        // for exahc local row, indert ndofLocal in glowbal row
        for(int j=0;j<h;++j)
        {
            int globalRow=adjacency[n][j]*degreeOfFreedom;

            // each of that node's dof rows repeats for each local col
            for(int k=0;k<degreeOfFreedom;++k)
            {
                rowNNZ[globalRow+k]+=ndofLocal;
            }

        }
    }

    // build onuter index
    std::vector<int> outer(numDofs+1);
    outer[0]=0;
    for(int i=0;i<numDofs;i++)
    {
        outer[i+1]=outer[i]+rowNNZ[i];
    }

    // allocate CSR buffers
    K.resize(numDofs,numDofs);
    K.makeCompressed();

    auto ptrOuter=K.outerIndexPtr();
    auto ptrInner=K.innerIndexPtr();
    auto ptrValue=K.valuePtr();

    //copy outer index
    std::copy(outer.begin(),outer.end(),ptrOuter);

    // scatter in to CSR
    std::vector<int> writePos=outer; // current write pos per row

    for(int n=0;n<numNodes;++n)
    {
        const auto& neigh = adjacency[n];
        int h = static_cast<int>(neigh.size());
        int ndofLocal = h * degreeOfFreedom;
        const auto& Kloc = localStiffness[n];

        // build gdofs
        std::vector<int> gdofs(ndofLocal);
        for (int j = 0; j < h; ++j) 
        {
            int nodeID = neigh[j];
            for (int k = 0; k < degreeOfFreedom; ++k) 
            {
                gdofs[j*degreeOfFreedom + k] = nodeID*degreeOfFreedom + k;
            }
        }

        // scatter values
        for (int a = 0; a < ndofLocal; ++a) 
        {
            int I = gdofs[a];
            for (int b = 0; b < ndofLocal; ++b) 
            {
                int J = gdofs[b];
                int idx = writePos[I]++;
                ptrInner[idx] = J;
                ptrValue[idx] = Kloc(a,b);
            }
        }
    }
    
    K.makeCompressed();

}

void SOLVER::assembleForceFromLocal(const std::vector<std::vector<int>>& adjacency, const std::vector<Eigen::MatrixXd>&  localForce,
    int degreeOfFreedom)
{
    int numNodes=static_cast<int>(adjacency.size());

    F.setZero();

    #pragma omp parallel for
    for(int n=0;n<numNodes;++n)
    {
        auto const& neigh=adjacency[n];
        int h=(int)neigh.size();

        int nd=h*degreeOfFreedom;
        auto const& Floc=localForce[n];

        // sanity check
        if (Floc.rows()!=nd || Floc.cols()!=1)
        {
            throw std::invalid_argument("wrong Floc dims");
        }

        for(int j=0;j<h;++j)
        {
            int base=neigh[j]*degreeOfFreedom;
            for(int k=0;k<degreeOfFreedom;++k)
            {
                int g=base+k;

                #pragma omp atomic
                F[g]+=Floc(j*degreeOfFreedom+k,0);
            }
        }
    }
}


void SOLVER::applyDirichletBCs(const std::vector<IndexSetEntry>& ISPV, const Eigen::VectorXd& VSPV, int degreeOfFreedom)
{
    int NSSV=static_cast<int>(ISPV.size());

    if(NSSV<=0)
    {
        return;
    }

    // build global range and values
    std::vector<char> isBC(numDofs, 0);
    std::vector<double> bcVal(numDofs, 0.0);
    for (int i = 0; i < NSSV; ++i) 
    {
        int node = ISPV[i].nodeId;
        int dof   = ISPV[i].setId;
        int gdof  = node*degreeOfFreedom + dof;

        if (gdof < 0 || gdof >= numDofs)
        {
            throw std::out_of_range("BC index out of range");
        }
        
        isBC[gdof]    = 1;
        bcVal[gdof]   = (i < (int)VSPV.size() ? VSPV(i) : 0.0);
    }

    // zero-out rows/cols in a single pass over nonzeros
    for(int col=0;col<numDofs;++col)
    {
        for(Eigen::SparseMatrix<double>::InnerIterator it(K,col);it;++it)
        {
            int row=it.row();
            if(isBC[row] || isBC[col])
            {
                it.valueRef()=(row==col && isBC[col]) ? 1.0 : 0.0;
            }
        }
    }

}

void SOLVER::applyNeumannBCs(const std::vector<IndexSetEntry>& ISSV,const Eigen::VectorXd& VSSV,int degreeOfFreedom)
{
    int nssv = static_cast<int>(ISSV.size());
    if (nssv <= 0) return;

    for (int i = 0; i < nssv; ++i) 
    {
        int node = ISSV[i].nodeId;
        int dof   = ISSV[i].setId;
        double val = (i < (int)VSSV.size() ? VSSV(i) : 0.0);
        int g = node*degreeOfFreedom + dof;

        //sanity check
        if (g<0||g>=numDofs) 
        {
            throw std::out_of_range("Neumann BC out of range");
        }

        // simply add the traction/force to the global RHS
        F[g] += val;
    }
}

Eigen::VectorXd SOLVER::computeNeumannValues(double q, std::vector<IndexSetEntry>& issv, double thkns,
        const std::vector<Coordinate>& pts,const std::string& loadType)
{
    int nssv = static_cast<int>(issv.size());
    Eigen::VectorXd vssv = Eigen::VectorXd::Zero(nssv);

    if(loadType=="distributed_load")
    {
        // Sort issv by the y-coordinate of the node
        std::sort(issv.begin(), issv.end(),[&](auto const &a, auto const &b)
                {return pts[a.nodeId].y < pts[b.nodeId].y;});

        // Build an array of sorted coordinates
        std::vector<Coordinate> sortedPts(nssv);
        for (int i = 0; i < nssv; ++i) 
        {
            sortedPts[i] = pts[issv[i].nodeId];
        }

        // Accumulate Neumann values along adjacent pairs
        for (int k = 0; k < nssv - 1; ++k) 
        {
            double dx = sortedPts[k].x - sortedPts[k+1].x;
            double dy = sortedPts[k].y - sortedPts[k+1].y;
            double length = std::sqrt(dx*dx + dy*dy);
            double qval   = q * length * thkns * 0.5;
            vssv[k]   += qval;
            vssv[k+1] += qval;
        }
    }
    else if(loadType=="point_load")
    {
        for (int i = 0; i < nssv; ++i)
        {
            vssv[i] += q;
        }
    }
    else
    {
        throw std::invalid_argument("Unknown loadType: " + loadType);
    }

    return vssv;
}


Eigen::VectorXd SOLVER::matrixSolver()
{
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> matsol;
    matsol.compute(K);
    if(matsol.info() != Eigen::Success)
    {
        throw std::runtime_error("Cholesky decomposition failed");
    }
    return matsol.solve(F);
}

Eigen::VectorXd SOLVER::matrixSolver_cg()
{
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> matsol;
    matsol.compute(K);
    if (matsol.info() != Eigen::Success)
    {
        throw std::runtime_error("Conjugate Gradient decomposition failed");
    }

    matsol.setTolerance(1e-6); // Adjust as needed
    matsol.setMaxIterations(2000);

    Eigen::VectorXd x = matsol.solve(F);
    if (matsol.info() != Eigen::Success)
    {
        throw std::runtime_error("Conjugate Gradient solve failed");
    }

    std::cout << "Conjugate Gradient converged in " << matsol.iterations()
              << " iterations with estimated error " << matsol.error() << std::endl;

    return x;
}








