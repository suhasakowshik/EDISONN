#define EIGEN_USE_MKL_ALL
#include <cassert>
#include <utility>
#include <vector>
#include <cmath>
#include <cassert>
#include <Eigen/Dense>
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct>
#include <string>

// custom headers
#include "EDISONN.h"


// function to calculate area of the polygon 
static double polygonArea2D(const std::vector<Coordinate>& coords)
{
    double area=0.0;
    int n=coords.size();
    for(int i=0;i<n;i++)
    {
        const auto& p1=coords[i];
        const auto& p2=coords[(i+1)%n];
        area+=(p1.x*p2.y-p2.x*p1.y);
    }
    return std::abs(area)*0.5;
}

// psuedo inverse
static Eigen::MatrixXd pinv(const Eigen::MatrixXd &M)
{
     return M.completeOrthogonalDecomposition().pseudoInverse();
}

std::vector<std::vector<int>> EDISONN::findConnectedNodes(const std::vector<std::vector<int>>& elementConnectivity,int numNodes)
{

    // 1) buid inverse map: which elements tpouch the node
    std::vector<std::vector<int>> elemsOfNode(numNodes);

    for (int e=0;e<(int)elementConnectivity.size();++e) 
    {
        for(int node:elementConnectivity[e])
        {   
            assert(node >= 0 && node < numNodes);
            elemsOfNode[node].push_back(e);
        }
    }

    //2) output and stamping
    std::vector<std::vector<int>> connected(numNodes);

    #pragma omp parallel
    {
        std::vector<int> lastSeen(numNodes,-1);
        int stamp=0;


        #pragma omp for
        for(int i=0;i<numNodes;++i)
        {
            ++stamp;
            auto &nbrs=connected[i];
            nbrs.reserve(elemsOfNode[i].size()*2 + 1);

            //including self node at the start
            nbrs.push_back(i);
            lastSeen[i]=stamp;

            // see only the elements that touch i
            for (int e : elemsOfNode[i]) 
            {
                auto const &elem=elementConnectivity[e];
                int p=(int)elem.size();

                //find local j such that its in elem
                int jLoc=0;
                while(jLoc <p && elem[jLoc] !=i) ++jLoc;
                assert(jLoc<p);

                if(p==3)
                {
                    //traingle
                    for(int v=1;v<=2;++v)
                    {
                        int nbr=elem[(jLoc+v)%3];
                        if(lastSeen[nbr]!=stamp)
                        {
                            nbrs.push_back(nbr);
                            lastSeen[nbr]=stamp;
                        }
                    }
                }

                else if (p==4) 
                {
                    //quad 
                    for (int v : {1,3})
                    {
                        int nbr=elem[(jLoc+v)%4];
                        if (lastSeen[nbr]!=stamp)
                        {
                            nbrs.push_back(nbr);
                            lastSeen[nbr]=stamp;
                        }
                    }
                }
            }
        }
    }

    return connected;

}


std::pair<std::vector<std::vector<int>>, std::vector<double>> EDISONN::getSharedAreasAndConnectedElements(
    const std::vector<Coordinate>& coordinates, const std::vector<std::vector<int>>& nodalConnectivity
)
{
    int numNodes = (int)coordinates.size();
    int numElements = (int)nodalConnectivity.size();

    std::vector<std::vector<int>> neighboringElements(numNodes);
    std::vector<double> sharedAreas(numNodes,0.0);

    // compure area per node of each element
    std::vector<double> areaPerElement(numElements);

    #pragma omp parallel for
    for(int e=0;e<numElements;++e)
    {
        const auto &elem=nodalConnectivity[e];
        int npe=(int)elem.size();
        assert(npe>=3); // polygon must have atleast 3 nodes

        // get vertex coords
        std::vector<Coordinate> vert(npe);

        for(int k=0;k<npe;++k)
        {
            int node=elem[k];
            assert(node>=0 && node<numNodes);
            vert[k]=coordinates[node];
        }
        double area=polygonArea2D(vert);
        areaPerElement[e]=area/npe;
    }

    // build the shared element list and total shared area per node
    for(int e=0;e<numElements;++e)
    {
        double share=areaPerElement[e];
        for(int node:nodalConnectivity[e])
        {
            neighboringElements[node].push_back(e);
            sharedAreas[node]+=share;
        }
    }

    return {neighboringElements,sharedAreas};

}

std::vector<Eigen::MatrixXd> EDISONN::GradientOperator(const std::vector<Coordinate>& nodes, 
    const std::vector<std::vector<int>>& connectedNodes,int degreeOfFreedom,int dim)
{
    const int Npts=static_cast<int>(nodes.size());
    assert(dim == 2 || dim == 3);
    assert(static_cast<int>(connectedNodes.size()) == Npts);
    assert(degreeOfFreedom > 0);

    //setup idenetu matrix based on size of dof
    Eigen::MatrixXd I=Eigen::MatrixXd::Identity(degreeOfFreedom,degreeOfFreedom);


    std::vector<Eigen::MatrixXd> Gradient;
    Gradient.resize(Npts);

    #pragma omp parallel for
    for(int i=0;i<Npts;++i)
    {
        const auto &neighbors=connectedNodes[i];
        int k=static_cast<int>(neighbors.size());
        assert(k>=1);

        Eigen::MatrixXd A(k,dim);
        for(int j=0;j<k;++j)
        {
            int idx=neighbors[j];
            A(j, 0) = nodes[idx].x;
            A(j, 1) = nodes[idx].y;
            if (dim == 3) 
            {
                A(j, 2) = nodes[idx].z;
            }
        }

        Eigen::RowVectorXd centroid=A.colwise().mean();
        Eigen::MatrixXd B=A.rowwise()-centroid;

        Eigen::MatrixXd G=pinv(B);

        Gradient[i]=Eigen::KroneckerProduct(G,I).eval();
    }

    return Gradient;
}


Eigen::MatrixXd EDISONN::materialProperty(double E, double nu, double thkns, const std::string& problemType, const std::string& planeType)
{
    // --- Plane elasticity: return 3×3 C matrix ---
    if (problemType == "plane_elasticity") 
    {
        Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
        double c33 = E / (2.0 * (1.0 + nu));
        if (planeType == "plate_stress") {
            double c11 = E / (1.0 - nu * nu);
            double c12 = nu * c11;
            C(0,0) = c11;  C(0,1) = c12;
            C(1,0) = c12;  C(1,1) = c11;
            C(2,2) = c33;
        }
        else if (planeType == "plane_strain") 
        {
            double del  = 1.0 - 3.0*nu*nu - 2.0*nu*nu*nu;
            double c11  = (1.0 - nu*nu) * E / del;
            double c12  = nu * (1.0 + nu) * E / del;
            C(0,0) = c11;  C(0,1) = c12;
            C(1,0) = c12;  C(1,1) = c11;
            C(2,2) = c33;
        }
        else 
        {
            throw std::invalid_argument("Unknown planeType: " + planeType);
        }

        // D0 (4×4) as per your MATLAB block
        Eigen::Matrix4d D;
        D << C(0,0),     0.0,     0.0, C(0,1),
                0.0,   C(2,2),  C(2,2),   0.0,
                0.0,   C(2,2),  C(2,2),   0.0,
             C(1,0),     0.0,     0.0, C(1,1);

        D *=thkns;

        return D;
    }

    // --- Plate (5-dof) model: return 10×10 stiffness matrix ---
    else if (problemType == "plate_model") 
    {
        // shear correction factor
        double Ks = 5.0/6.0;

        // build plane-stress C (3×3)
        Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
        double c11 = E / (1.0 - nu * nu);
        double c12 = nu * c11;
        double c33 = E / (2.0 * (1.0 + nu));
        C(0,0) = c11;  C(0,1) = c12;
        C(1,0) = c12;  C(1,1) = c11;
        C(2,2) = c33;

        // D0 (4×4) as per your MATLAB block
        Eigen::Matrix4d D0;
        D0 << C(0,0),     0.0,     0.0, C(0,1),
                0.0,   C(2,2),  C(2,2),   0.0,
                0.0,   C(2,2),  C(2,2),   0.0,
             C(1,0),     0.0,     0.0, C(1,1);

        // A = thkns * D0
        Eigen::Matrix4d A = thkns * D0;
        // D = (thkns^3 / 12) * D0
        Eigen::Matrix4d D = std::pow(thkns, 3) / 12.0 * D0;
        // G = thkns * Ks * C33 * I₂
        Eigen::Matrix2d G2 = thkns * Ks * C(2,2) * Eigen::Matrix2d::Identity();

        // Assemble 10×10 stiffness
        Eigen::MatrixXd Stiffness = Eigen::MatrixXd::Zero(10,10);

        // indices for A-block (MATLAB 1,2,6,7 → zero-based 0,1,5,6)
        std::array<int,4> ia = {0,1,5,6};
        for (int p = 0; p < 4; ++p)
            for (int q = 0; q < 4; ++q)
                Stiffness(ia[p], ia[q]) = A(p,q);

        // indices for D-block (MATLAB 4,5,9,10 → zero-based 3,4,8,9)
        std::array<int,4> id = {3,4,8,9};
        for (int p = 0; p < 4; ++p)
            for (int q = 0; q < 4; ++q)
                Stiffness(id[p], id[q]) = D(p,q);

        // indices for G-block (MATLAB 3,8 → zero-based 2,7)
        std::array<int,2> ig = {2,7};
        for (int p = 0; p < 2; ++p)
            for (int q = 0; q < 2; ++q)
                Stiffness(ig[p], ig[q]) = G2(p,q);

        return Stiffness;
    }

    // unknown problemType
    else {
        throw std::invalid_argument("Unknown problemType: " + problemType);
    }
}

std::pair<std::vector<Eigen::MatrixXd>,std::vector<Eigen::MatrixXd>> EDISONN::NodalStiffness(Eigen::MatrixXd& matProp,
    const std::vector<std::vector<int>>& adjacency,
    std::vector<Eigen::MatrixXd>& gradient,
    std::vector<double>& sharedVolume,int degreeOfFreedom)

{   
    int numNodes = static_cast<int>(adjacency.size());
    assert(static_cast<int>(gradient.size())    == numNodes);
    assert(static_cast<int>(sharedVolume.size())== numNodes);

    // Pre-allocate to numNodes size
    std::vector<Eigen::MatrixXd> elf_list(numNodes);
    std::vector<Eigen::MatrixXd> elk_list(numNodes);

    #pragma omp parallel for
    for(int n=0;n<numNodes;++n)
    {
        const auto& neigh = adjacency[n];      // connected nodes
        int h = static_cast<int>(neigh.size());
        assert(h>=1);

        const Eigen::MatrixXd& J=gradient[n]; // gradient operator for node n
        double w=sharedVolume[n];             // weight (shared area and/or volume)

        // Degrees of freedom for this node
        int ndof=h*degreeOfFreedom;

        // nodal internal force vector 
        Eigen::MatrixXd elf_per_node=Eigen::MatrixXd::Zero(ndof, 1);

        // nodal stiffness matrix: B'*C*B
        Eigen::MatrixXd elk_per_node=w*(J.transpose()*matProp*J);

        // Assign by index - thread safe
        elf_list[n] = std::move(elf_per_node);
        elk_list[n] = std::move(elk_per_node);
    }

    return {std::move(elf_list), std::move(elk_list)};
}