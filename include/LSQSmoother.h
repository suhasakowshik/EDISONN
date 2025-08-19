#ifndef LSQ_SMOOTHER_H
#define LSQ_SMOOTHER_H

#include<Eigen/Dense>
#include<Eigen/Cholesky>              
#include<vector>
#include<string>

class LSQSmoother
{
public:

    struct NodeData
    {
        int k; // number of neighbours
        int m; // number of basis functions
        Eigen::MatrixXd At; //m x k transpose of local basis
        Eigen::LLT<Eigen::MatrixXd> llt; // cholesky factor of (At * A)
    };

    /*
     * @brief Constructor: Precompute per-node factorization.
     * @param nodes   N×coordDim (2 or 3) nodal coordinates.
     * @param adj     adjacency list of size N.
     * @param fitType "linear" or "quadratic".
     */
    LSQSmoother(const Eigen::MatrixXd& nodes,const std::vector<std::vector<int>>& adj,
                const std::string& fitType);

    /*
     * @brief Apply smoothing to U_in.
     * @param U_in    (DOF·N)×1 flat displacement vector.
     * @param numIter Number of smoothing passes.
     * @param DOF     Degrees of freedom per node (e.g., 2, 3, 5).
     * @return        Smoothed (DOF·N)×1 vector.
    */
    Eigen::MatrixXd smooth(const Eigen::MatrixXd& U_in,int numIter,int DOF) const;
    
private:

    int N; //number of nodes
    int coordDim; //2D or 3D system
    std::string fitType; // linear or quadratic LSQ smoothing option
    std::vector<NodeData> data; // per nodes precomputed factors of A matrix in LSQ smoothing
    const std::vector<std::vector<int>>& adjacency; // immediate nodal neighbours

};

#endif // LSQ_SMOOTHER_H
