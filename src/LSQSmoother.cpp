#include "LSQSmoother.h"
#include <Eigen/Dense>
#include <stdexcept>

LSQSmoother::LSQSmoother(const Eigen::MatrixXd& nodes,const std::vector<std::vector<int>>& adj,
        const std::string& fit): N(nodes.rows()), coordDim(nodes.cols()),fitType(fit), data(N), adjacency(adj)

{
    if((coordDim!=2 && coordDim!=3) || (fitType!="linear" && fitType!="quadratic") || (adjacency.size()!=size_t(N)))
    {
        throw std::invalid_argument("Invalid Inputs to Least Squares smoother");
    }

    for(int i=0;i<N;i++)
    {
        const auto& neigh=adjacency[i];
        int k=neigh.size();
        if(k==0)
        {
            data[i].k=data[i].m=0;
            continue;
        }
    

        //Determine number of basis functions
        int m = (fitType == "linear")
              ? (1 + coordDim)       // [1, dx, dy, (dz)]
              : (coordDim == 2 ? 6   // [1, dx, dy, dx², dy², dx·dy]
                              : 10); // [1, dx,dy,dz, dx²,dy²,dz², dx·dy,dy·dz,dz·dx]

        // Build matrix A (k*m);
        Eigen::MatrixXd A(k,m);
        A.col(0).setOnes();

        if(fitType == "linear")
        {
            for(int j=0;j<k;++j)
            {
                for(int d=0;d<coordDim;++d)
                {
                    A(j,1+d)=nodes(neigh[j],d)-nodes(i,d);
                }
            }
        }
        else //quadratic
        {
            if(coordDim==2)
            {
                for(int j=0;j<k;++j)
                {
                    double dx=nodes(neigh[j],0)-nodes(i,0);
                    double dy=nodes(neigh[j],1)-nodes(i,1);
                    A(j,1)=dx;
                    A(j,2)=dy;
                    A(j,3)=dx*dx;
                    A(j,4)=dy*dy;
                    A(j,5)=dx*dy;
                }
            }
            else // 3d
            {   
                for(int j=0;j<k;++j)
                {
                    double dx=nodes(neigh[j],0)-nodes(i,0);
                    double dy=nodes(neigh[j],1)-nodes(i,1);
                    double dz=nodes(neigh[j],2)-nodes(i,2);
                    A(j,1)=dx;
                    A(j,2)=dy;
                    A(j,3)=dz;
                    A(j,4)=dx*dx;
                    A(j,5)=dy*dy;
                    A(j,6)=dz*dz;
                    A(j,7)=dx*dy; 
                    A(j,8)=dy*dz; 
                    A(j,9)=dz*dx;
                }
            }
        }

        auto& nd=data[i];
        nd.k=k;
        nd.m=m;
        nd.At=A.transpose();
        nd.llt.compute(nd.At*A);
        if (nd.llt.info()!=Eigen::Success) 
        {
            throw std::runtime_error("Cholesky factorization failed at node " + std::to_string(i));
        }
    }
}

Eigen::MatrixXd LSQSmoother::smooth(const Eigen::MatrixXd& U_in,int numIter,int DOF) const
{
    if(U_in.rows()!=N*DOF || U_in.cols()!=1)
    {
        throw std::invalid_argument("Input global displacement U size is mismatch");
    }

    Eigen::MatrixXd U_cur=U_in;
    Eigen::MatrixXd U_nxt=U_in;
    std::vector<Eigen::VectorXd> u_loc(DOF);

    for(int iter=0;iter<numIter;++iter)
    {
        for(int i=0;i<N;++i)
        {
            const NodeData& nd=data[i];
            if(nd.k==0)
            {
                //copy through if no neighbors
                U_nxt.block(i*DOF,0,DOF,1)=U_cur.block(i*DOF,0,DOF,1);
                continue;
            }

            //gather neighbourhood values
            for(int d=0;d<DOF;++d)
            {
                auto& vec=u_loc[d];
                vec.resize(nd.k);
                for(int j=0;j<nd.k;++j)
                {
                    vec(j)=U_cur(adjacency[i][j]*DOF+d,0);
                }
            }

            //solve for each DOF and take constant term
            for(int d=0;d<DOF;++d)
            {
                Eigen::VectorXd rhs=nd.At*u_loc[d];
                Eigen::VectorXd coeff=nd.llt.solve(rhs);
                U_nxt(i*DOF+d,0)=coeff(0);
            }
        }

        U_cur.swap(U_nxt);
    }

    return U_cur;
}