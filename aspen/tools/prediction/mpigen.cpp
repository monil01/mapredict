// Copyright 2013-2016 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>
#include <fstream>

#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "parser/Parser.h"
#include "walkers/RuntimeCounter.h"

#include <sys/time.h>

#include "walkers/AspenTool.h"
#include "app/ASTRequiresStatement.h"

void RankToIndex(int rank,
                int nx, int ny, int nz,
                int &i, int &j, int &k)
{
    i = rank % nx;
    j = (int)(rank / nx) % ny;
    k = (int)(rank / (nx*ny));
}

int IndexToRank(int nx, int ny, int nz,
                int i, int j, int k)
{
    return k*nx*ny + j*nx + i;
}

class MPIGenerator : public AspenTool
{
    ostream &out;
    string socket;
    TraversalMode mode;
  public:
    MPIGenerator(ostream &ostr,
                 ASTAppModel *app,
                 ASTMachModel *mach,
                 string socket,
                 TraversalMode m=Explicit)
        : AspenTool(app, mach), out(ostr), socket(socket), mode(m)
    {
        out << "#include <mpi.h>" << endl;
        out << "#include <cmath>" << endl;
        out << "#include <vector>" << endl;
        out << "#include <cstdlib>" << endl;
        out << "" << endl;
        out << "void RankToIndex(int rank," << endl;
        out << "                int nx, int ny, int nz," << endl;
        out << "                int &i, int &j, int &k)" << endl;
        out << "{" << endl;
        out << "    i = rank % nx;" << endl;
        out << "    j = (int)(rank / nx) % ny;" << endl;
        out << "    k = (int)(rank / (nx*ny));" << endl;
        out << "}" << endl;
        out << "" << endl;
        out << "int IndexToRank(int size, int nx, int ny, int nz," << endl;
        out << "                int i, int j, int k)" << endl;
        out << "{" << endl;
        out << "    int ri = (i+nx)%nx;" << endl;
        out << "    int rj = (j+ny)%ny;" << endl;
        out << "    int rk = (k+nz)%nz;" << endl;
        out << "    return rk*nx*ny + rj*nx + ri;" << endl;
        out << "}" << endl;
        out << "" << endl;
        out << "int main(int argc, char *argv[])" << endl;
        out << "{" << endl;
        out << "    MPI_Init(&argc, &argv);" << endl;
        out << "    srand(12345);" << endl;
        out << "    int rank, size;" << endl;
        out << "    MPI_Comm_rank(MPI_COMM_WORLD, &rank);" << endl;
        out << "    MPI_Comm_size(MPI_COMM_WORLD, &size);" << endl;
        out << "    int nx = ceil(pow(size,1./3.));" << endl;
        out << "    int ny = ceil(pow(size / nx,1./2.));" << endl;
        out << "    int nz = size / (nx*ny);" << endl;
        out << "    int total = nx*ny*nz;" << endl;
        out << "    if (rank==0)" << endl;
        out << "        printf(\"rank=%d size=%d nx=%d ny=%d nz=%d total=%d\\n\","
            <<                "rank,size,nx,ny,nz,total);" << endl;
        out << "    if (total != size) {" << endl;
        out << "         if (rank==0) printf(\"Didn't successfully factor tasks into nx/ny/nz, try a different -np.  (Sorry, the algorithm was done quickly and not robust.  Powers of 8 work great though!)\\n\");" << endl;
        out << "         return 1;" << endl;
        out << "    }" << endl;
        
    }
    ~MPIGenerator()
    {
        out << "    MPI_Finalize();" << endl;
        out << "}" << endl;
    }
  protected:
    virtual TraversalMode TraversalModeHint(const ASTControlStatement *here)
    {
        return mode;
    }
    virtual void StartIterate(TraversalMode mode, const ASTControlIterateStatement *s)
    {
        cerr << Indent(level) << "Starting iterate "<<s->GetLabel()<<"\n";
        if (mode == AspenTool::Explicit)
            return;

        int c = s->GetQuantity()->Expanded(app->paramMap)->Evaluate();
        out << "  for (int loopctr=0; loopctr<"<<c<<"; loopctr++) {" << endl;
    }
    virtual void EndIterate(TraversalMode mode, const ASTControlIterateStatement *s)
    {
        cerr << Indent(level) << "Ending iterate "<<s->GetLabel()<<"\n";
        if (mode == AspenTool::Explicit)
            return;
        out << "  }" << endl;
    }
    virtual void Execute(const ASTExecutionBlock *e)
    {
        cerr << Indent(level) << "Starting execution block "<<e->GetLabel()<<" (nres="<<e->GetStatements().size()<<")\n";

        for (unsigned int i=0; i<e->GetStatements().size(); ++i)
        {
            const ASTExecutionStatement *s = e->GetStatements()[i];
            const ASTRequiresStatement *req = dynamic_cast<const ASTRequiresStatement*>(s);
            if (!req)
                continue;

            // get the resource and traits
            string resource = req->GetResource();
            vector<string> traits;
            for (unsigned t = 0; t < req->GetNumTraits(); ++t)
                traits.push_back(req->GetTrait(t)->GetName());

            double val = req->GetQuantity()->Expanded(*paramStack)->Evaluate();
            double size = req->GetSize() ? req->GetSize()->Expanded(*paramStack)->Evaluate() : 4;
            double count = ceil(val / size);
            if (resource == "comm")
            {
                DoMessages(count, size,
                           traits.size()>0 ? traits[0] : "",
                           traits.size()>1 ? traits[1] : "",
                           traits.size()>2 ? traits[2] : "",
                           traits.size()>3 ? traits[3] : "");
            }
        }        
    }

    virtual void DoMessages(int count, int size, string type,
                            string mod1, string mod2, string mod3)
    {
        out << "   {" << endl;
        bool periodic = (mod1=="periodic"||mod2=="periodic"||mod3=="periodic");
        bool faceonly = (mod1=="face"||mod2=="face"||mod3=="face");
        bool poswavefront = (mod1=="pwave"||mod2=="pwave"||mod3=="pwave");
        bool negwavefront = (mod1=="nwave"||mod2=="nwave"||mod3=="nwave");
        bool nodal = (mod1=="nodal"||mod2=="nodal"||mod3=="nodal");

        if (type == "allreduce")
        {
            string op = "MPI_SUM";
            if (mod1 == "min")
                op = "MPI_MIN";
            else if (mod1 == "max")
                op = "MPI_MAX";
            else if (mod1 == "sum")
                op = "MPI_SUM";
            else if (mod1 == "prod")
                op = "MPI_PROD";
            out << "    {" << endl;
            int nwords = (count*size)/4;
            out << "      int nwords="<<nwords<<";\n";
            out << "      std::vector<float> sendvec(nwords, rank*1.0f);" << endl;
            out << "      std::vector<float> recvvec(nwords, -1.0f);" << endl;
            out << "      MPI_Allreduce(&(sendvec[0]), &(recvvec[0]), "<<nwords<<", MPI_FLOAT,"<<op<<", MPI_COMM_WORLD);\n";
            out << "    }" << endl;
        }
        else if (type == "random")
        {
            out << "    {" << endl;
            int nwords = (count*size)/4;
            out << "      int nwords="<<nwords<<";\n";
            out << "      std::vector<float> sendvec(nwords, rank*1.0f);" << endl;
            out << "      std::vector<float> recvvec(nwords, -1.0f);" << endl;
            out << "      std::vector<MPI_Request> srq;" << endl;
            out << "      std::vector<MPI_Request> rrq;" << endl;
            out << "      MPI_Status status;\n";
            out << "      MPI_Status statusarray[100];\n";
            out << "      for (int i=0; i<size; ++i)" << endl;
            out << "      {" << endl;
            out << "          int src = i;" << endl;
            out << "          int dest = rand() % size;" << endl;
            out << "          if (src == dest)" << endl;
            out << "              continue;" << endl;
            out << "          if (rank == src)" << endl;
            out << "          {" << endl;
            out << "              srq.push_back(MPI_Request());" << endl;
            out << "              MPI_Isend(&sendvec[0], nwords, MPI_FLOAT, dest, "
                <<               "0, MPI_COMM_WORLD, &(srq.back()));\n";
            out << "              printf(\""<<type<<": rank=%d src=%d dest=%d\\n\",rank,src,dest);\n";
            out << "          }" << endl;
            out << "          if (rank == dest)" << endl;
            out << "          {" << endl;
            out << "              rrq.push_back(MPI_Request());" << endl;
            out << "              MPI_Irecv(&recvvec[0], nwords, MPI_FLOAT, src, "
                <<                         "MPI_ANY_TAG, MPI_COMM_WORLD, &(rrq.back()));\n";
            out << "              printf(\"rank=%d sendvec[0]=%d recvvec[0]=%d\\n\",rank,int(sendvec[0]),int(recvvec[0]));" << endl;
            out << "          }" << endl;
            out << "      }" << endl;
            //out << "      for (int i=0; i<srq.size(); ++i)" << endl;
            //out << "          MPI_Wait(&(srq[i]), &status);\n";
            out << "      MPI_Waitall(srq.size(), &(srq[0]), statusarray);\n";
            out << "      for (int i=0; i<rrq.size(); ++i)" << endl;
            out << "          MPI_Wait(&(rrq[i]), &status);\n";
            out << "    }" << endl;
        }
        else if (type == "nn3d")
        {
            out << "    int i,j,k;" << endl;
            out << "    RankToIndex(rank,nx,ny,nz, i,j,k);" << endl;
            out << "    MPI_Request srqarr[100];" << endl;
            out << "    MPI_Request rrqarr[100];" << endl;
            out << "    int srq_count = 0;" << endl;
            out << "    int rrq_count = 0;" << endl;
            out << "    MPI_Status status;\n";
            out << "    MPI_Status statusarr[100];\n";
            out << "    int nn = " << count << ";" << endl;
            int numdirs = faceonly ? 6 : 26;
            for (int dir = 0 ; dir < numdirs ; dir++)
            {
                string si,sj,sk; // source indices
                string di,dj,dk; // dest indices
                string nitems;
                int nodalcount;
                if (dir < 6) 
                {
                    nitems = "nn*nn";
                    nodalcount = 4;
                }
                else if (dir < 18)
                {
                    nitems = "nn";
                    nodalcount = 2;
                }
                else
                {
                    nitems = "1";
                    nodalcount = 1;
                }
 
                switch (dir)
                {
                  // faces

                  case 0: // nx
                      {
                          si = "i+1"; sj="j";   sk="k";
                          di = "i-1"; dj="j";   dk="k";
                          break;
                      }
                  case 1: // px
                      {
                          si = "i-1"; sj="j";   sk="k";
                          di = "i+1"; dj="j";   dk="k";
                          break;
                      }
                  case 2: // ny
                      {
                          si = "i";   sj="j+1"; sk="k";
                          di = "i";   dj="j-1"; dk="k";
                          break;
                      }
                  case 3: // py
                      {
                          si = "i";   sj="j-1"; sk="k";
                          di = "i";   dj="j+1"; dk="k";
                          break;
                      }
                  case 4: // nz
                      {
                          si = "i";   sj="j";   sk="k+1";
                          di = "i";   dj="j";   dk="k-1";
                          break;
                      }
                  case 5: // pz
                      {
                          si = "i";   sj="j";   sk="k-1";
                          di = "i";   dj="j";   dk="k+1";
                          break;
                      }

                  // edges

                  case 6: // nxny
                      {
                          si = "i+1"; sj="j+1"; sk="k";
                          di = "i-1"; dj="j-1"; dk="k";
                          break;
                      }
                  case 7: // nxpy
                      {
                          si = "i+1"; sj="j-1"; sk="k";
                          di = "i-1"; dj="j+1"; dk="k";
                          break;
                      }
                  case 8: // pxny
                      {
                          si = "i-1"; sj="j+1"; sk="k";
                          di = "i+1"; dj="j-1"; dk="k";
                          break;
                      }
                  case 9: // pxpy
                      {
                          si = "i-1"; sj="j-1"; sk="k";
                          di = "i+1"; dj="j+1"; dk="k";
                          break;
                      }


                  case 10: // nxnz
                      {
                          si = "i+1"; sj="j";   sk="k+1";
                          di = "i-1"; dj="j";   dk="k-1";
                          break;
                      }
                  case 11: // nxpz
                      {
                          si = "i+1"; sj="j";   sk="k-1";
                          di = "i-1"; dj="j";   dk="k+1";
                          break;
                      }
                  case 12: // pxnz
                      {
                          si = "i-1"; sj="j";   sk="k+1";
                          di = "i+1"; dj="j";   dk="k-1";
                          break;
                      }
                  case 13: // pxpz
                      {
                          si = "i-1"; sj="j";   sk="k-1";
                          di = "i+1"; dj="j";   dk="k+1";
                          break;
                      }


                  case 14: // nynz
                      {
                          si = "i";   sj="j+1"; sk="k+1";
                          di = "i";   dj="j-1"; dk="k-1";
                          break;
                      }
                  case 15: // nypz
                      {
                          si = "i";   sj="j+1"; sk="k-1";
                          di = "i";   dj="j-1"; dk="k+1";
                          break;
                      }
                  case 16: // pynz
                      {
                          si = "i";   sj="j-1"; sk="k+1";
                          di = "i";   dj="j+1"; dk="k-1";
                          break;
                      }
                  case 17: // pypz
                      {
                          si = "i";   sj="j-1"; sk="k-1";
                          di = "i";   dj="j+1"; dk="k+1";
                          break;
                      }

                  // corners
                  case 18: // nxnynz
                      {
                          si = "i+1"; sj="j+1"; sk="k+1";
                          di = "i-1"; dj="j-1"; dk="k-1";
                          break;
                      }
                  case 19: // nxnypz
                      {
                          si = "i+1"; sj="j+1"; sk="k-1";
                          di = "i-1"; dj="j-1"; dk="k+1";
                          break;
                      }
                  case 20: // nxpynz
                      {
                          si = "i+1"; sj="j-1"; sk="k+1";
                          di = "i-1"; dj="j+1"; dk="k-1";
                          break;
                      }
                  case 21: // nxpypz
                      {
                          si = "i+1"; sj="j-1"; sk="k-1";
                          di = "i-1"; dj="j+1"; dk="k+1";
                          break;
                      }
                  case 22: // pxnynz
                      {
                          si = "i-1"; sj="j+1"; sk="k+1";
                          di = "i+1"; dj="j-1"; dk="k-1";
                          break;
                      }
                  case 23: // pxnypz
                      {
                          si = "i-1"; sj="j+1"; sk="k-1";
                          di = "i+1"; dj="j-1"; dk="k+1";
                          break;
                      }
                  case 24: // pxpynz
                      {
                          si = "i-1"; sj="j-1"; sk="k+1";
                          di = "i+1"; dj="j+1"; dk="k-1";
                          break;
                      }
                  case 25: // pxpypz
                      {
                          si = "i-1"; sj="j-1"; sk="k-1";
                          di = "i+1"; dj="j+1"; dk="k+1";
                          break;
                      }


                }
                out << "    {\n";
                out << "      //printf(\"rank=%02d i=%d j=%d k=%d\\n\",rank,i,j,k);" << endl;

                out << "      int si="<<si<<", sj="<<sj<<", sk="<<sk<<";\n";
                out << "      int di="<<di<<", dj="<<dj<<", dk="<<dk<<";\n";
                out << "      int src  = IndexToRank(size, nx,ny,nz, si,sj,sk);" << endl;
                out << "      int dest = IndexToRank(size, nx,ny,nz, di,dj,dk);" << endl;
                out << "      bool periodic = " << periodic << ";" << endl;
                out << "      bool poswavefront = " << poswavefront << ";" << endl;
                out << "      bool negwavefront = " << negwavefront << ";" << endl;
                out << "      int nodalcount = " << (nodal ? nodalcount : 1) << ";" << endl;
                out << "      int nn = " << count << ";" << endl;
                out << "      int nitems=nodalcount * "<<nitems<<" * "<<size<<" / sizeof(double);\n";
                out << "      double *sendvec = new double[nitems]; // set to rank*1.0f;" << endl;
                out << "      double *recvvec = new double[nitems]; // set to -1.0f;" << endl;
                out << "      bool send = true;"<<endl;
                out << "      send = send && (periodic || ((di>=0 && di<nx) && (dj>=0 && dj<ny) && (dk>=0 && dk<nz)));" << endl;
                out << "      send = send && (!poswavefront || (dest>rank));" << endl;
                out << "      send = send && (!negwavefront || (dest<rank));" << endl;
                out << "      bool recv = true;"<<endl;
                out << "      recv = recv && (periodic || ((si>=0 && si<nx) && (sj>=0 && sj<ny) && (sk>=0 && sk<nz)));" << endl;
                out << "      recv = recv && (!poswavefront || (src<rank));" << endl;
                out << "      recv = recv && (!negwavefront || (src>rank));" << endl;
                out << "      //printf(\""<<type<<" dir="<<dir<<": rank=%d src=%d recv=%s\\n\",rank,src,recv?\"YES\":\"no\");\n";
                out << "      //printf(\""<<type<<" dir="<<dir<<": rank=%d dst=%d send=%s\\n\",rank,dest,send?\"YES\":\"no\");\n";

                out << "      if (send)" << endl;
                out << "      {" << endl;
#if 1
                out << "        MPI_Isend(&sendvec[0], nitems, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &(srqarr[srq_count]));\n";
                out << "        srq_count++;\n";
#else
                out << "        MPI_Isend(&sendvec[0], nitems, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &srq);\n";
#endif
                out << "      }" << endl;

                out << "      if (recv)" << endl;
                out << "      {" << endl;
#if 1
                out << "        MPI_Irecv(&recvvec[0], nitems, MPI_DOUBLE, src, MPI_ANY_TAG, MPI_COMM_WORLD, &(rrqarr[rrq_count]));\n";
                out << "        rrq_count++;\n";
#else
                out << "        MPI_Irecv(&recvvec[0], nitems, MPI_DOUBLE, src, MPI_ANY_TAG, MPI_COMM_WORLD, &rrq);\n";
#endif

                out << "      }" << endl;

#if 1
#else
                out << "      if (send)" << endl;
                out << "      {" << endl;
                out << "        MPI_Wait(&srq, &status);\n";
                out << "      }" << endl;
#endif

#if 1
#else
                out << "      if (recv)" << endl;
                out << "      {" << endl;
                out << "        MPI_Wait(&rrq, &status);\n";
                out << "      }" << endl;
                out << "      //printf(\"rank=%d sendvec[0]=%d recvvec[0]=%d\\n\",rank,int(sendvec[0]),int(recvvec[0]));" << endl;
#endif

                out << "    }\n";
            }

            //
            // send waits
            //
#if 1
            // send waitall
            out << "    MPI_Waitall(srq_count, srqarr, statusarr);\n";
#else
            // send individual wait
            out << "    for (int p=0; p<srq_count; p++) MPI_Wait(&(srqarr[p]), &status);\n";
#endif

            //
            // receive waits
            //
#if 0
            // receive waitall
            out << "    MPI_Waitall(rrq_count, rrqarr, statusarr);\n";
#else
            // receive individual wait
            out << "    for (int p=0; p<rrq_count; p++) MPI_Wait(&(rrqarr[p]), &status);\n";
#endif
        }
        else
        {
            out << "    // error -- unknown type for message\n";
        }
        out << "   }" << endl;

    }
        
};

int main(int argc, char **argv)
{
    try {
        ASTAppModel *app = NULL;
        ASTMachModel *mach = NULL;
        if (argc >= 3)
        {
            app = LoadAppModel(argv[1]);
            mach = LoadMachineModel(argv[2]);
        }

        if (argc != 4 || !app || !mach)
        {
            cerr << "Usage: "<<argv[0]<<" [app.aspen] [mach.aspen] [socket]" << endl;
            return 1;
        }

        string socket = argv[3];

        cerr << "Parsed; calculating runtime\n";

        struct timeval tv;
        gettimeofday(&tv,NULL);
        srand(tv.tv_usec);

        std::ofstream mpicode("mpitest.cpp");
        MPIGenerator *t = new MPIGenerator(mpicode, app, mach, socket, AspenTool::Implicit);
	t->InitializeTraversal();
        app->kernelMap["main"]->Traverse(t);
        delete t; // must call destructor for now
        mpicode.close();
    }
    catch (const AspenException &exc)
    {
        cerr << exc.PrettyString() << endl;
        return -1;
    }
}
